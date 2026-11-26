"""Построение индекса: PDF → страницы → фрагменты → эмбеддинги.

PDF и манифест — в KB_DIR, результат — в INDEX_DIR; оба каталога вне репозитория и образа.
Файлы в INDEX_DIR:
- pages/{doc_id}.json      — кеш извлечённых страниц (ключ — хэш PDF + версия извлечения);
- ocr/{doc_id}/p{NNN}.txt  — распознанные vision-моделью сканы (scripts/ocr_scans.py);
- chunks.jsonl             — фрагменты с метаданными;
- embeddings.npy           — матрица эмбеддингов в порядке chunks.jsonl (если модель настроена);
- emb_cache.npz            — кеш эмбеддингов по хэшу текста: повторная сборка не платит заново;
- manifest.json            — что и когда построено.
BM25 строится при загрузке из chunks.jsonl (это доли секунды) и на диск не пишется.
"""
import hashlib
import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import Settings
from app.retrieval.chunk import Chunk, chunk_document
from app.retrieval.extract import Line, PageText, extract_pdf
from app.integrations.fleet import load_engine_families
from app.retrieval.manuals import Manual, check_families_known, load_manuals

EXTRACT_VERSION = 3      # поменять при изменении extract.py — кеш страниц пересоберётся
EMBED_BATCH = 64


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()[:24]


def load_pages(manual: Manual, settings: Settings, *, log=print) -> list[PageText]:
    """Страницы документа из кеша или PDF; проверяет число страниц по справочнику;
    подставляет распознанные сканы из index/ocr/ (source=ocr)."""
    pdf = settings.require_kb_dir() / manual.file
    cache = settings.require_index_dir() / "pages" / f"{manual.doc_id}.json"
    pdf_hash = _file_hash(pdf)
    if manual.sha256 and manual.sha256.lower() != pdf_hash:
        raise ValueError(f"{manual.doc_id}: sha256 файла {manual.file} не совпадает с kb.yaml")
    # в ключе — PDF, версия кода извлечения и правила документа из справочника
    rules = manual.model_dump_json(include={"page_number_offset", "strip_patterns", "title_band"})
    key = f"{pdf_hash[:16]}:{EXTRACT_VERSION}:{hashlib.sha256(rules.encode()).hexdigest()[:8]}"
    pages: list[PageText] | None = None
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("key") == key:
            pages = [PageText(pdf_page=p["pdf_page"], lines=[Line(**l) for l in p["lines"]],
                              title=p.get("title"), source=p.get("source", "text"))
                     for p in data["pages"]]
    if pages is None:
        started = time.perf_counter()
        pages = extract_pdf(pdf, manual)
        log(f"  {manual.doc_id}: извлечено {len(pages)} стр. за {time.perf_counter() - started:.0f} с")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "pages": [asdict(p) for p in pages]},
                                    ensure_ascii=False), encoding="utf-8")
    if len(pages) != manual.pages:
        raise ValueError(f"{manual.doc_id}: в PDF {len(pages)} стр., в справочнике {manual.pages}")

    ocr_dir = settings.require_index_dir() / "ocr" / manual.doc_id
    for n in manual.scan_pages:
        # номер страницы — только из справочника scan_pages, имя файла его не задаёт
        txt = ocr_dir / f"p{n:03d}.txt"
        if txt.exists():
            pages[n - 1] = ocr_page(manual, n, txt.read_text(encoding="utf-8"), title=pages[n - 1].title)
    if ocr_dir.exists():
        expected = {f"p{n:03d}.txt" for n in manual.scan_pages}
        for extra in sorted(p.name for p in ocr_dir.glob("*.txt") if p.name not in expected):
            log(f"  {manual.doc_id}: {extra} не входит в scan_pages — пропущен")
    return pages


def ocr_page(manual: Manual, n: int, text: str, *, title: str | None) -> PageText:
    """Страница из распознанного текста. Служебные строки вырезает код, а не промпт:
    напечатанный номер этой страницы и колонтитулы по strip_patterns (позиции на скане нет,
    поэтому без полос — по точному совпадению всей строки)."""
    patterns = [re.compile(p) for p in manual.strip_patterns]
    expected_no = None if manual.page_number_offset is None else str(n + manual.page_number_offset)
    lines = []
    for raw in text.splitlines():
        t = raw.strip()
        if not t or t == expected_no or any(p.search(t) for p in patterns):
            continue
        lines.append(Line(text=t, y0=0, y1=0, size=0, bold=False))
    return PageText(pdf_page=n, lines=lines, title=title, source="ocr")


def build_chunks(settings: Settings, *, log=print) -> list[Chunk]:
    chunks: list[Chunk] = []
    manuals = load_manuals(settings.require_kb_dir())
    check_families_known(manuals, set(load_engine_families(settings.config_dir).values()))
    for manual in manuals:
        pages = load_pages(manual, settings, log=log)
        doc_chunks = chunk_document(manual, pages)
        log(f"  {manual.doc_id}: {len(doc_chunks)} фрагментов")
        chunks.extend(doc_chunks)
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk_id не уникальны")
    return chunks


def embedding_input(chunk: Chunk) -> str:
    """Текст для эмбеддинга: контекст (документ, раздел, карта) + текст фрагмента."""
    head = " / ".join(x for x in (chunk.manual, chunk.title, chunk.section) if x)
    return f"{head}\n{chunk.text}"


def build_embeddings(chunks: list[Chunk], settings: Settings, *, log=print) -> np.ndarray | None:
    if not settings.embedding_model or settings.openai_api_key is None:
        log("  эмбеддинги пропущены: EMBEDDING_MODEL или OPENAI_API_KEY не задан (поиск будет только BM25)")
        return None
    from app.integrations.llm import embed, get_client

    cache_path = settings.require_index_dir() / "emb_cache.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path)
        cache = dict(zip(data["keys"].tolist(), data["vectors"]))
    inputs = [embedding_input(c) for c in chunks]
    keys = [text_hash(t, settings.embedding_model) for t in inputs]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    log(f"  эмбеддинги: в кеше {len(chunks) - len(missing)}, считаем {len(missing)}")
    client = get_client(settings) if missing else None
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start:start + EMBED_BATCH]
        vectors = embed(client, settings, [inputs[i] for i in batch], scenario="indexing", step="chunk_embedding")
        for i, vec in zip(batch, vectors):
            cache[keys[i]] = np.asarray(vec, dtype=np.float32)
        np.savez(cache_path, keys=np.array(list(cache.keys())), vectors=np.stack(list(cache.values())))
    matrix = np.stack([cache[k] for k in keys]).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix


def build_index(settings: Settings, *, embeddings: bool = True, log=print) -> dict:
    started = time.perf_counter()
    index_dir = settings.require_index_dir()
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(settings, log=log)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    matrix = build_embeddings(chunks, settings, log=log) if embeddings else None
    emb_file = index_dir / "embeddings.npy"
    if matrix is not None:
        np.save(emb_file, matrix)
    elif emb_file.exists():
        emb_file.unlink()   # старые эмбеддинги не соответствуют новым фрагментам
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": len(chunks),
        "by_doc": {m.doc_id: sum(c.doc_id == m.doc_id for c in chunks) for m in load_manuals(settings.require_kb_dir())},
        "ocr_chunks": sum(c.source == "ocr" for c in chunks),
        "embedding_model": settings.embedding_model if matrix is not None else None,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
