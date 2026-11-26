"""Гибридный поиск по руководствам: BM25 + эмбеддинги, слияние reciprocal rank fusion.

Правило 3 CLAUDE.md: поиск всегда ограничен документами двигателя выбранной машины.
- `engine_family` — обязательный параметр `search()`; его подставляет код из карточки ТС,
  модель на него не влияет (у инструмента агента этого аргумента не будет).
- Ограничение действует ДО ранжирования: BM25 строится отдельно на каждое семейство,
  косинусная близость считается только по строкам матрицы этого семейства. Фрагмент
  чужого двигателя не может попасть в выдачу даже теоретически.
- Неизвестное семейство → ValueError, а не «поиск по всем документам».

Если эмбеддингов нет (модель не настроена или индекс собран без них) или запрос эмбеддинга
упал — поиск работает только на BM25 и сообщает об этом в `SearchResult.mode`.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.retrieval.manuals import manuals_by_id
from app.storage.events import log_event

RRF_K = 60            # стандартная константа RRF
CANDIDATES = 50       # сколько лучших из каждого метода участвуют в слиянии
DEFAULT_K = 5

_TOKEN = re.compile(r"[a-zа-я0-9]+(?:[.,]\d+)?", re.IGNORECASE)
# Частые окончания русских слов — грубый стеммер без внешних зависимостей:
# «зазоры», «зазоров», «зазора» → «зазор»; «клапанов» → «клапан».
_ENDINGS = tuple(sorted((
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "иях", "ах", "ях", "ов", "ев", "ей",
    "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю", "ом", "ем", "ам", "ям",
    "ия", "ие", "ии", "ию", "ье", "ья", "ть", "ся", "а", "я", "о", "е", "ы", "и", "у", "ю", "ь",
), key=len, reverse=True))
_STOP = frozenset("и в во на с со по для от до из к ко не при или а но что как это то же ли".split())


def _stem(token: str) -> str:
    if token[0].isdigit() or len(token) <= 4:
        return token
    for ending in _ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 4:
            return token[: -len(ending)]
    return token


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    return [_stem(t) for t in _TOKEN.findall(text) if t not in _STOP]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    manual: str
    engine_family: str
    pdf_page: int
    section: str | None
    title: str | None
    text: str
    is_table: bool
    is_fault_table: bool
    source: str
    score: float
    bm25_rank: int | None
    vector_rank: int | None


@dataclass
class SearchResult:
    query: str
    engine_family: str
    mode: str                         # hybrid | bm25
    hits: list[SearchHit] = field(default_factory=list)


class _FamilyIndex:
    def __init__(self, rows: list[int], chunks: list[dict]):
        self.rows = rows                                  # номера строк в общей матрице
        self.bm25 = BM25Okapi([tokenize(_bm25_text(chunks[i])) for i in rows])


def _bm25_text(chunk: dict) -> str:
    head = " ".join(x for x in (chunk.get("title"), chunk.get("section")) if x)
    return f"{head}\n{chunk['text']}"


class SearchIndex:
    """Индекс из папки index/. Загружается один раз; BM25 строится при загрузке."""

    def __init__(self, index_dir: Path):
        index_dir = Path(index_dir)
        chunks_file = index_dir / "chunks.jsonl"
        if not chunks_file.exists():
            raise FileNotFoundError(f"Нет индекса {chunks_file}: запустите scripts/build_index.py")
        with chunks_file.open(encoding="utf-8") as f:
            self.chunks: list[dict] = [json.loads(line) for line in f if line.strip()]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        manifest_file = index_dir / "manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
        self.embedding_model: str | None = manifest.get("embedding_model")
        emb_file = index_dir / "embeddings.npy"
        self.embeddings: np.ndarray | None = None
        if self.embedding_model and emb_file.exists():
            matrix = np.load(emb_file)
            if matrix.shape[0] == len(self.chunks):
                self.embeddings = matrix
        families: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            families.setdefault(c["engine_family"], []).append(i)
        self.families = {fam: _FamilyIndex(rows, self.chunks) for fam, rows in families.items()}

    def get(self, chunk_id: str) -> dict | None:
        return self.by_id.get(chunk_id)

    def bm25_ranking(self, query: str, engine_family: str) -> list[int]:
        fam = self._family(engine_family)
        scores = fam.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:CANDIDATES]
        return [fam.rows[j] for j in order if scores[j] > 0]

    def vector_ranking(self, query_vector: np.ndarray, engine_family: str) -> list[int]:
        fam = self._family(engine_family)
        assert self.embeddings is not None
        q = np.asarray(query_vector, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        sims = self.embeddings[fam.rows] @ q
        order = np.argsort(-sims)[:CANDIDATES]
        return [fam.rows[j] for j in order]

    def _family(self, engine_family: str) -> _FamilyIndex:
        fam = self.families.get(engine_family)
        if fam is None:
            raise ValueError(f"Нет документов для семейства двигателя {engine_family!r}")
        return fam


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
    return scores


_index_cache: dict[Path, tuple[tuple, SearchIndex]] = {}
_query_vectors: dict[tuple[str, str], np.ndarray] = {}


def get_index(settings: Settings) -> SearchIndex:
    """Индекс из settings.index_dir; перечитывается, если пересобраны фрагменты или эмбеддинги.
    При загрузке семейство каждого фрагмента сверяется со справочником руководств: индекс,
    собранный по старому справочнику, не должен незаметно ослабить фильтр по двигателю."""
    files = ("chunks.jsonl", "embeddings.npy", "manifest.json")
    stamp = tuple((settings.index_dir / f).stat().st_mtime if (settings.index_dir / f).exists() else 0.0
                  for f in files)
    cached = _index_cache.get(settings.index_dir)
    if cached is None or cached[0] != stamp:
        index = SearchIndex(settings.index_dir)
        check_families(index, manuals_by_id(settings.config_dir))
        cached = (stamp, index)
        _index_cache[settings.index_dir] = cached
    return cached[1]


def check_families(index: SearchIndex, manuals: dict) -> None:
    for c in index.chunks:
        manual = manuals.get(c["doc_id"])
        if manual is None or manual.engine_family != c["engine_family"] or manual.manual != c["manual"]:
            raise ValueError(f"Индекс не соответствует config/manuals.json ({c['chunk_id']}): "
                             "пересоберите индекс scripts/build_index.py")


def embed_query(query: str, settings: Settings, *, scenario: str) -> np.ndarray | None:
    """Эмбеддинг запроса (с кешем в памяти процесса). None — эмбеддинги недоступны."""
    model = settings.embedding_model
    if not model or settings.openai_api_key is None:
        return None
    key = (model, query)
    if key not in _query_vectors:
        from app.integrations.llm import embed, get_client
        vector = embed(get_client(settings), settings, [query], scenario=scenario, step="query_embedding")[0]
        _query_vectors[key] = np.asarray(vector, dtype=np.float32)
    return _query_vectors[key]


def search(query: str, engine_family: str, settings: Settings, *, k: int = DEFAULT_K,
           scenario: str = "question", index: SearchIndex | None = None) -> SearchResult:
    """Лучшие k фрагментов по запросу среди документов семейства engine_family."""
    if not engine_family:
        raise ValueError("engine_family обязателен: поиск без фильтра по двигателю запрещён")
    index = index or get_index(settings)
    rankings = [index.bm25_ranking(query, engine_family)]
    mode = "bm25"
    if index.embeddings is not None and index.embedding_model == settings.embedding_model:
        try:
            vector = embed_query(query, settings, scenario=scenario)
        except Exception as exc:      # сбой провайдера не роняет поиск — остаётся BM25
            vector = None
            log_event(settings.db_path, "error", data={
                "where": "query_embedding", "error": type(exc).__name__, "fallback": "bm25"})
        if vector is not None:
            rankings.append(index.vector_ranking(vector, engine_family))
            mode = "hybrid"
    fused = rrf(rankings)
    ranks = [{row: r for r, row in enumerate(ranking, start=1)} for ranking in rankings]
    top = sorted(fused, key=lambda row: -fused[row])[:k]
    hits = []
    for row in top:
        c = index.chunks[row]
        if c["engine_family"] != engine_family:      # страховка: не должно случаться никогда
            raise AssertionError(f"фрагмент {c['chunk_id']} не из семейства {engine_family}")
        hits.append(SearchHit(
            chunk_id=c["chunk_id"], doc_id=c["doc_id"], manual=c["manual"],
            engine_family=c["engine_family"], pdf_page=c["pdf_page"], section=c.get("section"),
            title=c.get("title"), text=c["text"], is_table=c["is_table"],
            is_fault_table=c["is_fault_table"], source=c["source"], score=round(fused[row], 5),
            bm25_rank=ranks[0].get(row), vector_rank=ranks[1].get(row) if len(ranks) > 1 else None,
        ))
    return SearchResult(query=query, engine_family=engine_family, mode=mode, hits=hits)
