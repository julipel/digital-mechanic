"""Распознавание страниц-сканов vision-моделью (VISION_MODEL из .env) — один раз при индексации.

Какие страницы — поле scan_pages в манифесте KB_DIR/kb.yaml. Результат кешируется в
INDEX_DIR/ocr/{doc_id}/p{NNN}.txt и при следующей сборке индекса попадает во фрагменты
с source=ocr. Цифры из распознанных таблиц надо сверить с картинкой вручную.

    python scripts/ocr_scans.py            # распознать то, чего ещё нет в кеше
    python scripts/ocr_scans.py --force    # распознать заново
    python scripts/build_index.py          # затем пересобрать индекс
"""
import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.integrations.llm import LLMNotConfigured, get_client  # noqa: E402
from app.retrieval.manuals import load_manuals  # noqa: E402
from app.storage.events import log_llm_call  # noqa: E402

PROMPT = (
    "Это скан страницы технического руководства по ремонту двигателя. Перепиши весь текст "
    "страницы на русском дословно, без пересказа и без дополнений. Таблицы — построчно: "
    "одна строка таблицы на строку текста, ячейки через « | », сначала строка шапки. "
    "Если подписи столбцов многоуровневые — повтори их полностью в шапке. Числа, единицы, "
    "обозначения моделей двигателей и номера деталей переписывай точно как на скане. "
    "Неразборчивое место пометь [неразборчиво], не угадывай. Номер страницы не переписывай. "
    "Ответ — только текст страницы."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Распознавание сканов руководств")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.vision_model:
        raise LLMNotConfigured("VISION_MODEL не задан в .env")
    client = get_client(settings)
    for manual in load_manuals(settings.require_kb_dir()):
        if not manual.scan_pages:
            continue
        out_dir = settings.require_index_dir() / "ocr" / manual.doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(settings.require_kb_dir() / manual.file) as doc:
            for n in manual.scan_pages:
                out = out_dir / f"p{n:03d}.txt"
                if out.exists() and not args.force:
                    print(f"{manual.doc_id} стр. {n}: уже распознана")
                    continue
                png = doc[n - 1].get_pixmap(dpi=200).tobytes("png")
                url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                started = time.perf_counter()
                response = client.chat.completions.create(
                    model=settings.vision_model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                    ]}],
                )
                usage = response.usage
                details = getattr(usage, "prompt_tokens_details", None)
                log_llm_call(
                    settings.db_path, scenario="indexing", step="scan_ocr", model=settings.vision_model,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    cached_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                out.write_text(response.choices[0].message.content or "", encoding="utf-8")
                print(f"{manual.doc_id} стр. {n}: готово")


if __name__ == "__main__":
    main()
