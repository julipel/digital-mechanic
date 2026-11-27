"""Распознавание страниц-сканов vision-моделью (VISION_MODEL из .env) — один раз при индексации.

Какие страницы — поле scan_pages в манифесте KB_DIR/kb.yaml: сканы без текста и страницы с испорченным
текстовым слоем (слова разорваны на куски с повтором букв — 236.01.01 РК). Страница, где текст набран
поперёк листа, перед отправкой разворачивается. Результат кешируется в
INDEX_DIR/ocr/{doc_id}/p{NNN}.txt и при следующей сборке индекса попадает во фрагменты
с source=ocr. Цифры сверяются автоматически по текстовому слою: scripts/check_ocr_digits.py;
вручную — только расхождения из его отчёта.

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
from app.retrieval.extract import upright_rotation  # noqa: E402
from app.retrieval.manuals import load_manuals  # noqa: E402
from app.retrieval.ocr_check import MIN_LAYER_DIGITS, digits  # noqa: E402
from app.storage.events import log_llm_call  # noqa: E402

PROMPT = (
    "Это скан страницы технического руководства по ремонту двигателя. Перепиши весь текст "
    "страницы на русском дословно, без пересказа и без дополнений. Таблицы — построчно: "
    "одна строка таблицы на строку текста, ячейки через « | », сначала строка шапки. "
    "Если подписи столбцов многоуровневые — повтори их полностью в шапке. Числа, единицы, "
    "обозначения моделей двигателей и номера деталей переписывай точно как на скане. "
    "Неразборчивое место пометь [неразборчиво], не угадывай. Номер страницы не переписывай. "
    "Если на картинке нет текста, ответь пустой строкой — ничего не дописывай от себя. "
    "Ответ — только текст страницы."
)


def recognize(client, settings, page: pymupdf.Page, *, dpi: int, clip: pymupdf.Rect | None, step: str) -> str:
    """Одна картинка → текст. Повёрнутая таблица разворачивается; вызов пишется в llm_calls."""
    matrix = pymupdf.Matrix(dpi / 72, dpi / 72).prerotate(upright_rotation(page))
    png = page.get_pixmap(matrix=matrix, clip=clip).tobytes("png")
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
        settings.db_path, scenario="indexing", step=step, model=settings.vision_model,
        input_tokens=usage.prompt_tokens if usage else 0,
        cached_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return response.choices[0].message.content or ""


def halves(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Верх и низ страницы с перекрытием 10% — цифры в таблицах крупнее, чем на целой странице.
    Пустая половина в модель не отправляется: на пустой картинке модель «распознала» текст из другого
    руководства (РЭ ЯМЗ-236 стр. 57 → строки ЗМЗ-409 с номерами деталей)."""
    r = page.rect
    clips = [pymupdf.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * 0.55),
             pymupdf.Rect(r.x0, r.y0 + r.height * 0.45, r.x1, r.y1)]
    return [c for c in clips if not is_blank(page, c)]


def is_blank(page: pymupdf.Page, clip: pymupdf.Rect, *, dark_share: float = 0.003) -> bool:
    """Почти нет тёмных пикселей (меньше 0,3%) — на картинке нет текста. Номер страницы внизу
    в одиночку этот порог не превышает."""
    pix = page.get_pixmap(dpi=40, clip=clip, colorspace=pymupdf.csGRAY)
    samples = pix.samples
    dark = sum(1 for v in samples if v < 200)      # при 40 dpi тонкий шрифт серый, не чёрный
    return dark < dark_share * len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Распознавание сканов руководств")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="второе прочтение страниц без текстового слоя (300 dpi, по половинам) в ocr_verify/ "
                             "— для сверки двух прочтений в check_ocr_digits.py")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.vision_model:
        raise LLMNotConfigured("VISION_MODEL не задан в .env")
    client = get_client(settings)
    folder = "ocr_verify" if args.verify else "ocr"
    for manual in load_manuals(settings.require_kb_dir()):
        if not manual.scan_pages:
            continue
        out_dir = settings.require_index_dir() / folder / manual.doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(settings.require_kb_dir() / manual.file) as doc:
            for n in manual.scan_pages:
                out = out_dir / f"p{n:03d}.txt"
                page = doc[n - 1]
                if args.verify and len(digits(page.get_text())) >= MIN_LAYER_DIGITS:
                    continue          # есть текстовый слой — цифры сверяются по нему
                if out.exists() and not args.force:
                    print(f"{manual.doc_id} стр. {n}: уже распознана")
                    continue
                if args.verify:
                    text = "\n".join(recognize(client, settings, page, dpi=300, clip=clip, step="scan_ocr_verify")
                                     for clip in halves(page))
                else:
                    text = recognize(client, settings, page, dpi=200, clip=None, step="scan_ocr")
                out.write_text(text, encoding="utf-8")
                print(f"{manual.doc_id} стр. {n}: готово")


if __name__ == "__main__":
    main()
