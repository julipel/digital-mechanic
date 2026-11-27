"""Сверка цифр распознанных страниц с текстовым слоем PDF (см. app/retrieval/ocr_check.py).

Для каждой страницы из scan_pages, у которой есть текстовый слой, ищет числа распознанного текста
в цифрах слоя. Расхождения — в отчёт INDEX_DIR/ocr/check_report.txt (вне репозитория) и коротко
в консоль: документ, страница, число. Их и только их проверяют вручную по картинке страницы.
Страницы без текстового слоя (настоящие сканы): если есть второе прочтение (ocr_scans.py --verify),
сравниваются два прочтения, расхождения — в тот же отчёт; без второго прочтения — ручная сверка целиком.

    python scripts/check_ocr_digits.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.retrieval.manuals import load_manuals  # noqa: E402
from app.retrieval.ocr_check import MIN_LAYER_DIGITS, digits, numbers, reading_differences, unmatched_numbers  # noqa: E402



def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    index_dir = settings.require_index_dir()
    report = []
    checked = skipped = total_bad = 0
    for manual in load_manuals(settings.require_kb_dir()):
        if not manual.scan_pages:
            continue
        cache = index_dir / "pages" / f"{manual.doc_id}.json"
        if not cache.exists():
            print(f"{manual.doc_id}: нет кеша страниц — сначала scripts/build_index.py")
            return 2
        layer = {p["pdf_page"]: "\n".join(l["text"] for l in p["lines"])
                 for p in json.loads(cache.read_text(encoding="utf-8"))["pages"]}
        for n in manual.scan_pages:
            ocr_file = index_dir / "ocr" / manual.doc_id / f"p{n:03d}.txt"
            if not ocr_file.exists():
                print(f"{manual.doc_id} стр. {n}: не распознана")
                continue
            text_layer = layer.get(n, "")
            if len(digits(text_layer)) < MIN_LAYER_DIGITS:
                skipped += 1
                verify = index_dir / "ocr_verify" / manual.doc_id / f"p{n:03d}.txt"
                if verify.exists():
                    first = ocr_file.read_text(encoding="utf-8")
                    only1, only2 = reading_differences(first, verify.read_text(encoding="utf-8"))
                    # половины во втором прочтении перекрываются — лишние повторы там не в счёт,
                    # показываем только числа второго прочтения, которых в индексе нет совсем
                    in_first = set(numbers(first))
                    only2 = [n for n in only2 if n not in in_first]
                    if only1 or only2:
                        total_bad += len(only1)
                        report.append(f"{manual.doc_id} стр. {n} (скан, два прочтения): "
                                      f"в индексе {', '.join(only1) or '—'} | во втором {', '.join(only2) or '—'}")
                continue
            checked += 1
            bad = unmatched_numbers(ocr_file.read_text(encoding="utf-8"), text_layer)
            if bad:
                total_bad += len(bad)
                report.append(f"{manual.doc_id} стр. {n}: {', '.join(bad)}")
    (index_dir / "ocr" / "check_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"проверено страниц: {checked}, без текстового слоя (ручная сверка): {skipped}")
    print(f"чисел не найдено в текстовом слое: {total_bad} на {len(report)} стр.")
    for line in report:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
