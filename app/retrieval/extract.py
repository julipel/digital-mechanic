"""Постраничное извлечение текста из PDF (PyMuPDF).

Номер страницы = индекс страницы в PDF с 1 — берётся из цикла по страницам, никогда из текста.
Служебные строки вырезаются по правилам документа из config/manuals.json (их нашёл разбор PDF):
- напечатанный номер: строка из одних цифр в нижней полосе, равная «индекс PDF + offset»
  (у 536.3902150 РЭ offset = −1). Проверка значения нужна: внизу бывают и цифры таблиц;
- колонтитулы по регуляркам strip_patterns в верхней/нижней полосе (536.01.02 РК);
- название карты операции (536.01.02 РК) стоит в полосе title_band на каждой странице карты:
  оно уходит в метаданные (title), а из текста вырезается.
find_tables не используется: на этих PDF он рвёт и сдвигает ячейки. Таблицы сохраняются тем,
что фрагмент со строками таблицы не режется внутри страницы (см. chunk.py).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.retrieval.manuals import Manual

TOP_BAND = 0.09
BOTTOM_BAND = 0.88
_NUMBER = re.compile(r"^\d{1,3}$")
_SPACES = re.compile(r"\s+")


@dataclass
class Line:
    text: str
    y0: float
    y1: float
    size: float
    bold: bool
    x0: float = 0.0
    x1: float = 0.0


@dataclass
class PageText:
    pdf_page: int                   # индекс в PDF с 1
    lines: list[Line] = field(default_factory=list)
    title: str | None = None        # название карты операции, если документ его задаёт
    source: str = "text"            # text | ocr

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def _raw_lines(page: pymupdf.Page) -> list[Line]:
    """Строки страницы в порядке чтения, как их отдаёт PyMuPDF (ещё без склейки рядов)."""
    lines: list[Line] = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        for raw in block["lines"]:
            spans = [s for s in raw["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = _SPACES.sub(" ", "".join(s["text"] for s in raw["spans"])).strip()
            x0, y0, x1, y1 = raw["bbox"]
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            lines.append(Line(text, y0, y1, max(s["size"] for s in spans), bold, x0, x1))
    return lines


def merge_rows(lines: list[Line]) -> list[Line]:
    """Куски одного ряда (слова текста по ширине в 236 РЭ, ячейки таблицы) склеиваются:
    малый зазор → пробел, большой → « | »."""
    merged: list[Line] = []
    for line in lines:
        prev = merged[-1] if merged else None
        if prev is not None and abs(line.y0 - prev.y0) < 2.0 and line.x0 >= prev.x1 - 1:
            gap = line.x0 - prev.x1
            prev.text = f"{prev.text}{' | ' if gap > 3 * line.size else ' '}{line.text}"
            prev.y1, prev.x1 = max(prev.y1, line.y1), line.x1
            prev.size = max(prev.size, line.size)
            prev.bold = prev.bold and line.bold
        else:
            merged.append(Line(**{**line.__dict__}))
    return merged


def clean_page(lines: list[Line], pdf_page: int, height: float, manual: Manual) -> PageText:
    """Убирает служебные строки одной страницы (до склейки рядов, чтобы «2013-02-12» и
    «Страница N» из одного ряда совпали с шаблонами по отдельности) и выделяет название карты."""
    patterns = [re.compile(p) for p in manual.strip_patterns]
    expected_no = None if manual.page_number_offset is None else str(pdf_page + manual.page_number_offset)
    kept: list[Line] = []
    title_parts: list[str] = []
    for line in lines:
        top = line.y1 <= height * TOP_BAND
        bottom = line.y0 >= height * BOTTOM_BAND
        if bottom and _NUMBER.match(line.text) and line.text == expected_no:
            continue
        if (top or bottom) and any(p.search(line.text) for p in patterns):
            continue
        if manual.title_band and manual.title_band[0] <= line.y0 <= manual.title_band[1] and not line.bold:
            title_parts.append(line.text)
            continue
        kept.append(line)
    return PageText(pdf_page=pdf_page, lines=merge_rows(kept), title=" ".join(title_parts) or None)


def extract_pdf(path: Path, manual: Manual) -> list[PageText]:
    """Все страницы документа без колонтитулов и напечатанных номеров."""
    with pymupdf.open(path) as doc:
        return [clean_page(_raw_lines(page), i, page.rect.height, manual)
                for i, page in enumerate(doc, start=1)]
