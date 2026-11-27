"""Постраничное извлечение текста из PDF (PyMuPDF).

Номер страницы = индекс страницы в PDF с 1 — берётся из цикла по страницам, никогда из текста.
Служебные строки вырезаются по правилам документа из манифеста KB_DIR/kb.yaml (их нашёл разбор PDF):
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


HORIZONTAL = (1, 0)


def _to_landscape(bbox: tuple[float, float, float, float], direction: tuple[int, int],
                  width: float, height: float) -> tuple[float, float, float, float]:
    """bbox строки повёрнутого текста → координаты альбомного листа, в котором этот текст
    горизонтален: x — вдоль строки, y — сверху вниз по строкам."""
    x0, y0, x1, y1 = bbox
    if direction == (0, -1):          # текст снизу вверх: верх альбомного листа — левый край
        return height - y1, x0, height - y0, x1
    if direction == (0, 1):           # текст сверху вниз: верх альбомного листа — правый край
        return y0, width - x1, y1, width - x0
    return bbox


def _dominant(lines: list[tuple[tuple[int, int], Line]]) -> tuple[int, int]:
    """Направление, в котором на странице больше всего символов текста."""
    chars: dict[tuple[int, int], int] = {}
    for direction, line in lines:
        chars[direction] = chars.get(direction, 0) + len(line.text)
    return max(chars, key=chars.get) if chars else HORIZONTAL


def upright_rotation(page: pymupdf.Page) -> int:
    """На сколько градусов повернуть картинку страницы, чтобы преобладающий текст стал горизонтальным
    (для распознавания сканов vision-моделью): 90 — текст снизу вверх, -90 — сверху вниз, 0 — как есть."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for raw in block.get("lines", []):
            text = "".join(s["text"] for s in raw["spans"]).strip()
            if text:
                lines.append(((round(raw["dir"][0]), round(raw["dir"][1])), Line(text, 0, 0, 0, False)))
    return {(0, -1): 90, (0, 1): -90}.get(_dominant(lines), 0)


def _raw_lines(page: pymupdf.Page) -> list[Line]:
    """Строки страницы в порядке чтения, как их отдаёт PyMuPDF (ещё без склейки рядов).

    Таблица, набранная поперёк книжного листа (карты дефектации 236.01.01 РК), даёт строки
    с направлением (0, ∓1); в координатах книжного листа их порядок и склейка рядов теряются
    («Деф», «ектоскоп»). Если такого текста на странице больше, чем горизонтального, его координаты
    переводятся в альбомный лист и строки упорядочиваются там. Строки других направлений
    (напечатанный номер страницы) остаются в своих координатах и идут после — их вырезает clean_page.
    Сам текст строк не меняется."""
    raw_lines: list[tuple[tuple[int, int], Line]] = []
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
            direction = (round(raw["dir"][0]), round(raw["dir"][1]))
            raw_lines.append((direction, Line(text, y0, y1, max(s["size"] for s in spans), bold, x0, x1)))
    dominant = _dominant(raw_lines)
    if dominant == HORIZONTAL:
        return [line for _, line in raw_lines]
    turned = []
    for direction, line in raw_lines:
        if direction != dominant:
            continue
        x0, y0, x1, y1 = _to_landscape((line.x0, line.y0, line.x1, line.y1), dominant,
                                       page.rect.width, page.rect.height)
        turned.append(Line(line.text, y0, y1, line.size, line.bold, x0, x1))
    return _reading_order(turned) + [line for direction, line in raw_lines if direction != dominant]


def _reading_order(lines: list[Line], tolerance: float = 2.0) -> list[Line]:
    """Ряды сверху вниз (строки с y0 в пределах tolerance — один ряд), в ряду — слева направо."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: l.y0):
        if rows and line.y0 - rows[-1][0].y0 < tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda l: l.x0)]


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
