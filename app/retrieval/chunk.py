"""Нарезка страниц на фрагменты.

Правила (CLAUDE.md, «Ловушки, известные по анализу PDF»):
- фрагмент никогда не пересекает границу страницы → у каждого ровно один pdf_page;
- таблица целиком в одном фрагменте: строки таблиц на этих PDF переносятся и перемешиваются,
  поэтому страница с таблицей не режется вовсе (она и так ограничена одним листом);
- остальные страницы делятся по заголовкам разделов, длинные куски — по строкам;
- текущий заголовок раздела переносится через страницы и пишется в метаданные (section).
Метаданные (doc_id, manual, engine_family, pdf_page) копируются из справочника руководств
и цикла по страницам — текст фрагмента на них не влияет.
"""
import re
from collections import Counter
from dataclasses import asdict, dataclass

from app.retrieval.extract import Line, PageText
from app.retrieval.manuals import Manual

MIN_CHARS = 250      # короче — приклеиваем к соседнему блоку, чтобы не плодить обрывки
MAX_CHARS = 1600     # длиннее — режем по строке
TABLE_PAGE_MAX = 6000  # страница с таблицей длиннее этого режется по заголовкам (на практике не бывает)
HEADING_MAX_CHARS = 120

_NUMBERED_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+[А-ЯЁA-Z]")
_TOC_DOTS = re.compile(r"\.{4,}|…{2,}")
# «Таблица 3», «табл. 23», «Продолжение таблицы 3», «Таблица А.1», «Таблица 6.2»
_TABLE_CAPTION = re.compile(r"(?:^|\s)табл\w*\.?\s*[А-ЯA-Z]?\.?\s*\d", re.IGNORECASE)
# шапка таблицы неисправностей без подписи «Таблица»
_FAULT_HEADER = re.compile(r"^(?:причин\w*(?:\s+неисправност\w*)?|(?:способ|метод)\w*\s+устранени\w*|возможн\w+\s+причин\w*)$",
                           re.IGNORECASE)
TABLE_ROWS_MIN = 6   # столько строк вида «ячейка | ячейка» — страница табличная
_VALUE = re.compile(r"\d+(?:[,.]\d+)?\s*(?:[-–…±]|\.\.\.)\s*\d|\d+[,.]\d+")
_FAULT_WORDS = re.compile(r"неисправност|способ\w*\s+устранен|причин\w*\s+неисправ", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(?<=[а-яёa-z])-\n(?=[а-яёa-z])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    manual: str
    engine_family: str
    pdf_page: int
    section: str | None
    title: str | None          # название карты операции (536.01.02 РК)
    text: str
    is_table: bool
    is_fault_table: bool
    source: str                # text | ocr

    def to_dict(self) -> dict:
        return asdict(self)


def body_font_size(pages: list[PageText]) -> float:
    """Основной размер шрифта документа — самый частый по числу символов."""
    sizes: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            sizes[round(line.size, 1)] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def is_heading(line: Line, body_size: float) -> bool:
    text = line.text
    if len(text) > HEADING_MAX_CHARS or len(text) < 3 or _TOC_DOTS.search(text):
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3 or text.rstrip().endswith((",", ";", ":")):
        return False
    if _NUMBERED_HEADING.match(text):
        # «4.7 Фильтр масляный» — заголовок; «1. Настроить датчик…» — пункт списка,
        # если не выделен шрифтом
        multilevel = re.match(r"^\d{1,2}\.\d", text) is not None
        if multilevel or line.bold or line.size >= body_size + 0.5:
            return True
    upper = sum(ch.isupper() for ch in letters) / len(letters) > 0.8
    return line.size >= body_size + 1.5 or (line.bold and upper)


def page_has_table(page: PageText) -> bool:
    if page.source == "ocr":
        return True
    lines = page.lines
    if any(_TABLE_CAPTION.search(l.text) for l in lines):
        return True
    # шапка таблицы неисправностей: после склейки ряда это «Причина | Способ устранения»
    if any(_FAULT_HEADER.match(cell.strip()) for l in lines for cell in l.text.split("|")):
        return True
    if sum(" | " in l.text for l in lines) >= TABLE_ROWS_MIN:
        return True
    short = [l for l in lines if len(l.text) < 60]
    return len(lines) >= 8 and sum(bool(_VALUE.search(l.text)) for l in short) >= 0.3 * len(lines)


def clean_text(text: str) -> str:
    """Склейка переносов «ци-\\nлиндра» → «цилиндра» (для поиска по словам)."""
    return _HYPHEN_BREAK.sub("", text).strip()


def _blocks(lines: list[Line], body_size: float) -> list[tuple[str | None, list[Line]]]:
    """Делит строки страницы на блоки «заголовок + строки до следующего заголовка».
    Первый блок страницы может быть без заголовка (продолжение раздела)."""
    blocks: list[tuple[str | None, list[Line]]] = []
    heading: str | None = None
    current: list[Line] = []
    for line in lines:
        if is_heading(line, body_size):
            if current and all(is_heading(l, body_size) for l in current):
                heading = f"{heading} {line.text}"   # перенос длинного заголовка
                current.append(line)
                continue
            if current:
                blocks.append((heading, current))
            heading, current = line.text, [line]
        else:
            current.append(line)
    if current:
        blocks.append((heading, current))
    return blocks


def _split_long(lines: list[Line]) -> list[list[Line]]:
    parts: list[list[Line]] = [[]]
    size = 0
    for line in lines:
        if size >= MAX_CHARS and parts[-1]:
            parts.append([])
            size = 0
        parts[-1].append(line)
        size += len(line.text) + 1
    return parts


def _page_parts(page: PageText, body_size: float, section: str | None) -> tuple[list[tuple[str | None, list[Line]]], str | None]:
    """Части страницы (раздел, строки) и раздел, действующий после этой страницы."""
    blocks = _blocks(page.lines, body_size)
    for heading, _ in blocks:
        if heading is not None:
            section = heading
    total = sum(len(l.text) for l in page.lines)
    if page_has_table(page) and total <= TABLE_PAGE_MAX:
        first = next((h for h, _ in blocks if h), None)
        return [(first or section, page.lines)], section

    parts: list[tuple[str | None, list[Line]]] = []
    current_section = None
    for heading, lines in blocks:
        current_section = heading or current_section
        if parts and sum(len(l.text) for l in parts[-1][1]) < MIN_CHARS:
            parts[-1] = (parts[-1][0], parts[-1][1] + lines)
        else:
            parts.append((heading or current_section, lines))
    # первый блок без заголовка — продолжение раздела с прошлой страницы
    return [(s, piece) for s, lines in parts for piece in _split_long(lines)], section


def chunk_document(manual: Manual, pages: list[PageText]) -> list[Chunk]:
    body_size = body_font_size([p for p in pages if p.source == "text"])
    chunks: list[Chunk] = []
    section: str | None = None
    for page in pages:
        prev_section = section
        parts, section = _page_parts(page, body_size, section)
        is_table = page_has_table(page)
        n = 0
        for part_section, lines in parts:
            text = clean_text("\n".join(l.text for l in lines))
            if len(text) < 20:
                continue
            n += 1
            part_section = part_section or prev_section
            chunks.append(Chunk(
                chunk_id=f"{manual.doc_id}-p{page.pdf_page:03d}-{n}",
                doc_id=manual.doc_id,
                manual=manual.manual,
                engine_family=manual.engine_family,
                pdf_page=page.pdf_page,
                section=part_section,
                title=page.title,
                text=text,
                is_table=is_table,
                is_fault_table=is_table and bool(_FAULT_WORDS.search(f"{part_section or ''}\n{text}")),
                source=page.source,
            ))
    return chunks
