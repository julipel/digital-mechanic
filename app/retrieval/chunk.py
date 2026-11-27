"""Нарезка страниц на фрагменты.

Правила (CLAUDE.md, «Ловушки, известные по анализу PDF»):
- фрагмент никогда не пересекает границу страницы → у каждого ровно один pdf_page;
- таблица целиком в одном фрагменте: строки таблиц на этих PDF переносятся и перемешиваются,
  поэтому страница с таблицей не режется вовсе (она и так ограничена одним листом);
- остальные страницы делятся по заголовкам разделов, длинные куски — по строкам;
- текущий заголовок раздела переносится через страницы и пишется в метаданные (section);
  нумерованный пункт-предложение («2. Запрещается…», «9.3 Проверить зазор…») режет страницу
  на фрагменты, как заголовок, но разделом не становится и через страницы не переносится;
- на распознанных сканах (source=ocr) шрифта нет: заголовок — строка заглавными буквами.
Метаданные (doc_id, manual, engine_family, pdf_page) копируются из справочника руководств
и цикла по страницам — текст фрагмента на них не влияет.
"""
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

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
_CLAUSE_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+")
# первое слово пункта-предложения: инфинитив («Проверить», «Разряжать») или предписание
_CLAUSE_VERB = re.compile(r"^(?:не|[а-яё]+(?:ть|ться|ти)|запрещается|допускается|разрешается|следует|"
                          r"необходимо|рекомендуется|должн\w*|нельзя)$", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{4,}")
_OCR_NOT_HEADING = re.compile(r"^(?:внимание|предупреждение|осторожно|примечание|опасно)\b", re.IGNORECASE)


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
    heading: str | None = None  # ближайший заголовок или пункт выше фрагмента — контекст для эмбеддинга

    def to_dict(self) -> dict:
        return asdict(self)


def body_font_size(pages: list[PageText]) -> float:
    """Основной размер шрифта документа — самый частый по числу символов."""
    sizes: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            sizes[round(line.size, 1)] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _mostly_upper(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.8


def is_ocr_heading(line: Line) -> bool:
    """Заголовок на распознанном скане: строка заглавными, не ячейка таблицы и не «ВНИМАНИЕ!»."""
    text = line.text.strip()
    if len(text) > HEADING_MAX_CHARS or "|" in text or _TOC_DOTS.search(text) or _OCR_NOT_HEADING.match(text):
        return False
    return sum(ch.isalpha() for ch in text) >= 6 and _mostly_upper(text)


def is_sentence(text: str, next_text: str | None) -> bool:
    """Строка-заголовок на деле начало предложения: пункт требований или шаг процедуры.
    Строки заглавными — всегда заголовки (у них бывает перенос «ЭЛЕКТРО-»)."""
    if _mostly_upper(text):
        return False
    t = text.rstrip()
    if t.endswith((".", "-", "‑")):
        return True
    words = _CLAUSE_NUMBER.sub("", t).split()
    if words and _CLAUSE_VERB.match(words[0]):
        return True
    # продолжение со строчной — предложение, но только у многоуровневого пункта («6.1.2 Стенд должен…»):
    # глава «6 Общие требования к обкатке…» и ненумерованный заголовок тоже переносятся на вторую строку
    multilevel = "." in (_clause_number(t) or "")
    return multilevel and bool(next_text) and next_text[:1].islower()


def is_heading(line: Line, body_size: float, *, ocr: bool = False) -> bool:
    if not _WORD.search(line.text):      # «А Б I В II» — подписи на рисунке
        return False
    if ocr:
        return is_ocr_heading(line)
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


Block = tuple[str | None, list[Line], bool]    # заголовок, строки, заголовок — раздел (не предложение)


def _blocks(lines: list[Line], body_size: float, *, ocr: bool = False) -> list[Block]:
    """Делит строки страницы на блоки «заголовок + строки до следующего заголовка».
    Первый блок страницы может быть без заголовка (продолжение раздела)."""
    blocks: list[Block] = []
    heading: str | None = None
    current: list[Line] = []
    carries = False
    for i, line in enumerate(lines):
        next_text = lines[i + 1].text if i + 1 < len(lines) else None
        if is_heading(line, body_size, ocr=ocr):
            # перенос длинного заголовка — склеиваем; строка со своим номером («3.1 …») — новый блок
            if (current and _clause_number(line.text) is None
                    and all(is_heading(l, body_size, ocr=ocr) for l in current)):
                heading = f"{heading} {line.text}"   # перенос длинного заголовка
                current.append(line)          # раздел или предложение — решает первая строка
                continue
            if current:
                blocks.append((heading, current, carries))
            heading, current = line.text, [line]
            carries = not is_sentence(line.text, next_text)
        else:
            current.append(line)
    if current:
        blocks.append((heading, current, carries))
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


def _clause_number(text: str | None) -> str | None:
    m = _CLAUSE_NUMBER.match(text or "")
    return m.group(0).strip().rstrip(".") if m else None


@dataclass
class SectionState:
    """Раздел, который переносится через страницы.
    current — действующий раздел; top — последний ненумерованный заголовок; numbered — встреченные
    нумерованные заголовки-разделы по номеру. Пункт-предложение раздел не задаёт, но если его номер
    не из текущего раздела (2.8.22 после «2.2 Требования…»), текущий раздел закончился — откат
    к ближайшему предку по номеру («2.8 …», затем «2 …»), а без него — к top."""
    current: str | None = None
    top: str | None = None
    numbered: dict[str, str] = field(default_factory=dict)
    nearest: str | None = None     # последний заголовок любого вида, включая пункты-предложения

    def apply(self, heading: str, carries: bool) -> None:
        n = _clause_number(heading)
        if carries:
            self.current = heading
            if n is None:
                self.top, self.numbered = heading, {}
            else:
                self.numbered[n] = heading
            return
        c = _clause_number(self.current)
        if n and c and n != c and not n.startswith(c + "."):
            parents = [k for k in self.numbered if n.startswith(k + ".")]
            self.current = self.numbered[max(parents, key=len)] if parents else self.top


Part = tuple[str | None, str | None, list[Line]]    # раздел, ближайший заголовок, строки


def _page_parts(page: PageText, body_size: float, state: SectionState) -> list[Part]:
    """Части страницы; state обновляется до раздела, действующего после страницы."""
    blocks = _blocks(page.lines, body_size, ocr=page.source == "ocr")
    total = sum(len(l.text) for l in page.lines)
    if page_has_table(page) and total <= TABLE_PAGE_MAX:
        # в таблице строки с многоуровневым номером («7.5 Открыты двери автобуса») — причины неисправностей,
        # и номера позиций карты дефектации, а не разделы: на раздел они не влияют вовсе
        first = first_any = None
        nearest_before = state.nearest
        for heading, _, carries in blocks:
            if heading is None:
                continue
            first_any = first_any or heading
            state.nearest = heading
            if carries and "." not in (_clause_number(heading) or ""):
                state.apply(heading, carries)
                first = first or heading
        return [(first or state.current, first_any or nearest_before, page.lines)]

    parts: list[Part] = []
    for heading, lines, carries in blocks:
        if heading is not None:
            state.apply(heading, carries)
            state.nearest = heading
        if parts and sum(len(l.text) for l in parts[-1][2]) < MIN_CHARS:
            parts[-1] = (parts[-1][0], parts[-1][1], parts[-1][2] + lines)
        else:
            parts.append((state.current, state.nearest, lines))
    # первый блок без заголовка — продолжение раздела с прошлой страницы
    return [(s, h, piece) for s, h, lines in parts for piece in _split_long(lines)]


def chunk_document(manual: Manual, pages: list[PageText]) -> list[Chunk]:
    body_size = body_font_size([p for p in pages if p.source == "text"])
    chunks: list[Chunk] = []
    state = SectionState()
    for page in pages:
        prev_section = state.current
        parts = _page_parts(page, body_size, state)
        is_table = page_has_table(page)
        n = 0
        for part_section, part_heading, lines in parts:
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
                heading=part_heading,
            ))
    return chunks
