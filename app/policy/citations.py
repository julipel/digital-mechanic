"""Ссылки на руководство (правила 1–3 CLAUDE.md).

Модель ссылается только на chunk_id фрагментов, которые поиск вернул В ЭТОМ ХОДЕ
(одно действие механика). Обозначение документа и номер страницы PDF подставляет код
из метаданных фрагмента — модель их не пишет и подменить не может.

CitationRegistry создаётся на ход для двигателя текущей машины:
- register(hit) — фрагмент из выдачи поиска; фрагмент другого двигателя — ошибка
  (поиск фильтруется по двигателю, так что это признак поломки, а не обычная ситуация);
- resolve(chunk_id) — Citation или None (неизвестный id / id не из выдачи этого хода);
- render(text) — заменяет маркеры [chunk_id] в ответе на «[документ, стр. N]».
  Всё, что похоже на ссылку, но написано моделью, вырезается: квадратные скобки,
  не разрешившиеся в фрагмент; номера страниц/листов («стр. 50», «145-я страница»,
  «на страницах 50 и 51»); обозначения документов («236.01.01 РК»). Поиск идёт по копии
  текста с латинскими двойниками, заменёнными на кириллицу («cтp. 50» латиницей).
"""
import re
from dataclasses import dataclass
from typing import Iterable, Protocol


class CitationError(ValueError):
    """Фрагмент не того двигателя попал в ссылки — нарушение правила 3."""


class HitLike(Protocol):
    chunk_id: str
    doc_id: str
    manual: str
    engine_family: str
    pdf_page: int
    text: str


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    doc_id: str
    manual: str
    pdf_page: int
    text: str            # текст фрагмента — для проверки номеров деталей; в заявку не идёт

    @property
    def label(self) -> str:
        return f"{self.manual}, стр. {self.pdf_page}"


# Латиница, похожая на кириллицу, → кириллица (длина строки не меняется — позиции совпадают).
_HOMOGLYPHS = str.maketrans("aceopxyiABCEHKMOPTXY", "асеорхуіАВСЕНКМОРТХУ")
_ID = re.compile(r"[A-Za-z0-9_.\-]+")
_MARKER = re.compile(r"\[\s*([A-Za-z0-9_.\-]+(?:\s*[,;]\s*[A-Za-z0-9_.\-]+)*)\s*\]")
_NUM = r"\d+(?:\s*(?:[-–—]|,|и|or|and)\s*\d+)*"
# После замены двойников латинское «page» выглядит как «раgе», «p.» — как «р.».
_PAGE_WORD = r"(?:стр\w*\.?|страниц\w*|лист\w*|раgе\w*)"
# «с.» — страница только не после числа («5 с.» — секунды); «р»/«рр» (латинское p) — только перед цифрой.
# Вторая форма («145-я страница») не срабатывает, если за словом сразу номер: в «236.01.01 стр. 9»
# страница — «стр. 9», а не «01 стр.».
_PAGE = (rf"(?:(?:см\.|на|по)\s+)?"
         rf"(?:(?:{_PAGE_WORD}|\bрр?\.?(?=\s*\d)|(?<!\d\s)(?<!\d)\bс\.)[\s:\-–—]*№?\s*{_NUM}"
         rf"|{_NUM}\s*-?\s*[а-яё]{{0,2}}\s+{_PAGE_WORD}(?![\s:]*№?\s*\d))")
# Обозначения документов в общем виде: «236-3902150-Б РЭ», «236.01.01 РК», «536.01.02 РК».
_DESIGNATION = r"\d{3}[.\-][\w.\-]*\d[\w.\-]*\s*(?:РЭ|РК)\b|\b(?:РЭ|РК)\b|\b\d{3}\.\d{2}\.\d{2}\b"
_BRACKETS = re.compile(r"\s*\[[^\[\]]*\]")
_PAREN_REF = re.compile(rf"\s*\([^()]*(?:{_PAGE}|{_DESIGNATION})[^()]*\)", re.IGNORECASE)
_FORGED = re.compile(rf"\s*(?:{_PAGE}|{_DESIGNATION})", re.IGNORECASE)
_PLACEHOLDER = "{}"
_PRIVATE_USE = re.compile("[-]")     # символы меток render; из текста модели вычищаются


def _spans(pattern: re.Pattern, text: str) -> list[tuple[int, int]]:
    probe = text.translate(_HOMOGLYPHS)
    return [m.span() for m in pattern.finditer(probe)]


def _cut(text: str, spans: list[tuple[int, int]], dropped: list[str]) -> str:
    for start, end in sorted(spans, reverse=True):
        dropped.append(text[start:end].strip())
        text = text[:start] + text[end:]
    return text


def find_reference_like(text: str, manuals: Iterable[str] = ()) -> list[str]:
    """Что в тексте похоже на ссылку, написанную не кодом (для полей заявки)."""
    found: list[str] = []
    _cut(text, _spans(_reference_pattern(manuals), text), found)
    return [f for f in reversed(found) if f]


def _reference_pattern(manuals: Iterable[str]) -> re.Pattern:
    names = "|".join(re.escape(m.translate(_HOMOGLYPHS)) for m in sorted(set(manuals), key=len, reverse=True) if m)
    parts = [r"\[[^\[\]]*\]", rf"\([^()]*(?:{_PAGE}|{_DESIGNATION})[^()]*\)", _PAGE, _DESIGNATION]
    if names:
        parts.insert(0, names)
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)


class CitationRegistry:
    def __init__(self, engine_family: str, manuals: Iterable[str]):
        """manuals — обозначения документов из манифеста (их модель тоже не пишет сама)."""
        if not engine_family:
            raise ValueError("engine_family обязателен")
        self.engine_family = engine_family
        self.manuals = tuple(manuals)
        self._by_id: dict[str, Citation] = {}

    def register(self, hit: HitLike) -> Citation:
        if hit.engine_family != self.engine_family:
            raise CitationError(f"фрагмент {hit.chunk_id} относится к двигателю {hit.engine_family}, "
                                f"а машина — {self.engine_family}")
        if not isinstance(hit.pdf_page, int) or hit.pdf_page < 1:
            raise CitationError(f"фрагмент {hit.chunk_id}: некорректный номер страницы PDF")
        citation = Citation(chunk_id=hit.chunk_id, doc_id=hit.doc_id, manual=hit.manual,
                            pdf_page=hit.pdf_page, text=hit.text)
        self._by_id[hit.chunk_id] = citation
        return citation

    def register_all(self, hits) -> None:
        for hit in hits:
            self.register(hit)

    def resolve(self, chunk_id: str) -> Citation | None:
        return self._by_id.get(str(chunk_id).strip())

    def __contains__(self, chunk_id: str) -> bool:
        return self.resolve(chunk_id) is not None

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self._by_id.values()]

    def render(self, text: str) -> tuple[str, list[Citation], list[str]]:
        """Маркеры → «[документ, стр. N]». Возвращает (текст, использованные ссылки, вырезанное)."""
        used: dict[str, Citation] = {}
        dropped: list[str] = []
        labels: list[str] = []

        # 1. Настоящие маркеры → временные метки (в них нет цифр и скобок — следующие шаги их не тронут).
        def to_placeholder(match: re.Match) -> str:
            found = []
            for cid in _ID.findall(match.group(1)):
                citation = self.resolve(cid)
                if citation is None:
                    dropped.append(cid)
                    continue
                used.setdefault(cid, citation)
                if citation.label not in found:
                    found.append(citation.label)
            if not found:
                return ""
            labels.append(f"[{'; '.join(found)}]")
            return _PLACEHOLDER.format(chr(0xE100 + len(labels) - 1))

        out = _MARKER.sub(to_placeholder, _PRIVATE_USE.sub("", text))
        # 2. Всё, что похоже на ссылку, но написано моделью.
        out = _cut(out, _spans(_BRACKETS, out), dropped)
        out = _cut(out, _spans(_PAREN_REF, out), dropped)
        if self.manuals:
            out = _cut(out, _spans(_reference_pattern(self.manuals), out), dropped)
        out = _cut(out, _spans(_FORGED, out), dropped)
        # 3. Метки → подписи от кода.
        for i, label in enumerate(labels):
            out = out.replace(_PLACEHOLDER.format(chr(0xE100 + i)), label)
        out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
        return out, list(used.values()), [d for d in dropped if d]
