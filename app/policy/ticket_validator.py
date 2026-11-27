"""Валидатор заявки (правило 8 CLAUDE.md). Без модели: возвращает список нарушений.

Вход — то, что предложила модель (DraftInput: вывод, работы, запчасти, chunk_id ссылок),
и то, что знает код (результат проверки, машина, реестр ссылок этого хода). Поля, которые
модель не выбирает, в DraftInput отсутствуют: тип неисправности берётся из диагностики,
приоритет — из compute_urgency, документ и страница — из реестра ссылок.

Проверки:
- статус «отклонение», качество сигнала пригодное (не bad), код применим к двигателю;
- проверка принадлежит машине и машина — та же, что в результате диагностики;
- вывод и работы не пустые;
- хотя бы одна ссылка; каждая — chunk_id из выдачи поиска этого хода по документам
  двигателя этой машины;
- у каждой запчасти указан фрагмент-источник, он среди ссылок заявки, а каталожный номер
  буквально встречается в тексте этого фрагмента; в заявку идёт написание из фрагмента;
- в выводе, работах и названиях деталей нет «ссылок», написанных моделью (документ и страница
  идут только в references), и нет номеров, которых нет в цитируемых фрагментах.
Лишние поля от модели (urgency, references, pdf_page…) — ошибка схемы, а не молчаливый пропуск.
"""
import re
import unicodedata
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from app.integrations.models import Diagnosis
from app.policy.applicability import Applicability
from app.policy.citations import Citation, CitationRegistry, find_reference_like

PART_NUMBER_MIN_LEN = 5
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
# Исполнение через пробел: «1002011- Д», «1007146 -В2», «1007146 - В2» → «…-Д», «…-В2».
# Только одна заглавная буква с цифрами и граница после неё: «– Сталь 45», «- АК7М2Мг», «- СЧ15» не трогаем.
_SUFFIX_GAP = re.compile(r"(?<=\d{4})(?:\s+-\s*|-\s+)(?=[А-ЯЁA-Z]\d*(?:[\s|;,.)]|$))")
# Пробел у дефиса между цифрами в СВОБОДНОМ тексте заявки: «236 -1007011» → «236-1007011», чтобы такой
# номер тоже сверялся с фрагментом. Текст фрагмента так не склеивается: там «50.1307023- 1 0» или
# «12- 311401-П29» (выноска) дали бы номер, которого в карте нет.
_DIGIT_DASH_GAP = re.compile(r"(?<=\d{3})(?:\s+-(?=\d)|-\s+(?=\d))")
# после буквы исполнения через пробел цифра: «-А 2» — то ли «А2», то ли соседняя ячейка
# не в счёт: следующий номер «111-…», марка «08ПС», пункт «4.13.1»; «А 2.» в конце фразы — в счёт
_DIGIT_AFTER_SUFFIX = re.compile(r"\s\d{1,2}(?![\d\-]|[А-ЯЁA-Zа-яё]|[.,]\d|\.\.)")   # «Г 53...63 HRC» — твёрдость
# номер, кончающийся цифрой, за которым через пробел ещё 1–3 цифры: разорван пробелом внутри цифр
# («236-10070 92-Б2», «…130718 0;»); номер выноски на следующей строке («… 18 - …») не в счёт
_DIGITS_AFTER_NUMBER = re.compile(r"\s\d{1,3}(?:-(?!\d{4})|[;)]|$)")   # «111-2222444» дальше — следующий номер
# короткая основная группа (до 6 цифр) и за ней через пробел ещё цифры: «7511.10071 90», «236-1007 23» —
# номер разорван пробелом, засчитывать начало нельзя (полный номер — 7 цифр в группе)
_SHORT_GROUP = re.compile(r"\d{2,}[А-ЯЁA-Z]{0,3}[-.]\d{1,6}")
_DIGITS_GAP = re.compile(r"\s\d")
_DASH_SPACE = re.compile(r"\s-|-\s")
# Похоже на каталожный номер: «236-1007010», «238-1005170-А», «5336.1003010-10».
# Также крепёж «201457-П29», номер с пробелом «236 1007010» и любой ряд из 6+ цифр.
# Перед номером не цифра, не дефис и не «цифра.»; буква или «№» вплотную («№236-…», «кат.236-…») не мешают.
_PART_LIKE = re.compile(r"(?<![\d\-])(?<!\d\.)(?:\d{2,}[А-ЯЁA-Z]{1,3}[-.]\d{4,}(?:[-.]\w+)*|\d{2,}[-.]\d{4,}(?:[-.]\w+)*|\d{5,}-[А-ЯЁA-Z]\w*|\d{3} \d{7}|\d{6,})(?!\w)")


class SparePartInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_number: str = Field(description="Каталожный номер ровно как в тексте фрагмента")
    name: str = Field(default="", description="Название детали из руководства")
    chunk_id: str = Field(description="Фрагмент, где встречается номер")


class DraftInput(BaseModel):
    """То, что предлагает модель через инструмент draft_ticket."""
    model_config = ConfigDict(extra="forbid")

    conclusion: str = ""
    operations: list[str] = Field(default_factory=list)
    spare_parts: list[SparePartInput] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


@dataclass
class ValidationResult:
    violations: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)   # ссылки по порядку, без повторов
    spare_parts: list[str] = field(default_factory=list)      # «номер из фрагмента + название»

    @property
    def ok(self) -> bool:
        return not self.violations


def normalize_for_match(text: str) -> str:
    """Для «буквального» сравнения номеров: единый вид дефисов и пробелов, без переноса строк.
    Пробел между дефисом и буквой исполнения убирается: в картах дефектации 236.01.01 РК номер
    набран как «236-1002011- Д», это тот же номер, что «236-1002011-Д».
    Буквы и цифры не меняются — «236-1007010-В» не совпадёт с «236-1007010-Б»."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = _SUFFIX_GAP.sub("-", text)
    return re.sub(r"\s+", " ", text).strip()


def find_part_number(part_number: str, text: str) -> str | None:
    """Номер в тексте целиком (не часть более длинного: «236-1007010» не засчитывается внутри
    «236-1007010-В», «236-1007010- В» или «1236-1007010»). Номер, кончающийся буквой, не засчитывается,
    если в тексте за ним через пробел цифра («236-1007176-А 2»): это может быть «А2» — номер неоднозначен,
    лучше отказ, чем другая деталь в заявке. Возвращает написание из текста или None."""
    number = normalize_for_match(part_number)
    if not number or _DASH_SPACE.search(number):
        return None          # «236-1011208- 2» — обрывок испорченного текста, а не номер
    # за номером нельзя: буква/цифра, «-В» (исполнение), висящий дефис «- -В2», «- 2 | Г» (испорченный текст
    # 236.01.01 РК), « -В2» с границей после; можно: « – Сталь 45», « - АК7М2Мг» (материал)
    pattern = (rf"(?<![\w\-]){re.escape(number)}"
               r"(?![\w]|-\s?[\w-]|\s?-\s?[А-ЯЁA-Z]\d*(?:[\s|;,.)]|$))")
    haystack = normalize_for_match(text)
    for match in re.finditer(pattern, haystack):
        if number[-1].isalpha() and _DIGIT_AFTER_SUFFIX.match(haystack, match.end()):
            continue
        if number[-1].isdigit() and _DIGITS_AFTER_NUMBER.match(haystack, match.end()):
            continue         # «236-10070 92-Б2» — номер разорван пробелом, «236-10070» — его обрывок
        if _SHORT_GROUP.fullmatch(number) and _DIGITS_GAP.match(haystack, match.end()):
            continue
        return match.group(0)
    return None


def part_number_in_text(part_number: str, text: str) -> bool:
    return find_part_number(part_number, text) is not None


def validate_draft(draft: DraftInput, *, diagnosis: Diagnosis, applicability: Applicability,
                   vehicle_id: str, vehicle_sessions: set[str], engine_family: str | None,
                   registry: CitationRegistry,
                   part_number_min_len: int = PART_NUMBER_MIN_LEN) -> ValidationResult:
    result = ValidationResult()
    v = result.violations

    if diagnosis.status != "anomaly":
        v.append(f"Заявка возможна только при отклонении, статус проверки — {diagnosis.status}")
    if diagnosis.signal_quality == "bad":
        v.append("Качество записи непригодно — нужна повторная запись, а не заявка")
    if not applicability.applicable:
        v.append(f"Тип неисправности {diagnosis.anomaly_type} не применим к двигателю: {applicability.reason}")
    if diagnosis.anomaly_type is None:
        v.append("Нет типа неисправности в ответе диагностики")
    if diagnosis.anomaly_score is None:
        v.append("Нет anomaly_score в ответе диагностики — приоритет не рассчитать")
    if diagnosis.vehicle_id != vehicle_id:
        v.append("Результат диагностики относится к другой машине")
    if diagnosis.session_id not in vehicle_sessions:
        v.append(f"Проверка {diagnosis.session_id} не принадлежит машине {vehicle_id}")
    if engine_family is None or registry.engine_family != engine_family:
        v.append("Ссылки проверены не по документам двигателя этой машины")

    if not draft.conclusion.strip():
        v.append("Пустой вывод")
    operations = [op.strip() for op in draft.operations]
    if not operations or not all(operations):
        v.append("Перечень работ пуст или содержит пустую строку")

    by_id: dict[str, Citation] = {}
    if not draft.chunk_ids:
        v.append("Нет ни одной ссылки на руководство")
    for cid in draft.chunk_ids:
        citation = registry.resolve(cid)
        if citation is None:
            v.append(f"Ссылка {cid!r} не из выдачи поиска по руководствам этого двигателя")
        else:
            by_id.setdefault(citation.chunk_id, citation)
    result.citations = list(by_id.values())

    for part in draft.spare_parts:
        number = normalize_for_match(part.part_number)
        if len(number) < part_number_min_len or not any(ch.isdigit() for ch in number):
            v.append(f"Каталожный номер {part.part_number!r} слишком короткий или без цифр")
            continue
        citation = by_id.get(part.chunk_id.strip())
        if citation is None:
            v.append(f"Запчасть {part.part_number!r}: фрагмент-источник {part.chunk_id!r} не среди ссылок заявки")
            continue
        found = find_part_number(number, citation.text)
        if found is None:
            v.append(f"Каталожный номер {part.part_number!r} не найден в тексте фрагмента "
                     f"({citation.manual}, стр. {citation.pdf_page})")
            continue
        result.spare_parts.append(f"{found} {part.name.strip()}".strip())

    # Свободный текст заявки: ссылки — только в references, номера — только из цитируемых фрагментов.
    cited_texts = [c.text for c in result.citations]
    fields = [("вывод", draft.conclusion), *(("работы", op) for op in draft.operations),
              *(("название детали", p.name) for p in draft.spare_parts)]
    for label, text in fields:
        for ref in find_reference_like(text, registry.manuals):
            v.append(f"Поле «{label}»: ссылка на документ или страницу в тексте не допускается ({ref!r}) — "
                     "источники указываются только через chunk_ids")
        for number in _PART_LIKE.findall(_DIGIT_DASH_GAP.sub("-", normalize_for_match(text))):
            if not any(part_number_in_text(number, t) for t in cited_texts):
                v.append(f"Поле «{label}»: номер {number!r} не найден в тексте цитируемых фрагментов")
    return result
