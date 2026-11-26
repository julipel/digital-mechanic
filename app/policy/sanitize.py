"""Очистка поля description от внедрённых инструкций (prompt injection).

description приходит от ML-сервиса и является НЕДОВЕРЕННЫМИ данными. Статус, тип и числа
берутся только из структурных полей JSON; текст описания — лишь пояснение для механика.
Если в тексте найден маркер инструкции (обращение к ИИ, «SYSTEM:», «игнорируй инструкции»,
команда создать заявку и т.п.), всё начиная с этого места вырезается — до модели
доходит только описание звука до маркера. Реализация — шаблоны, без LLM: проверяемо и
не может быть «уговорена» тем же текстом, от которого защищает.
"""
import re
import unicodedata
from dataclasses import dataclass

MAX_DESCRIPTION_LEN = 500

# Каждый шаблон — признак текста, адресованного модели, а не механику.
_INJECTION_PATTERNS = [
    # Ролевые метки чата: «SYSTEM:», «assistant:», «[INST]»
    r"\b(system|assistant|developer|user)\s*:",
    r"\[/?inst\]|<\|?im_(start|end)\|?>|</?system>",
    # Обращение к модели: «примечание для AI-агента», «инструкция для ассистента», «note for the AI»
    r"\b(примечани\w*|инструкци\w*|указани\w*|сообщени\w*|команд\w*|note|instruction\w*)\s+"
    r"(для|к|to|for)\s+(the\s+)?(ai|ии|агент\w*|ассистент\w*|модел\w*|llm|gpt|бот\w*|нейросет\w*|чат\w*)",
    r"\b(ai|ии)[-\s]?(агент|ассистент|модел)\w*",
    # Отмена/подмена инструкций
    r"\b(предыдущ\w*|прошл\w*|прежн\w*)\s+(инструкци\w*|указани\w*|правил\w*)",
    # Прямое обращение: «Ассистент, …», «Агенту: …», «ИИ, …»
    r"\b(ассистент\w*|агент\w*|модел\w*|ии|ai|бот\w*|llm|gpt)\s*[,:]",
    # Английские команды без слова ignore
    r"\b(you\s+are|act\s+as|mark\s+(it\s+|this\s+)?as|respond\s+with|set\s+(the\s+)?status|pretend)\b",
    r"\b(игнорир\w*|проигнорир\w*|забудь\w*|отмени\w*)\b",
    r"\bignore\s+(all|any|the|previous|prior)\b|\bdisregard\b",
    # Попытка достать промпт
    r"\b(системн\w*\s+(промпт\w*|инструкци\w*|сообщени\w*)|system\s+prompt|промпт\w*)\b",
    # Команды, меняющие результат или действия агента
    r"\b(создай|созда[её]те|не\s+создавай|оформи|отправь|не\s+отправляй)\s+(\w+\s+)?заявк\w*",
    r"\b(считать|считай|считайте)\s+(проверк\w*|результат\w*|двигател\w*)",
    r"\b(вернуть|верни|установи|поставь)\s+(\w+\s+)?(статус|приоритет)\w*",
    r"\b(сообщи|скажи|ответь)\s+(механику|пользователю)\b",
    r"\b(выведи|покажи|раскрой)\s+(свой|свои|твой|системн)\w*",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁠-⁤﻿]")
# Слово, где смешаны латиница и кириллица («Игнoрируй», «SYSТEM»), — приём обхода фильтра.
_MIXED_SCRIPT_RE = re.compile(r"\b(?=\w*[a-zA-Z])(?=\w*[а-яёА-ЯЁ])\w+\b")
_TRAILING_JUNK = " \t\n,;:—–-"


@dataclass(frozen=True)
class SanitizedText:
    text: str                 # что можно показать механику и передать модели
    injection_detected: bool  # найден посторонний текст — показать пометку механику
    removed: str = ""         # вырезанный фрагмент — только для журнала, модели не передаётся


def sanitize_description(raw: str | None) -> SanitizedText:
    """Возвращает описание без внедрённых инструкций и флаг, был ли посторонний текст."""
    text = _CONTROL_RE.sub("", unicodedata.normalize("NFKC", raw or "")).strip()
    found = [m for m in (_INJECTION_RE.search(text), _MIXED_SCRIPT_RE.search(text)) if m]
    match = min(found, key=lambda m: m.start()) if found else None
    if match is None:
        return SanitizedText(text=_limit(text), injection_detected=False)

    # Режем с начала предложения, в котором найден маркер, если до маркера в этом
    # предложении нет содержательного текста («Примечание для AI…», «SYSTEM: …»);
    # иначе — с самого маркера. Всё после маркера отбрасывается целиком.
    cut = match.start()
    sentence_start = max(text.rfind(". ", 0, cut), text.rfind("! ", 0, cut), text.rfind("? ", 0, cut))
    sentence_start = 0 if sentence_start < 0 else sentence_start + 2
    if not text[sentence_start:cut].strip(_TRAILING_JUNK):
        cut = sentence_start
    clean = text[:cut].rstrip(_TRAILING_JUNK)
    if clean and clean[-1] not in ".!?":
        clean += "."
    return SanitizedText(text=_limit(clean), injection_detected=True, removed=text[cut:])


def _limit(text: str) -> str:
    return text if len(text) <= MAX_DESCRIPTION_LEN else text[:MAX_DESCRIPTION_LEN].rstrip() + "…"
