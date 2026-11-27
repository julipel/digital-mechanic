"""Автоматическая сверка цифр распознанной страницы с текстовым слоем PDF.

Зачем: страницы с испорченным текстовым слоем (236.01.01 РК: слова разорваны на куски, буквы
на стыках повторяются) распознаются vision-моделью. Буквы в таком слое ненадёжны, а цифры в нём
есть и идут по порядку — только разбиты пробелами («34 ,93», «20,2±0,0 65»). Поэтому каждое число
из распознанного текста ищется в потоке цифр текстового слоя той же страницы. Не нашлось — число
попадает в отчёт для ручной проверки по картинке. Найдено — модель его не выдумала и не исказила
(с точностью до порядка: число из цифр, стоящих в слое подряд).

Однозначные числа не проверяются: в потоке цифр они найдутся всегда.
"""
import re

MIN_LAYER_DIGITS = 20   # меньше цифр в текстовом слое страницы — слоя нет (настоящий скан)
_NUMBER = re.compile(r"\d+(?:[.,\-–—/±…]\d+)*")


def numbers(text: str) -> list[str]:
    """Числа и номера из текста: «236-1011208», «20,2±0,065», «34,93»."""
    return _NUMBER.findall(text)


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def found_in_layer(d: str, stream: str) -> bool:
    """Цифры числа есть в потоке цифр слоя:
    - подряд;
    - двумя кусками от 4 цифр (номер в подписи к рисунку разорван строкой соседней колонки:
      «236-10070 | клапан … 24»);
    - двумя кусками от 3 цифр (у номера из 10+ цифр — от 4) с перекрытием в одну цифру (слой повторяет
      символ на стыке кусков:
      «169…2» + «212» = «169…212»).
    Второй кусок ищется только сразу за первым (не дальше MAX_GAP цифр) — иначе кусок «236» или
    «1007» нашёлся бы где угодно на странице и подмена цифры («236-…» → «238-…») прошла бы."""
    if d in stream:
        return True
    pieces = [(d[:i], d[i:], 0) for i in range(4, len(d) - 3)]                 # два куска подряд
    least = 4 if len(d) >= 10 else 3
    pieces += [(d[:i + 1], d[i:], 1) for i in range(least - 1, len(d) - least + 1)]   # стык с повтором цифры
    return any(_in_order(first, second, overlap, stream) for first, second, overlap in pieces)


MAX_GAP = 10


def _in_order(first: str, second: str, overlap: int, stream: str) -> bool:
    start = stream.find(first)
    while start >= 0:
        end = start + len(first) - overlap
        if second in stream[end:end + MAX_GAP + len(second)]:
            return True
        start = stream.find(first, start + 1)
    return False


def unmatched_numbers(ocr_text: str, layer_text: str) -> list[str]:
    """Числа распознанного текста, цифры которых не нашлись в текстовом слое страницы."""
    stream = digits(layer_text)
    seen: set[str] = set()
    out = []
    for number in numbers(ocr_text):
        d = digits(number)
        if len(d) < 2 or number in seen or found_in_layer(d, stream):
            continue
        seen.add(number)
        out.append(number)
    return out


def reading_differences(first: str, second: str) -> tuple[list[str], list[str]]:
    """Сверка двух прочтений скана без текстового слоя: числа (от двух цифр), которые есть
    только в первом и только во втором. Совпавшие числа считаются подтверждёнными."""
    def counted(text: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in numbers(text):
            if len(digits(n)) >= 2:
                out[n] = out.get(n, 0) + 1
        return out
    a, b = counted(first), counted(second)
    # с учётом числа повторов: «16,9» вместо «18,9» в одной из двух одинаковых строк тоже расхождение
    only_first = [n for n in a if a[n] > b.get(n, 0)]
    only_second = [n for n in b if b[n] > a.get(n, 0)]
    return only_first, only_second
