"""Приоритет заявки (правило 9): детерминированная функция, возвращает уровень и причины.

Порядок расчёта (пороги — config/policy.json):
1. базовый уровень по anomaly_score;
2. не ниже минимума для типа неисправности (например, низкое давление масла — сразу high);
3. +1 уровень, если то же отклонение было на прошлой проверке (повтор) или anomaly_score
   вырос на заметную величину; поднимается не больше чем на один уровень и не выше high;
4. помехи и низкая уверенность уровень НЕ снижают: это причина проверить вручную,
   а не повод считать риск меньше.
Модель приоритет не выбирает: черновик заявки берёт его только отсюда.
Прошлая проверка `normal` (ремонт или ложное срабатывание) обрывает «повтор» и «рост».
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from app.policy.settings import load_policy

LEVELS = ("low", "medium", "high")
LEVEL_RU = {"low": "низкий", "medium": "средний", "high": "высокий"}


class CheckLike(Protocol):
    status: str
    anomaly_type: str | None
    anomaly_score: float | None
    confidence: float | None
    signal_quality: str


@dataclass(frozen=True)
class UrgencyResult:
    level: str
    reasons: tuple[str, ...]


def compute_urgency(current: CheckLike, previous: Iterable[CheckLike], config_dir: Path,
                    fault_titles: dict[str, str] | None = None) -> UrgencyResult:
    """current — текущая проверка (status=anomaly); previous — проверки этой машины ДО неё,
    по возрастанию времени."""
    if current.status != "anomaly" or current.anomaly_type is None or current.anomaly_score is None:
        raise ValueError("Приоритет считается только для отклонения с кодом и anomaly_score")
    cfg = load_policy(Path(config_dir))["urgency"]
    score = current.anomaly_score
    code = current.anomaly_type
    title = (fault_titles or {}).get(code)
    fault = f"{code} ({title})" if title else code
    reasons: list[str] = []

    if score >= cfg["score_high_from"]:
        level = "high"
        reasons.append(f"Выраженность отклонения {score:.2f} — не ниже {cfg['score_high_from']:.2f}")
    elif score >= cfg["score_medium_from"]:
        level = "medium"
        reasons.append(f"Выраженность отклонения {score:.2f} — от {cfg['score_medium_from']:.2f} "
                       f"до {cfg['score_high_from']:.2f}")
    else:
        level = "low"
        reasons.append(f"Выраженность отклонения {score:.2f} — ниже {cfg['score_medium_from']:.2f}")

    minimum = cfg["fault_minimum"].get(code)
    if minimum and LEVELS.index(minimum) > LEVELS.index(level):
        level = minimum
        reasons.append(f"Тип неисправности {fault}: приоритет не ниже «{LEVEL_RU[minimum]}»")

    last = _last_valid(previous)
    raise_reason = None
    if last is not None and last.status == "anomaly" and last.anomaly_type == code:
        delta = score - (last.anomaly_score or 0.0)
        if delta >= cfg["growth_min_delta"]:
            raise_reason = (f"Рост относительно прошлой проверки: {last.anomaly_score:.2f} → {score:.2f}")
        elif cfg.get("repeat_raises", True):
            raise_reason = f"То же отклонение {code} было на прошлой проверке"
    if raise_reason:
        reasons.append(raise_reason)
        if level != "high":
            level = LEVELS[LEVELS.index(level) + 1]
            reasons[-1] += f" — приоритет повышен до «{LEVEL_RU[level]}»"

    if current.signal_quality == "warning":
        reasons.append("Помехи при записи: результат подтвердить ручной проверкой (приоритет не снижен)")
    low_from = load_policy(Path(config_dir))["modes"]["low_confidence_below"]
    if current.confidence is None or current.confidence < low_from:
        conf = "нет" if current.confidence is None else f"{current.confidence:.2f}"
        reasons.append(f"Низкая уверенность ML ({conf}): подтвердить ручной проверкой (приоритет не снижен)")
    return UrgencyResult(level=level, reasons=tuple(reasons))


def _last_valid(previous: Iterable[CheckLike]) -> CheckLike | None:
    """Последняя прошлая проверка, по которой можно судить о двигателе (сбои записи пропускаем)."""
    valid = [p for p in previous if p.status != "check_failed" and p.signal_quality != "bad"]
    return valid[-1] if valid else None
