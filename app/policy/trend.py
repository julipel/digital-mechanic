"""Оценка тенденции anomaly_score по проверкам машины (B2). Детерминированная функция, не LLM.

Ряд — последние проверки до выбранной включительно (не больше `window`). Прогноз строится
только по «непрерывному» хвосту ряда: он начинается после последнего разрыва.
Разрыв — это:
- сбой записи (check_failed или плохой сигнал): значения нет, тенденция прерывается;
- код неисправности, невозможный для этого двигателя (ML противоречит устройству двигателя):
  такое значение недостоверно;
- возврат в норму после отклонения (ремонт или ложное срабатывание): рост не тянем дальше,
  хвост начинается с этой нормальной проверки.

Прогноз не строится, если выбранная проверка — разрыв (сбой или противоречие), в хвосте
меньше `min_points` точек или ряд нестабилен (разброс вокруг прямой больше `max_rmse`).
Иначе — линейный тренд (МНК по дням) и оценка, через сколько дней при сохранении темпа
выраженность достигнет порога карантина. Это оценка тенденции, а не дата поломки:
подпись, число точек и оговорки возвращаются вместе с числом. Пороги — config/policy.json.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from app.policy.applicability import check_applicability
from app.policy.settings import load_policy

LABEL = "Оценка тенденции"
CAVEAT = ("Оценка по прямой через последние проверки при сохранении нынешнего темпа. "
          "Это не дата поломки: ремонт, режим работы или ошибка записи меняют картину.")

RISING, STABLE, FALLING = "rising", "stable", "falling"
DIRECTION_RU = {RISING: "растёт", STABLE: "без заметного роста", FALLING: "снижается"}


class CheckLike(Protocol):
    session_id: str
    checked_at: datetime
    status: str
    anomaly_type: str | None
    anomaly_score: float | None
    signal_quality: str


@dataclass(frozen=True)
class Point:
    """Точка графика: все проверки окна, у сбоя записи score = None."""
    session_id: str
    checked_at: str
    status: str
    anomaly_type: str | None
    anomaly_score: float | None
    used: bool                      # участвует в расчёте тренда
    note: str = ""                  # почему точка — разрыв


@dataclass(frozen=True)
class Trend:
    built: bool
    label: str
    points: tuple[Point, ...]
    threshold: float
    n_points: int = 0                         # точек в расчёте
    reason: str = ""                          # почему прогноз не построен
    direction: str | None = None
    slope_per_week: float | None = None
    days_to_threshold: int | None = None      # None — порог не достигается в горизонте или нет роста
    threshold_reached: bool = False
    horizon_days: int = 0
    summary: str = ""
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "built": self.built, "label": self.label, "threshold": self.threshold, "n_points": self.n_points,
            "reason": self.reason, "direction": self.direction,
            "direction_ru": DIRECTION_RU.get(self.direction) if self.direction else None,
            "slope_per_week": self.slope_per_week, "days_to_threshold": self.days_to_threshold,
            "threshold_reached": self.threshold_reached, "horizon_days": self.horizon_days,
            "summary": self.summary, "caveats": list(self.caveats),
            "points": [dict(p.__dict__) for p in self.points],   # копии: вызывающий может их менять
        }


def _failed(c: CheckLike) -> bool:
    return c.status == "check_failed" or c.signal_quality == "bad" or c.anomaly_score is None


def compute_trend(checks: Iterable[CheckLike], config_dir: Path, *,
                  inconsistent_ids: frozenset[str] | set[str] = frozenset()) -> Trend:
    """checks — проверки ОДНОЙ машины до выбранной включительно, в любом порядке; выбранная —
    самая поздняя. inconsistent_ids — проверки, чей код невозможен для этого двигателя."""
    cfg = load_policy(Path(config_dir))["trend"]
    threshold = float(cfg["threshold_score"])
    ordered = sorted(checks, key=lambda c: c.checked_at)[-int(cfg["window"]):]

    # Начало непрерывного хвоста: индекс первой точки после последнего разрыва.
    start, notes = 0, {}
    for i, c in enumerate(ordered):
        if _failed(c):
            notes[i], start = "сбой записи — тенденция прерывается", i + 1
        elif c.session_id in inconsistent_ids:
            notes[i], start = "код не подходит двигателю — значение недостоверно", i + 1
        elif c.status == "normal" and i > 0 and ordered[i - 1].status == "anomaly" and i - 1 not in notes:
            notes[i], start = "норма после отклонения (ремонт или ложное срабатывание) — отсчёт заново", i
    tail = ordered[start:]
    points = tuple(Point(c.session_id, c.checked_at.isoformat(), c.status, c.anomaly_type,
                         c.anomaly_score, used=i >= start, note=notes.get(i, ""))
                   for i, c in enumerate(ordered))
    base = dict(label=LABEL, points=points, threshold=threshold, n_points=len(tail),
                horizon_days=int(cfg["horizon_days"]))

    if not ordered:
        return Trend(built=False, reason="Проверок нет", **base)
    last = ordered[-1]
    if _failed(last):
        return Trend(built=False, reason="Выбранная проверка — сбой записи: оценки тенденции нет", **base)
    if last.session_id in inconsistent_ids:
        return Trend(built=False, reason="Результат выбранной проверки противоречит устройству двигателя: "
                                         "оценки тенденции нет", **base)
    if len(tail) < int(cfg["min_points"]):
        why = f" после разрыва ряда — {notes[max(notes)]}" if notes else ""
        return Trend(built=False, reason=f"Мало проверок для оценки: {len(tail)}{why}; "
                                         f"нужно не меньше {cfg['min_points']}", **base)

    t0 = tail[0].checked_at
    xs = [(c.checked_at - t0).total_seconds() / 86400 for c in tail]
    ys = [float(c.anomaly_score) for c in tail]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return Trend(built=False, reason="Проверки в один день — тенденцию не оценить", **base)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx         # в сутки
    intercept = my - slope * mx
    rmse = (sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    if rmse > float(cfg["max_rmse"]):
        return Trend(built=False, reason=f"Ряд нестабилен: значения меняются скачком (разброс "
                                         f"{rmse:.2f} больше {cfg['max_rmse']}), прямая их не описывает", **base)

    per_30 = slope * 30
    min_change = float(cfg["min_change_per_30d"])
    direction = RISING if per_30 >= min_change else FALLING if per_30 <= -min_change else STABLE
    caveats = [CAVEAT, f"Точек в расчёте: {n}."]
    caveats += [f"{_day(p.checked_at)}: {p.note}." for p in points if p.note]
    if len({c.anomaly_type for c in tail if c.status == "anomaly"}) > 1:
        caveats.append("В ряду разные типы отклонений — выраженность сравнивается условно.")

    horizon = int(cfg["horizon_days"])
    days, reached = None, ys[-1] >= threshold
    if reached:
        summary = f"Выраженность {ys[-1]:.2f} — порог карантина {threshold:.2f} уже достигнут."
    elif direction == RISING:
        # от последнего измеренного значения, а не от прямой: механику понятнее «отсюда при таком темпе»
        hit = (threshold - ys[-1]) / slope
        if hit <= horizon:
            days = max(1, math.ceil(hit))
            summary = (f"Выраженность растёт на {slope * 7:.2f} в неделю; при сохранении темпа порог "
                       f"карантина {threshold:.2f} — примерно через {days} дн.")
        else:
            summary = (f"Выраженность растёт на {slope * 7:.2f} в неделю; порог карантина "
                       f"{threshold:.2f} при таком темпе — не раньше чем через {horizon} дн.")
    else:
        summary = f"Выраженность {DIRECTION_RU[direction]} ({slope * 7:+.2f} в неделю); роста к порогу нет."
    return Trend(built=True, direction=direction, slope_per_week=round(slope * 7, 3), days_to_threshold=days,
                 threshold_reached=reached, summary=summary, caveats=tuple(caveats), **base)


def _day(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d.%m")


def vehicle_trend(vehicle, session_id: str | None, config_dir: Path) -> Trend:
    """Тренд машины на момент проверки session_id (по умолчанию — последней). Противоречивые
    коды определяет таблица применимости, не модель."""
    sessions = list(vehicle.sessions)
    if session_id is not None:
        current = vehicle.session(session_id)
        if current is None:
            raise KeyError(session_id)
        sessions = [s for s in sessions if s.checked_at <= current.checked_at]
    inconsistent = {s.session_id for s in sessions
                    if not check_applicability(config_dir, vehicle.engine_family, s.anomaly_type).applicable}
    return compute_trend(sessions, config_dir, inconsistent_ids=inconsistent)
