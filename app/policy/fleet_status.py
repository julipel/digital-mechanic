"""Режим «Главный механик» (B3): статус машины — норма / требует внимания / карантин.

Детерминированная функция, без модели. Пороги — config/policy.json → fleet_status.
Статус считается на момент проверки (по умолчанию — последней):

1. проверок нет → «требует внимания»: состояние двигателя неизвестно;
2. проверка, по которой нельзя судить о двигателе, — сбой записи или код, невозможный для
   этого двигателя (правила 4 и 6). Выводов по ней нет: статус не ниже «требует внимания»
   с причиной «повторить запись», а состояние двигателя берётся по последней пригодной
   проверке ДО неё. Сбой записи не снимает карантин, но и сам его не ставит;
3. пригодная проверка — отклонение: приоритет считает compute_urgency (правило 9, те же
   причины, что в заявке); уровень из `quarantine_urgency` → карантин, иначе «требует внимания»;
4. пригодная проверка — норма → норма. Прошлое отклонение, после которого пришла норма
   (ремонт или ложное срабатывание, ловушка данных №3), статус НЕ повышает — это пометка;
5. оценка тенденции (B2) может только повысить статус: рост к порогу карантина за
   `attention_trend_days` → не ниже «требует внимания», за `quarantine_trend_days` → карантин
   (null — по прогнозу карантин не ставится: это оценка, а не измерение).

reasons — почему такой статус (показываются в интерфейсе); notes — сведения, статус не
меняющие (заявка создана, прошлое отклонение). Текст description от ML сюда не попадает.
"""
from dataclasses import dataclass, field
from pathlib import Path

from app.policy.applicability import check_applicability
from app.policy.settings import load_policy
from app.policy.trend import RISING, compute_trend
from app.policy.urgency import LEVEL_RU, LEVELS, compute_urgency, previous_checks

NORMAL, ATTENTION, QUARANTINE = "normal", "attention", "quarantine"
STATUSES = (NORMAL, ATTENTION, QUARANTINE)
STATUS_RU = {NORMAL: "норма", ATTENTION: "требует внимания", QUARANTINE: "карантин"}


@dataclass(frozen=True)
class VehicleStatus:
    status: str
    reasons: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    session_id: str | None = None          # проверка, на момент которой посчитан статус
    basis_session_id: str | None = None    # пригодная проверка, по которой судим о двигателе
    urgency: str | None = None             # приоритет отклонения (если основание — отклонение)
    anomaly_type: str | None = None
    anomaly_score: float | None = None
    days_to_threshold: int | None = None   # из оценки тенденции, если она построена
    ticket_id: str | None = None           # наша заявка по проверке-основанию

    def sort_key(self) -> tuple:
        """Приоритетный список: карантин → внимание → норма; внутри — приоритет отклонения,
        ближе порог по тенденции, выше выраженность."""
        return (-STATUSES.index(self.status),
                -(LEVELS.index(self.urgency) if self.urgency else -1),
                self.days_to_threshold if self.days_to_threshold is not None else 10 ** 6,
                -(self.anomaly_score or 0.0))


def _failed(s) -> bool:
    return s.status == "check_failed" or s.signal_quality == "bad" or (
        s.status == "anomaly" and s.anomaly_score is None)


def _day(s) -> str:
    return s.checked_at.strftime("%d.%m")


def _raise(status: str, to: str) -> str:
    return to if STATUSES.index(to) > STATUSES.index(status) else status


def vehicle_status(vehicle, config_dir: Path, *, session_id: str | None = None,
                   fault_titles: dict[str, str] | None = None,
                   ticket_ids: dict[tuple[str, str], str] | None = None) -> VehicleStatus:
    """vehicle — integrations.fleet.Vehicle; ticket_ids — (vehicle_id, session_id) → номер
    созданной заявки. KeyError — проверки session_id нет у этой машины."""
    config_dir = Path(config_dir)
    cfg = load_policy(config_dir)["fleet_status"]
    titles = fault_titles or {}
    tickets = ticket_ids or {}
    sessions = list(vehicle.sessions)
    if session_id is not None:
        current = vehicle.session(session_id)
        if current is None:
            raise KeyError(session_id)
        sessions = [s for s in sessions if s.checked_at <= current.checked_at]
    if not sessions:
        return VehicleStatus(ATTENTION, ("Проверок нет — состояние двигателя неизвестно, нужна запись",))

    def applicable(s) -> bool:
        return check_applicability(config_dir, vehicle.engine_family, s.anomaly_type).applicable

    def fault(s) -> str:
        return f"{s.anomaly_type} ({titles[s.anomaly_type]})" if s.anomaly_type in titles else str(s.anomaly_type)

    last = sessions[-1]
    status, reasons, notes = NORMAL, [], []
    if _failed(last):
        status = ATTENTION
        reasons.append(f"Проверка {_day(last)} — сбой записи: выводов о двигателе нет, повторите запись")
    elif not applicable(last):
        status = ATTENTION
        reasons.append(f"Проверка {_day(last)}: код {fault(last)} не подходит двигателю {vehicle.engine_model} — "
                       "результат ML недостоверен, повторите запись или проверьте вручную")

    basis_idx = next((i for i in range(len(sessions) - 1, -1, -1)
                      if not _failed(sessions[i]) and applicable(sessions[i])), None)
    base = dict(session_id=last.session_id)
    if basis_idx is None:
        if not reasons:
            reasons.append("Нет пригодных проверок — состояние двигателя неизвестно")
        return VehicleStatus(status, tuple(reasons), **base)
    basis = sessions[basis_idx]
    if basis is not last:
        notes.append(f"Состояние двигателя — по последней пригодной проверке {_day(basis)} "
                     f"({'отклонение' if basis.status == 'anomaly' else 'норма'})")

    urgency = None
    ticket_id = tickets.get((vehicle.vehicle_id, basis.session_id))
    if basis.status == "anomaly":
        u = compute_urgency(basis, previous_checks(vehicle, basis.session_id, config_dir), config_dir, titles)
        urgency = u.level
        to = QUARANTINE if u.level in cfg["quarantine_urgency"] else ATTENTION
        status = _raise(status, to)
        reasons.append(f"Отклонение {fault(basis)} {_day(basis)}, приоритет «{LEVEL_RU[u.level]}»")
        reasons.extend(u.reasons)
        notes.append(f"Заявка {ticket_id} создана" if ticket_id else "Заявки по этому отклонению нет")
    else:
        earlier = next((s for s in reversed(sessions[:basis_idx])
                        if s.status == "anomaly" and not _failed(s) and applicable(s)), None)
        if earlier is not None:
            t = tickets.get((vehicle.vehicle_id, earlier.session_id))
            notes.append(f"{_day(earlier)} было отклонение {fault(earlier)}, затем норма "
                         f"(ремонт или ложное срабатывание){f' — заявка {t}' if t else ''}")

    inconsistent = {s.session_id for s in sessions[:basis_idx + 1] if not applicable(s)}
    trend = compute_trend(sessions[:basis_idx + 1], config_dir, inconsistent_ids=inconsistent)
    days = None
    if trend.built and trend.direction == RISING and trend.days_to_threshold is not None:
        days = trend.days_to_threshold
        q_days, a_days = cfg.get("quarantine_trend_days"), cfg.get("attention_trend_days")
        if q_days is not None and days <= q_days:
            status = _raise(status, QUARANTINE)
            reasons.append(f"{trend.label}: {trend.summary} — не дольше {q_days} дн., карантин")
        elif a_days is not None and days <= a_days:
            status = _raise(status, ATTENTION)
            reasons.append(f"{trend.label}: {trend.summary}")
        else:
            notes.append(f"{trend.label}: {trend.summary}")

    if status == NORMAL and not reasons:
        reasons.append(f"Проверка {_day(basis)} — норма, выраженность {basis.anomaly_score:.2f}"
                       if basis.anomaly_score is not None else f"Проверка {_day(basis)} — норма")
    return VehicleStatus(status, tuple(reasons), tuple(notes), urgency=urgency, anomaly_type=basis.anomaly_type
                         if basis.status == "anomaly" else None, anomaly_score=basis.anomaly_score,
                         days_to_threshold=days, ticket_id=ticket_id, basis_session_id=basis.session_id, **base)


def status_history(vehicle, config_dir: Path, *, fault_titles: dict[str, str] | None = None,
                   ticket_ids: dict[tuple[str, str], str] | None = None) -> list[dict]:
    """Статус машины на момент каждой проверки, по возрастанию времени — «история по клику»."""
    return [{"session_id": s.session_id, "checked_at": s.checked_at.isoformat(),
             "status": vehicle_status(vehicle, config_dir, session_id=s.session_id,
                                      fault_titles=fault_titles, ticket_ids=ticket_ids).status}
            for s in vehicle.sessions]


def status_since(history: list[dict]) -> str | None:
    """С какой проверки держится нынешний статус (checked_at первой проверки серии)."""
    if not history:
        return None
    since = history[-1]
    for h in reversed(history):
        if h["status"] != since["status"]:
            break
        since = h
    return since["checked_at"]
