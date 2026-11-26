"""Память по ТС (B1): прошлые проверки машины и наши заявки — сводка для агента.

Детерминированная функция, без модели: повторы и возвраты в норму считает код, агент их
только пересказывает. Ряд — проверки до выбранной включительно (будущих относительно
выбранной проверки агент не видит), не больше `max_checks` последних.

Что считается:
- повтор: сколько проверок подряд, считая выбранную, показывают тот же код неисправности.
  Сбой записи между ними ряд не прерывает (о двигателе он ничего не говорит), но учитывается
  в пояснении; нормальная проверка или другой код — прерывают;
- норма после отклонения: прошлое отклонение, после которого пришла норма (ремонт или ложное
  срабатывание) — рост дальше не тянем (ловушка данных №3);
- наши заявки по машине из SQLite: по какой проверке, код, приоритет, номер ТОиР.

Поле description из vehicles.json сюда не попадает: это недоверенный текст ML-сервиса.
Код, невозможный для этого двигателя, помечается по таблице применимости, а не моделью.
"""
from dataclasses import dataclass, field
from pathlib import Path

from app.policy.applicability import check_applicability
from app.policy.settings import load_policy

STATUS_RU = {"normal": "норма", "anomaly": "отклонение", "check_failed": "сбой записи"}
QUALITY_RU = {"good": "хорошее", "warning": "помехи", "bad": "непригодное"}


def _failed(s) -> bool:
    return s.status == "check_failed" or s.signal_quality == "bad"


@dataclass(frozen=True)
class History:
    checks: tuple[dict, ...]
    tickets: tuple[dict, ...]
    observations: tuple[str, ...] = field(default_factory=tuple)
    repeat_count: int = 0                    # проверок подряд с кодом выбранной (0 — не отклонение)
    ticket_for_selected: str | None = None   # номер заявки ТОиР по выбранной проверке

    def as_dict(self) -> dict:
        return {"checks": list(self.checks), "tickets": list(self.tickets),
                "observations": list(self.observations), "repeat_count": self.repeat_count,
                "ticket_for_selected": self.ticket_for_selected}


def vehicle_history(vehicle, session_id: str, tickets: list[dict], config_dir: Path, *,
                    selected_ticket_pending: bool = False) -> History:
    """vehicle — карточка ТС (integrations.fleet.Vehicle); tickets — созданные заявки машины
    (storage.tickets.list_tickets); selected_ticket_pending — по выбранной проверке заявка
    отправляется или ответ ТОиР не получен. KeyError — проверки нет у этой машины."""
    current = vehicle.session(session_id)
    if current is None:
        raise KeyError(session_id)
    max_checks = int(load_policy(Path(config_dir)).get("history", {}).get("max_checks", 10))
    series = [s for s in vehicle.sessions if s.checked_at <= current.checked_at][-max_checks:]
    ticket_by_session = {t["session_id"]: t for t in tickets}

    def applicable(s) -> bool:
        return check_applicability(config_dir, vehicle.engine_family, s.anomaly_type).applicable

    checks = []
    for s in reversed(series):                           # новые сверху
        item = {"session_id": s.session_id, "checked_at": s.checked_at.isoformat(), "selected": s is current,
                "status": s.status, "status_ru": STATUS_RU.get(s.status, s.status),
                "anomaly_type": s.anomaly_type, "anomaly_title": s.anomaly_title,
                "anomaly_score": s.anomaly_score, "confidence": s.confidence,
                "signal_quality": s.signal_quality, "signal_quality_ru": QUALITY_RU.get(s.signal_quality)}
        if s.anomaly_type and not applicable(s):
            # ML противоречит устройству двигателя: название и числа не передаём, чтобы модель
            # не пересказала невозможную неисправность как прошлую неисправность машины
            item.update(code_not_applicable=True, anomaly_title=None, anomaly_score=None, confidence=None,
                        note="результат ML противоречит устройству двигателя — недостоверен")
        if s.session_id in ticket_by_session:
            item["ticket_id"] = ticket_by_session[s.session_id]["ticket_id"]
        checks.append(item)

    observations: list[str] = []
    repeat = 0
    if current.status == "anomaly" and not _failed(current) and current.anomaly_type and applicable(current):
        failed_between = 0
        for s in reversed(series):
            if _failed(s):
                failed_between += 1
                continue
            if s.status == "anomaly" and s.anomaly_type == current.anomaly_type:
                repeat += 1
                continue
            break
        if repeat >= 2:
            note = f"{current.anomaly_type} — {repeat}-я проверка подряд с этим отклонением"
            if failed_between:
                note += " (между ними был сбой записи)"
            observations.append(note + ".")
        elif len(series) == 1:
            observations.append("Это первая проверка машины в истории — сравнить не с чем.")
        else:
            observations.append(f"Предыдущая проверка не показывала {current.anomaly_type} — отклонение новое.")

    for prev, nxt in zip(series, series[1:]):
        if prev.status == "anomaly" and nxt.status == "normal" and not _failed(nxt):
            ticket = ticket_by_session.get(prev.session_id)
            observations.append(
                f"После отклонения {prev.anomaly_type} ({prev.checked_at:%d.%m}) следующая проверка — норма "
                f"({nxt.checked_at:%d.%m}): ремонт или ложное срабатывание"
                + (f"; по тому отклонению была заявка {ticket['ticket_id']}" if ticket else "") + ".")

    n_failed = sum(1 for s in series if _failed(s))
    if n_failed:
        observations.append(f"Сбоев записи в ряду: {n_failed}.")

    tickets_view = tuple(
        {"ticket_id": t["ticket_id"], "session_id": t["session_id"], "created_at": t["created_at"],
         "anomaly_type": t["payload"].get("anomaly_type"), "urgency": t["payload"].get("urgency"),
         "operations": len(t["payload"].get("operations") or [])}
        for t in reversed(tickets))
    selected_ticket = ticket_by_session.get(session_id, {}).get("ticket_id")
    if selected_ticket:
        observations.append(f"По выбранной проверке заявка уже создана: {selected_ticket}. Вторую не создать.")
    elif selected_ticket_pending:
        observations.append("По выбранной проверке заявка уже отправлялась, ответ ТОиР не получен — "
                            "новую не готовить, сверить с ТОиР.")
    return History(checks=tuple(checks), tickets=tickets_view, observations=tuple(observations),
                   repeat_count=repeat, ticket_for_selected=selected_ticket)
