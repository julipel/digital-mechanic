"""Память по ТС (B1): прошлые проверки машины и наши заявки — сводка для агента.

Детерминированная функция, без модели: повторы и возвраты в норму считает код, агент их
только пересказывает. Ряд — проверки до выбранной включительно (будущих относительно
выбранной проверки агент не видит), не больше `max_checks` последних.

Что считается:
- повтор: сколько проверок подряд, считая выбранную, показывают тот же код неисправности.
  Сбой записи и код, невозможный для двигателя, между ними ряд не прерывают (о двигателе они
  ничего не говорят — так же считает compute_urgency), но учитываются в пояснении; нормальная
  проверка или другой возможный код — прерывают;
- норма после отклонения: прошлое отклонение, после которого пришла норма (ремонт или ложное
  срабатывание) — рост дальше не тянем (ловушка данных №3);
- наши заявки по машине из SQLite: по какой проверке, код, приоритет, номер ТОиР.

Поле description из vehicles.json сюда не попадает: это недоверенный текст ML-сервиса.
Календарных дат в сводке нет — только «дней до выбранной проверки»: сводку читает модель, и
дата проверки рядом со сроком из оценки тенденции позволила бы ей сложить «дату поломки».
Порядок заявок — по времени создания, сама дата не передаётся.
Код, невозможный для этого двигателя, помечается по таблице применимости, а не моделью.
"""
from dataclasses import dataclass, field
from pathlib import Path

from app.policy.applicability import check_applicability
from app.policy.settings import load_policy

STATUS_RU = {"normal": "норма", "anomaly": "отклонение", "check_failed": "сбой записи"}
QUALITY_RU = {"good": "хорошее", "warning": "помехи", "bad": "непригодное"}


def _failed(s) -> bool:
    """Как в trend и fleet_status: отклонение без anomaly_score тоже непригодно для выводов."""
    return s.status == "check_failed" or s.signal_quality == "bad" or (
        s.status == "anomaly" and s.anomaly_score is None)


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

    def ago(s) -> int:
        """Сколько полных дней прошло от проверки s до выбранной (у выбранной — 0)."""
        return (current.checked_at - s.checked_at).days

    def when(s) -> str:
        return "выбранная проверка" if s is current else f"{ago(s)} дн. до выбранной"

    def applicable(s) -> bool:
        return check_applicability(config_dir, vehicle.engine_family, s.anomaly_type).applicable

    checks = []
    for s in reversed(series):                           # новые сверху
        item = {"session_id": s.session_id, "days_before_selected": ago(s), "selected": s is current,
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
        failed_between = inconsistent_between = 0
        for s in reversed(series):
            if _failed(s):
                failed_between += 1
                continue
            if s.anomaly_type and not applicable(s):
                inconsistent_between += 1
                continue
            if s.status == "anomaly" and s.anomaly_type == current.anomaly_type:
                repeat += 1
                continue
            break
        if repeat >= 2:
            note = f"{current.anomaly_type} — {repeat}-я проверка подряд с этим отклонением"
            gaps = (["сбой записи"] if failed_between else []) + (
                ["недостоверный результат ML"] if inconsistent_between else [])
            if gaps:
                note += f" (между ними был {' и '.join(gaps)})"
            observations.append(note + ".")
        elif len(series) == 1:
            observations.append("Это первая проверка машины в истории — сравнить не с чем.")
        else:
            observations.append(f"Предыдущая проверка не показывала {current.anomaly_type} — отклонение новое.")

    # Норма после отклонения — по соседним ПРИГОДНЫМ проверкам: сбой записи между ними не скрывает
    # возврат в норму, а недостоверный код (невозможный для двигателя) отклонением не считается.
    valid = [(i, s) for i, s in enumerate(series) if not _failed(s) and not (s.anomaly_type and not applicable(s))]
    for (i, prev), (j, nxt) in zip(valid, valid[1:]):
        if prev.status == "anomaly" and nxt.status == "normal":
            ticket = ticket_by_session.get(prev.session_id)
            between = series[i + 1:j]
            gaps = (["сбой записи"] if any(_failed(s) for s in between) else []) + (
                ["недостоверный результат ML"] if any(not _failed(s) and s.anomaly_type and not applicable(s)
                                                      for s in between) else [])
            gap = f" (между ними был {' и '.join(gaps)})" if gaps else ""
            observations.append(
                f"После отклонения {prev.anomaly_type} ({when(prev)}) следующая пригодная проверка — норма "
                f"({when(nxt)}){gap}: ремонт или ложное срабатывание"
                + (f"; по тому отклонению была заявка {ticket['ticket_id']}" if ticket else "") + ".")

    n_failed = sum(1 for s in series if _failed(s))
    if n_failed:
        observations.append(f"Сбоев записи в ряду: {n_failed}.")

    tickets_view = tuple(
        {"ticket_id": t["ticket_id"], "session_id": t["session_id"],
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
