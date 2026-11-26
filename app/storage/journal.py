"""Журнал ТС: лента действий по одной машине понятным языком (пожелание заказчика, B1).

Отдельного хранилища нет: источник — таблица events (наши действия) и история проверок
из vehicles.json. Технические события (вызовы инструментов, выдача поиска, ошибки кода)
в ленту механика не попадают. Тексты здесь пишет код, а не модель; из данных ML берутся
только структурные поля — описание проверки (недоверенный текст) не показывается.

Типы записей (фильтр в интерфейсе и в CSV):
- check     — проверка из истории (дата, статус, код, выраженность, качество записи);
- request   — наш запрос результата проверки и разбор его агентом;
- question  — вопрос механика, первая строка ответа и страницы руководства;
- ticket    — черновик и созданная заявка (номер ТОиР, приоритет, работы);
- warning   — помехи, сбой записи, низкая уверенность, посторонний текст в описании,
              код не подходит двигателю, отклонённый черновик, ошибка ТОиР или ML.
"""
import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.policy.applicability import check_applicability
from app.policy.history import QUALITY_RU, STATUS_RU
from app.policy.settings import load_policy
from app.policy.urgency import LEVEL_RU
from app.storage.db import connect
from app.storage.tickets import list_tickets

KINDS = ("check", "request", "question", "ticket", "warning")
MAX_EVENTS = 3000


def _num(x) -> str:
    return "—" if x is None else f"{x:.2f}".replace(".", ",")


def _fault(code: str | None, titles: dict[str, str]) -> str:
    if not code:
        return ""
    return f"{code} «{titles[code]}»" if code in titles else code


def _pages(refs) -> list[dict]:
    out, seen = [], set()
    for r in refs or []:
        key = (r.get("manual") or r.get("doc_id"), r.get("pdf_page"))
        if key[0] and key[1] and key not in seen:
            seen.add(key)
            out.append({"manual": key[0], "pdf_page": key[1]})
    return out


def _entry(ts: str, kind: str, title: str, details: str = "", *, session_id: str | None = None,
           pages: list[dict] | None = None) -> dict:
    return {"ts": ts, "kind": kind, "session_id": session_id, "title": title, "details": details,
            "pages": pages or []}


def _check_entries(vehicle, config_dir: Path) -> list[dict]:
    out = []
    for s in vehicle.sessions:
        parts = []
        if s.anomaly_type:
            parts.append(f"{s.anomaly_type} «{s.anomaly_title}»" if s.anomaly_title else s.anomaly_type)
        if s.anomaly_score is not None:
            parts.append(f"выраженность {_num(s.anomaly_score)}")
        if s.confidence is not None:
            parts.append(f"уверенность {_num(s.confidence)}")
        parts.append(f"запись: {QUALITY_RU.get(s.signal_quality, s.signal_quality)}")
        if s.anomaly_type and not check_applicability(config_dir, vehicle.engine_family, s.anomaly_type).applicable:
            parts.append("код не подходит этому двигателю")
        if s.injection_detected:
            parts.append("в описании от ML посторонний текст — проигнорирован")
        out.append(_entry(s.checked_at.isoformat(), "check",
                          f"Проверка {s.session_id}: {STATUS_RU.get(s.status, s.status)}",
                          ", ".join(parts), session_id=s.session_id))
    return out


def _event_entries(row, data: dict, titles: dict[str, str], tickets: dict[str, dict],
                   low_confidence_below: float) -> list[dict]:
    ts, sid, kind = row["ts"], row["session_id"], row["event_type"]
    if kind == "diagnose_result":
        status = data.get("status")
        parts = [STATUS_RU.get(status, str(status))]
        if data.get("anomaly_type"):
            parts.append(_fault(data["anomaly_type"], titles))
        if data.get("anomaly_score") is not None:
            parts.append(f"выраженность {_num(data['anomaly_score'])}")
        out = [_entry(ts, "request", f"Запрошен результат проверки {sid}", ", ".join(parts), session_id=sid)]
        quality, conf = data.get("signal_quality"), data.get("confidence")
        if status == "check_failed" or quality == "bad":
            out.append(_entry(ts, "warning", "Запись непригодна — выводов о двигателе нет",
                              "Предложено повторить запись", session_id=sid))
        else:
            if quality == "warning":
                out.append(_entry(ts, "warning", "Помехи при записи",
                                  "Результат нужно подтвердить ручной проверкой", session_id=sid))
            if status == "anomaly" and (conf is None or conf < low_confidence_below):
                out.append(_entry(ts, "warning", "Низкая уверенность ML-модели",
                                  f"Уверенность {_num(conf)} — результат нужно подтвердить", session_id=sid))
        return out
    if kind == "injection_detected":
        return [_entry(ts, "warning", "В описании от ML-сервиса посторонний текст",
                       "Проигнорирован: выводы — только по структурным полям", session_id=sid)]
    if kind == "ml_inconsistent":
        return [_entry(ts, "warning", f"Код {data.get('anomaly_type')} не подходит двигателю "
                                      f"{data.get('engine_model') or ''}".rstrip(),
                       (data.get("reason") or "") + " Заявка не создаётся.", session_id=sid)]
    if kind == "answer_sent":
        pages = _pages(data.get("citations"))
        details = "Модель была недоступна — ответа нет" if data.get("llm_failed") else data.get("answer_short", "")
        if data.get("scenario") == "explain":
            return [_entry(ts, "request", "Агент разобрал результат проверки", details, session_id=sid, pages=pages)]
        question = data.get("question")
        title = f"Вопрос: {question}" if question else "Вопрос механика"
        return [_entry(ts, "question", title, details, session_id=sid, pages=pages)]
    if kind == "ticket_draft":
        level = LEVEL_RU.get(data.get("urgency"), data.get("urgency"))
        return [_entry(ts, "ticket", f"Черновик заявки: {_fault(data.get('anomaly_type'), titles)}",
                       f"приоритет {level}, работ {data.get('operations', 0)}, запчастей {data.get('spare_parts', 0)}",
                       session_id=sid, pages=_pages(data.get("references")))]
    if kind == "ticket_rejected":
        violations = [str(v) for v in data.get("violations") or []]
        title = ("Отправка черновика отклонена" if data.get("stage") == "confirm"
                 else "Черновик заявки отклонён проверкой")
        more = f" (и ещё {len(violations) - 3})" if len(violations) > 3 else ""
        return [_entry(ts, "warning", title, "; ".join(violations[:3]) + more, session_id=sid)]
    if kind == "ticket_created":
        level = LEVEL_RU.get(data.get("urgency"), data.get("urgency"))
        ticket = tickets.get(sid) or {}
        operations = (ticket.get("payload") or {}).get("operations") or []
        details = f"приоритет {level}" + (f"; работы: {'; '.join(operations)}" if operations else "")
        return [_entry(ts, "ticket", f"Заявка {data.get('ticket_id')} создана в ТОиР: "
                                     f"{_fault(data.get('anomaly_type'), titles)}",
                       details, session_id=sid, pages=_pages(data.get("references")))]
    if kind == "toir_error":
        return [_entry(ts, "warning", "ТОиР не принял заявку",
                       f"{data.get('error', '')} {data.get('status_code') or ''}".strip(), session_id=sid)]
    if kind == "error" and data.get("where") == "diagnose":
        return [_entry(ts, "warning", "Результат проверки не получен",
                       "ML-сервис недоступен или не нашёл проверку", session_id=sid)]
    return []                      # технические события — не для ленты механика


def _sort_key(entry: dict) -> datetime:
    return datetime.fromisoformat(entry["ts"])


def vehicle_journal(db_path: Path, vehicle, *, config_dir: Path, fault_titles: dict[str, str],
                    kinds: set[str] | None = None) -> list[dict]:
    """Лента по машине, новые сверху. kinds — фильтр по типам записей (None — все)."""
    threshold = float(load_policy(Path(config_dir))["modes"]["low_confidence_below"])
    tickets = {t["session_id"]: t for t in list_tickets(db_path, vehicle.vehicle_id)}
    with connect(db_path) as conn:
        rows = conn.execute("SELECT ts, session_id, event_type, data FROM events WHERE vehicle_id = ?"
                            " ORDER BY id DESC LIMIT ?", (vehicle.vehicle_id, MAX_EVENTS)).fetchall()
    entries = _check_entries(vehicle, config_dir)
    for row in rows:
        try:
            data = json.loads(row["data"] or "{}")
        except ValueError:
            data = {}
        entries += _event_entries(row, data if isinstance(data, dict) else {}, fault_titles, tickets, threshold)
    if kinds:
        entries = [e for e in entries if e["kind"] in kinds]
    entries.sort(key=_sort_key, reverse=True)
    return entries


KIND_RU = {"check": "Проверка", "request": "Запрос результата", "question": "Вопрос",
           "ticket": "Заявка", "warning": "Предупреждение"}


def journal_csv(entries: list[dict]) -> str:
    """CSV для Excel: разделитель «;», BOM добавляет вызывающий код при отдаче файла."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    writer.writerow(["Время (МСК)", "Тип", "Проверка", "Событие", "Подробности", "Страницы руководства"])
    for e in entries:
        pages = ", ".join(f"{p['manual']}, стр. {p['pdf_page']}" for p in e["pages"])
        writer.writerow([_msk(e["ts"]), KIND_RU.get(e["kind"], e["kind"]), e["session_id"] or "",
                         _csv_safe(e["title"]), _csv_safe(e["details"]), pages])
    return buf.getvalue()


MSK = timezone(timedelta(hours=3))


def _msk(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(MSK).strftime("%d.%m.%Y %H:%M")


def _csv_safe(text: str) -> str:
    """Защита от формул в Excel: текст может начинаться с = + - @, табуляции или перевода строки."""
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text
