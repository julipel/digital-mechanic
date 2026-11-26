"""SQLite: черновики заявок и созданные заявки. Только хранение — правила в policy/tickets.py."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.storage.db import connect
from app.storage.events import request_id_var


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def insert_draft(db_path: Path, *, draft_id: str, vehicle_id: str, session_id: str,
                 payload: dict, content_hash: str, extra: dict) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO ticket_drafts (draft_id, created_at, request_id, vehicle_id, session_id,"
            " payload, content_hash, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (draft_id, _now(), request_id_var.get(), vehicle_id, session_id,
             json.dumps(payload, ensure_ascii=False), content_hash, json.dumps(extra, ensure_ascii=False)))


def get_draft(db_path: Path, draft_id: str) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM ticket_drafts WHERE draft_id = ?", (draft_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["payload"] = json.loads(data["payload"])
    data["extra"] = json.loads(data["extra"])
    return data


def set_draft_status(db_path: Path, draft_id: str, status: str, *, only_from: str = "draft") -> bool:
    with connect(db_path) as conn:
        cur = conn.execute("UPDATE ticket_drafts SET status = ? WHERE draft_id = ? AND status = ?",
                           (status, draft_id, only_from))
        return cur.rowcount == 1


def get_ticket(db_path: Path, vehicle_id: str, session_id: str) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM tickets WHERE vehicle_id = ? AND session_id = ?",
                           (vehicle_id, session_id)).fetchone()
    return dict(row) if row else None


def reserve_ticket(db_path: Path, vehicle_id: str, session_id: str, draft_id: str) -> bool:
    """Занимает «место» заявки по проверке. False — по этой проверке заявка уже есть
    или отправляется прямо сейчас (PRIMARY KEY не даст вставить вторую строку)."""
    try:
        with connect(db_path) as conn:
            conn.execute("INSERT INTO tickets (vehicle_id, session_id, draft_id, status, created_at)"
                         " VALUES (?, ?, ?, 'sending', ?)", (vehicle_id, session_id, draft_id, _now()))
        return True
    except sqlite3.IntegrityError:
        return False


def complete_ticket(db_path: Path, vehicle_id: str, session_id: str, ticket_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE tickets SET ticket_id = ?, status = 'created' WHERE vehicle_id = ? AND session_id = ?",
                     (ticket_id, vehicle_id, session_id))


def mark_ticket_unknown(db_path: Path, vehicle_id: str, session_id: str) -> None:
    """Ответа ТОиР нет — заявка могла создаться. Место НЕ освобождаем: повторная отправка
    создала бы вторую заявку. Снять блокировку можно вручную после сверки с ТОиР."""
    with connect(db_path) as conn:
        conn.execute("UPDATE tickets SET status = 'unknown' WHERE vehicle_id = ? AND session_id = ?"
                     " AND status = 'sending'", (vehicle_id, session_id))


def release_ticket(db_path: Path, vehicle_id: str, session_id: str) -> None:
    """Отправка не удалась — освобождаем место, чтобы механик мог повторить."""
    with connect(db_path) as conn:
        conn.execute("DELETE FROM tickets WHERE vehicle_id = ? AND session_id = ? AND status = 'sending'",
                     (vehicle_id, session_id))


def list_tickets(db_path: Path, vehicle_id: str) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT t.*, d.payload FROM tickets t JOIN ticket_drafts d ON d.draft_id = t.draft_id"
            " WHERE t.vehicle_id = ? AND t.status = 'created' ORDER BY t.created_at", (vehicle_id,)).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


def created_ticket_ids(db_path: Path) -> dict[tuple[str, str], str]:
    """(vehicle_id, session_id) → ticket_id созданных заявок по всему парку — один запрос для списка машин."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT vehicle_id, session_id, ticket_id FROM tickets WHERE status = 'created'").fetchall()
    return {(r["vehicle_id"], r["session_id"]): r["ticket_id"] for r in rows}
