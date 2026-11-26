"""SQLite: подключение и схема. Одно соединение на операцию — просто и безопасно для потоков."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    request_id  TEXT,
    vehicle_id  TEXT,
    session_id  TEXT,
    event_type  TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_vehicle ON events(vehicle_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,
    request_id          TEXT,
    scenario            TEXT NOT NULL,
    step                TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    latency_ms          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_request ON llm_calls(request_id);

-- Черновики заявок: payload — тело заявки ТОиР (JSON), content_hash — его sha256.
-- В таблицу попадают только черновики, прошедшие валидатор.
CREATE TABLE IF NOT EXISTS ticket_drafts (
    draft_id     TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    request_id   TEXT,
    vehicle_id   TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    extra        TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'draft'   -- draft | cancelled | sent
);
CREATE INDEX IF NOT EXISTS idx_drafts_session ON ticket_drafts(vehicle_id, session_id);

-- Заявки: одна проверка — одна заявка. Строка вставляется ДО отправки в ТОиР
-- (status=sending) — так два одновременных подтверждения не создадут две заявки.
CREATE TABLE IF NOT EXISTS tickets (
    vehicle_id   TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    draft_id     TEXT NOT NULL,
    ticket_id    TEXT,
    status       TEXT NOT NULL,                  -- sending | created | unknown
    created_at   TEXT NOT NULL,
    PRIMARY KEY (vehicle_id, session_id)
);
"""

_initialized: set[Path] = set()


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Открывает соединение, при первом обращении к файлу создаёт схему; коммитит на выходе."""
    db_path = Path(db_path)
    if db_path not in _initialized:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if db_path not in _initialized:
            conn.executescript(SCHEMA)
            _initialized.add(db_path)
        yield conn
        conn.commit()
    finally:
        conn.close()
