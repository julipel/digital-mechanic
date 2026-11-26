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
