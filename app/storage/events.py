"""Журнал событий и учёт вызовов LLM.

- log_event: значимое действие → таблица `events` и одна JSON-строка в stdout.
  vehicle_id обязателен для событий, связанных с машиной (передаётся вызывающим кодом).
- log_llm_call: каждый вызов LLM/эмбеддингов → таблица `llm_calls` (для отчёта о стоимости).
- request_id: одно действие механика (сообщение или кнопка). Ставится middleware
  в main.py через contextvar, поэтому функции журнала не требуют передавать его явно.

Секреты не пишутся: ключи с «опасными» именами в data заменяются на «***».
"""
import json
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.storage.db import connect

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

EVENT_TYPES = frozenset({
    "diagnose_called", "diagnose_result", "injection_detected", "ml_inconsistent",
    "tool_called", "search_results", "trend_computed", "answer_sent",
    "ticket_draft", "ticket_rejected", "ticket_confirmed", "ticket_created", "toir_error",
    "error",
})

SCENARIOS = frozenset({"explain", "question", "ticket", "indexing"})

_SECRET_MARKERS = ("api_key", "apikey", "authorization", "password", "secret", "system_prompt")

def _emit_stdout(record: dict[str, Any]) -> None:
    """Одна JSON-строка в stdout. Поток берётся в момент записи (работает с перехватом в тестах);
    если консоль не умеет UTF-8 (cp1251 в Windows), строка выводится с \\u-экранированием,
    а не роняет запрос."""
    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        print(line, file=sys.stdout, flush=True)
    except UnicodeEncodeError:
        print(json.dumps(record, ensure_ascii=True, default=str), file=sys.stdout, flush=True)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "***" if any(m in str(k).lower() for m in _SECRET_MARKERS) else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


def log_event(
    db_path: Path,
    event_type: str,
    *,
    vehicle_id: str | None = None,
    session_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Пишет событие в SQLite и stdout. Возвращает записанную запись (удобно для тестов)."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Неизвестный тип события: {event_type}")
    record = {
        "ts": _now(),
        "request_id": request_id_var.get(),
        "vehicle_id": vehicle_id,
        "session_id": session_id,
        "event": event_type,
        "data": _redact(data or {}),
    }
    payload = json.dumps(record["data"], ensure_ascii=False, default=str)
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO events (ts, request_id, vehicle_id, session_id, event_type, data)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (record["ts"], record["request_id"], vehicle_id, session_id, event_type, payload),
        )
    _emit_stdout(record)
    return record


def log_llm_call(
    db_path: Path,
    *,
    scenario: str,
    step: str,
    model: str,
    input_tokens: int = 0,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
) -> None:
    """Пишет один вызов модели. Индексация — scenario='indexing', в стоимость запроса не входит."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Неизвестный сценарий: {scenario}")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO llm_calls (ts, request_id, scenario, step, model, input_tokens,"
            " cached_input_tokens, output_tokens, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), request_id_var.get(), scenario, step, model,
             input_tokens, cached_input_tokens, output_tokens, latency_ms),
        )
