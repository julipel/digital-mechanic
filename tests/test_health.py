"""/health и журнал событий. Заглушки работают в процессе через ASGITransport — сеть не нужна."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.main import REQUEST_ID_HEADER, create_app
from app.storage.events import log_event, log_llm_call, request_id_var
from tests.conftest import DownTransport, RoutingTransport


def test_health_ok_when_both_mocks_up(settings, mock_transport):
    client = TestClient(create_app(settings, http_transport=mock_transport))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["services"]["mock_ml"]["ok"] and body["services"]["mock_toir"]["ok"]


def test_health_degraded_when_mock_down(settings, mock_transport):
    transport = RoutingTransport({"mock-ml.test": mock_transport.routes["mock-ml.test"]})
    client = TestClient(create_app(settings, http_transport=transport))
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["services"]["mock_ml"]["ok"] is True
    assert body["services"]["mock_toir"] == {"ok": False, "error": "ConnectError"}


def test_health_degraded_when_all_down(settings):
    client = TestClient(create_app(settings, http_transport=DownTransport()))
    assert client.get("/health").status_code == 503


def test_request_id_header_roundtrip(settings):
    client = TestClient(create_app(settings, http_transport=DownTransport()))
    r = client.get("/health", headers={REQUEST_ID_HEADER: "abc123"})
    assert r.headers[REQUEST_ID_HEADER] == "abc123"
    assert len(client.get("/health").headers[REQUEST_ID_HEADER]) == 16


def test_log_event_writes_sqlite_and_stdout(settings, capsys):
    token = request_id_var.set("rid-1")
    try:
        log_event(settings.db_path, "diagnose_called", vehicle_id="B418HK98", session_id="S-0009",
                  data={"note": "проверка", "openai_api_key": "sk-secret"})
    finally:
        request_id_var.reset(token)

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["event"] == "diagnose_called" and line["request_id"] == "rid-1"
    assert line["data"]["openai_api_key"] == "***"

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute("SELECT request_id, vehicle_id, session_id, event_type, data FROM events").fetchone()
    assert row[:4] == ("rid-1", "B418HK98", "S-0009", "diagnose_called")
    assert "sk-secret" not in row[4] and "проверка" in row[4]


def test_log_event_rejects_unknown_type(settings):
    with pytest.raises(ValueError):
        log_event(settings.db_path, "something_else")


def test_log_llm_call(settings):
    log_llm_call(settings.db_path, scenario="question", step="agent_final", model="m",
                 input_tokens=100, cached_input_tokens=20, output_tokens=30, latency_ms=500)
    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute("SELECT scenario, step, input_tokens, cached_input_tokens, output_tokens"
                           " FROM llm_calls").fetchone()
    assert row == ("question", "agent_final", 100, 20, 30)
    with pytest.raises(ValueError):
        log_llm_call(settings.db_path, scenario="other", step="x", model="m")
