"""/health и журнал событий. Заглушки работают в процессе через ASGITransport — сеть не нужна."""
import json
import sqlite3
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.main import REQUEST_ID_HEADER, create_app
from app.storage.events import log_event, log_llm_call, request_id_var

MOCKS_DIR = PROJECT_ROOT / "mocks"
ML_URL = "http://mock-ml.test"
TOIR_URL = "http://mock-toir.test"


class RoutingTransport(httpx.AsyncBaseTransport):
    """Направляет запрос в нужное ASGI-приложение по хосту."""

    def __init__(self, routes: dict[str, httpx.AsyncBaseTransport]):
        self.routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self.routes.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError("нет маршрута", request=request)
        return await transport.handle_async_request(request)


class DownTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("сервис недоступен", request=request)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, mock_ml_url=ML_URL, mock_toir_url=TOIR_URL,
                    db_path=tmp_path / "test.sqlite")


@pytest.fixture
def mock_transport() -> RoutingTransport:
    if not (MOCKS_DIR / "data" / "vehicles.json").exists():
        pytest.skip("нет mocks/data/vehicles.json (не публикуется в git)")
    sys.path.insert(0, str(MOCKS_DIR))
    from mock_ml.app import app as ml_app
    from mock_toir.app import app as toir_app
    return RoutingTransport({
        "mock-ml.test": httpx.ASGITransport(app=ml_app),
        "mock-toir.test": httpx.ASGITransport(app=toir_app),
    })


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
