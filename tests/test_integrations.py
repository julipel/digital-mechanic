"""Парк ТС и клиенты Mock API: каждый статус ответа заглушки → понятный результат или доменная ошибка."""
import asyncio
import json
import sqlite3

import pytest

from app.integrations.errors import BadRequest, InvalidResponse, LegacySchema, NotFound, ServiceUnavailable
from app.integrations.ml_client import HttpMlClient
from app.integrations.models import Reference, TicketRequest
from app.integrations.toir_client import HttpToirClient, submit_ticket
from tests.conftest import ML_URL, TOIR_URL, DownTransport, fixed_response, needs_data


def run(coro):
    return asyncio.run(coro)


def ticket(**over) -> TicketRequest:
    data = dict(vehicle_id="B418HK98", session_id="S-0009", anomaly_type="F01",
                conclusion="Повышенный зазор в клапанном механизме", urgency="medium",
                operations=["Проверить и отрегулировать зазоры клапанов"], spare_parts=[],
                references=[Reference(manual="236-3902150-Б РЭ", pdf_page=50)])
    data.update(over)
    return TicketRequest(**data)


# --- Парк ТС -------------------------------------------------------------------------

@needs_data
def test_fleet_loads_all_vehicles_with_families(fleet):
    vehicles = fleet.list()
    assert len(vehicles) == 50
    assert all(v.engine_family for v in vehicles), "у каждой машины есть семейство двигателя"
    assert fleet.get("B418HK98").engine_family == "yamz-236"
    assert fleet.get("b418hk98") is None, "регистр госномера важен"


@needs_data
def test_fleet_sessions_sorted_by_time(fleet):
    for v in fleet.list():
        times = [s.checked_at for s in v.sessions]
        assert times == sorted(times) and 3 <= len(times) <= 5


@needs_data
def test_fleet_history_descriptions_are_sanitized(fleet):
    flagged = {(v.vehicle_id, s.session_id) for v in fleet.list() for s in v.sessions if s.injection_detected}
    assert flagged == {("T792HO98", "S-0043"), ("H142OX78", "S-0113"), ("E622KM47", "S-0195")}
    for v in fleet.list():
        for s in v.sessions:
            assert "SYSTEM" not in s.description and "промпт" not in s.description


# --- Mock ML ---------------------------------------------------------------------------

@needs_data
@pytest.mark.parametrize("vid, sid, status", [
    ("B418HK98", "S-0009", "anomaly"),
    ("P456AT178", "S-0001", "normal"),
    ("P456AT178", "S-0003", "check_failed"),
])
def test_ml_statuses(mock_transport, vid, sid, status):
    d = run(HttpMlClient(ML_URL, transport=mock_transport).diagnose(vid, sid))
    assert (d.vehicle_id, d.session_id, d.status) == (vid, sid, status)
    if status == "check_failed":
        assert d.anomaly_score is None and d.signal_quality == "bad"


@needs_data
def test_ml_404_unknown_vehicle_and_foreign_session(mock_transport):
    client = HttpMlClient(ML_URL, transport=mock_transport)
    with pytest.raises(NotFound):
        run(client.diagnose("X000XX00", "S-0001"))
    with pytest.raises(NotFound):
        run(client.diagnose("B418HK98", "S-0001"))  # проверка чужой машины


@needs_data
def test_ml_422_bad_schema(mock_transport):
    with pytest.raises(BadRequest) as e:
        run(HttpMlClient(ML_URL, transport=mock_transport).diagnose("", "S-0001"))
    assert e.value.status_code == 422


def test_ml_400_and_5xx_and_down():
    with pytest.raises(BadRequest):
        run(HttpMlClient(ML_URL, transport=fixed_response(400, "There was an error parsing the body")).diagnose("A", "S"))
    with pytest.raises(ServiceUnavailable):
        run(HttpMlClient(ML_URL, transport=fixed_response(500, "boom")).diagnose("A", "S"))
    with pytest.raises(ServiceUnavailable):
        run(HttpMlClient(ML_URL, transport=DownTransport()).diagnose("A", "S"))


@pytest.mark.parametrize("body", [
    {"vehicle_id": "A", "session_id": "S", "status": "anomaly", "anomaly_type": None, "anomaly_title": None,
     "anomaly_score": 0.7, "confidence": 0.8, "signal_quality": "good", "description": "", "model_version": "x"},
    {"vehicle_id": "A", "session_id": "S", "status": "normal", "anomaly_type": "F01", "anomaly_title": None,
     "anomaly_score": 0.1, "confidence": 0.8, "signal_quality": "good", "description": "", "model_version": "x"},
    {"vehicle_id": "A", "session_id": "S", "status": "anomaly", "anomaly_type": "F99", "anomaly_title": None,
     "anomaly_score": 0.7, "confidence": 0.8, "signal_quality": "good", "description": "", "model_version": "x"},
    {"vehicle_id": "A", "session_id": "S", "status": "anomaly", "anomaly_type": "F01", "anomaly_title": None,
     "anomaly_score": 1.7, "confidence": 0.8, "signal_quality": "good", "description": "", "model_version": "x"},
    {"vehicle_id": "OTHER", "session_id": "S", "status": "normal", "anomaly_type": None, "anomaly_title": None,
     "anomaly_score": 0.1, "confidence": 0.8, "signal_quality": "good", "description": "", "model_version": "x"},
    {"vehicle_id": "A", "session_id": "S", "status": "normal", "anomaly_type": None, "anomaly_title": None,
     "anomaly_score": 0.1, "confidence": 0.8, "signal_quality": "good"},  # нет description и model_version
    "не json",
])
def test_ml_invalid_response(body):
    with pytest.raises(InvalidResponse):
        run(HttpMlClient(ML_URL, transport=fixed_response(200, body)).diagnose("A", "S"))


# --- Mock ТОиР -------------------------------------------------------------------------

@needs_data
def test_toir_accepts_contract_ticket(mock_transport, settings):
    ticket_id = run(submit_ticket(HttpToirClient(TOIR_URL, transport=mock_transport), ticket(), settings.db_path))
    assert ticket_id.startswith("TKT-")
    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute("SELECT event_type, data FROM events").fetchone()
    assert row[0] == "ticket_created" and json.loads(row[1])["ticket_id"] == ticket_id


@needs_data
def test_toir_404_logged_as_toir_error(mock_transport, settings):
    with pytest.raises(NotFound):
        run(submit_ticket(HttpToirClient(TOIR_URL, transport=mock_transport),
                          ticket(session_id="S-0001"), settings.db_path))
    with sqlite3.connect(settings.db_path) as conn:
        assert conn.execute("SELECT event_type FROM events").fetchone()[0] == "toir_error"


@pytest.mark.parametrize("code, error", [(400, BadRequest), (422, BadRequest), (418, LegacySchema),
                                         (503, ServiceUnavailable)])
def test_toir_error_codes(code, error):
    with pytest.raises(error):
        run(HttpToirClient(TOIR_URL, transport=fixed_response(code, {"detail": "x"})).create_ticket(ticket()))


def test_toir_unexpected_success_body():
    with pytest.raises(InvalidResponse):
        run(HttpToirClient(TOIR_URL, transport=fixed_response(201, {"status": "created"})).create_ticket(ticket()))


def test_ticket_model_uses_v6_field_names():
    body = ticket().model_dump(mode="json")
    assert set(body) == {"vehicle_id", "session_id", "anomaly_type", "conclusion", "urgency",
                         "operations", "spare_parts", "references"}
    assert set(body["references"][0]) == {"manual", "pdf_page"}
    with pytest.raises(ValueError):
        ticket(operations=[])
    with pytest.raises(ValueError):
        TicketRequest(**{**body, "priority": "high"})  # старое имя поля не проходит


def test_empty_openai_base_url_in_environment_falls_back_to_official_api(monkeypatch):
    # docker compose передаёт пустой OPENAI_BASE_URL из .env в окружение; SDK при base_url=None
    # прочитал бы эту пустую строку и не смог бы подключиться (APIConnectionError).
    from app.config import OPENAI_DEFAULT_BASE_URL, Settings
    from app.integrations.llm import get_client

    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings(_env_file=None)
    assert settings.openai_base_url == OPENAI_DEFAULT_BASE_URL
    assert str(get_client(settings).base_url).startswith(OPENAI_DEFAULT_BASE_URL)
