"""B1: память по ТС и журнал машины. Без LLM и сети: история — детерминированная функция
(policy/history), журнал собирается из events и vehicles.json (storage/journal), в API —
JSON и CSV по машине. Тексты руководств в тесты не кладём — только коды, номера и страницы."""
import json
from datetime import datetime

import pytest

from app.integrations.fleet import fault_titles
from app.policy import modes
from app.policy.history import vehicle_history
from app.storage.events import log_event
from app.storage.journal import journal_csv, vehicle_journal
from tests.test_agent import DIAG, chunk_id
from tests.test_api import client_for, of_type, sse  # noqa: F401 — client_for нужен как фикстура


def _ticket(session_id, ticket_id="TKT-1", urgency="high", code="F01"):
    return {"session_id": session_id, "ticket_id": ticket_id, "created_at": "2026-11-20T10:00:00+00:00",
            "payload": {"anomaly_type": code, "urgency": urgency, "operations": ["Проверить зазоры"]}}


# --- policy/history ----------------------------------------------------------------------------

def test_repeat_of_same_code_counted(fleet, settings):
    h = vehicle_history(fleet.get("B418HK98"), "S-0009", [], settings.config_dir)
    assert h.repeat_count == 2 and "F01 — 2-я проверка подряд" in h.observations[0]


def test_repeat_survives_failed_recording_between(fleet, settings):
    h = vehicle_history(fleet.get("O658MX98"), "S-0063", [], settings.config_dir)
    assert h.repeat_count == 2 and "сбой записи" in h.observations[0]


def test_different_code_breaks_repeat(fleet, settings):
    h = vehicle_history(fleet.get("P348EE78"), "S-0035", [], settings.config_dir)      # F04 → F05
    assert h.repeat_count == 1 and "новое" in h.observations[0]


def test_normal_after_anomaly_noted_with_ticket(fleet, settings):
    h = vehicle_history(fleet.get("X792BH98"), "S-0026", [_ticket("S-0025", code="F06")], settings.config_dir)
    assert h.repeat_count == 0                                    # выбранная — норма, рост не тянем
    assert any("следующая проверка — норма" in o and "TKT-1" in o for o in h.observations)
    assert next(c for c in h.checks if c["session_id"] == "S-0025")["ticket_id"] == "TKT-1"


def test_only_checks_up_to_selected(fleet, settings):
    h = vehicle_history(fleet.get("X792BH98"), "S-0024", [], settings.config_dir)
    assert [c["session_id"] for c in h.checks] == ["S-0024", "S-0023"] and h.checks[0]["selected"]


def test_ticket_for_selected_reported(fleet, settings):
    h = vehicle_history(fleet.get("B418HK98"), "S-0009", [_ticket("S-0009", "TKT-9")], settings.config_dir)
    assert h.ticket_for_selected == "TKT-9" and any("Вторую не создать" in o for o in h.observations)


def test_inapplicable_code_marked_and_not_counted(fleet, settings):
    h = vehicle_history(fleet.get("K354KC98"), "S-0087", [], settings.config_dir)     # F12 у дизеля
    c = h.checks[0]
    assert c["code_not_applicable"] is True and h.repeat_count == 0
    assert c["anomaly_title"] is None and c["anomaly_score"] is None                    # невозможное не пересказать


def test_pending_ticket_for_selected_reported(fleet, settings):
    h = vehicle_history(fleet.get("B418HK98"), "S-0009", [], settings.config_dir, selected_ticket_pending=True)
    assert any("ответ ТОиР не получен" in o for o in h.observations)


def test_first_check_has_nothing_to_compare(fleet, settings):
    v = fleet.get("B418HK98")
    one = v.model_copy(update={"sessions": (v.session("S-0009"),)})
    assert "первая проверка" in vehicle_history(one, "S-0009", [], settings.config_dir).observations[0]


@pytest.mark.parametrize("vehicle_id,session_id", [("T792HO98", "S-0043"), ("H142OX78", "S-0113"),
                                                   ("E622KM47", "S-0195")])
def test_history_has_no_ml_description(fleet, settings, vehicle_id, session_id):
    h = vehicle_history(fleet.get(vehicle_id), session_id, [], settings.config_dir)
    assert all("description" not in c for c in h.checks)


def test_unknown_session_raises(fleet, settings):
    with pytest.raises(KeyError):
        vehicle_history(fleet.get("B418HK98"), "S-0003", [], settings.config_dir)


# --- storage/journal ---------------------------------------------------------------------------

def _journal(settings, fleet, vehicle_id, **kw):
    return vehicle_journal(settings.db_path, fleet.get(vehicle_id), config_dir=settings.config_dir,
                           fault_titles=fault_titles(settings.fault_types_path), **kw)


def test_journal_merges_checks_and_events_newest_first(fleet, settings):
    db = settings.db_path
    log_event(db, "diagnose_result", vehicle_id="T792HO98", session_id="S-0043",
              data={"status": "anomaly", "anomaly_type": "F02", "anomaly_score": 0.85, "confidence": 0.5,
                    "signal_quality": "warning"})
    log_event(db, "injection_detected", vehicle_id="T792HO98", session_id="S-0043", data={"removed_chars": 40})
    log_event(db, "tool_called", vehicle_id="T792HO98", session_id="S-0043", data={"tool": "search_manual"})
    log_event(db, "answer_sent", vehicle_id="T792HO98", session_id="S-0043",
              data={"scenario": "question", "question": "Можно своим ходом?", "answer_short": "Нет.",
                    "citations": [{"doc_id": "yamz-236_re", "manual": "236-3902150-Б РЭ", "pdf_page": 50}]})
    log_event(db, "answer_sent", vehicle_id="B418HK98", session_id="S-0009", data={"scenario": "question"})
    entries = _journal(settings, fleet, "T792HO98")
    kinds = [e["kind"] for e in entries]
    assert kinds[0] == "question" and entries[0]["pages"] == [{"manual": "236-3902150-Б РЭ", "pdf_page": 50}]
    assert {"Помехи при записи", "Низкая уверенность ML-модели",
            "В описании от ML-сервиса посторонний текст"} <= {e["title"] for e in entries}
    assert kinds.count("check") == 4 and kinds.count("question") == 1              # чужая машина не попала
    assert "search_manual" not in json.dumps(entries, ensure_ascii=False)          # техника не для механика
    ts = [datetime.fromisoformat(e["ts"]) for e in entries]
    assert ts == sorted(ts, reverse=True)


def test_journal_filter_and_inapplicable_check(fleet, settings):
    entries = _journal(settings, fleet, "K354KC98", kinds={"check"})
    assert {e["kind"] for e in entries} == {"check"}
    assert "код не подходит этому двигателю" in next(e for e in entries if e["session_id"] == "S-0087")["details"]


def test_journal_check_with_injection_shows_note_not_text(fleet, settings):
    entry = next(e for e in _journal(settings, fleet, "H142OX78") if e["session_id"] == "S-0113")
    assert "посторонний текст" in entry["details"]
    raw = fleet.get("H142OX78").session("S-0113").description
    assert not raw or raw not in json.dumps(entry, ensure_ascii=False)


def test_journal_csv_excel_safe(fleet, settings):
    log_event(settings.db_path, "answer_sent", vehicle_id="B418HK98", session_id="S-0009",
              data={"scenario": "question", "question": "x", "answer_short": "=HYPERLINK(1)"})
    head, *rows = journal_csv(_journal(settings, fleet, "B418HK98")).split("\r\n")
    assert head.startswith("Время (МСК);Тип;Проверка")
    assert any(";'=HYPERLINK(1);" in r for r in rows)


# --- API ---------------------------------------------------------------------------------------

def test_journal_api_after_ticket_created(client_for):
    cid = chunk_id("yamz-236")
    draft_args = {"conclusion": "Стук", "operations": ["Проверить зазоры"], "spare_parts": [], "chunk_ids": [cid]}
    client, _, _ = client_for(DIAG, [(modes.SEARCH_MANUAL, {"query": "клапаны"})],
                              [(modes.DRAFT_TICKET, draft_args)], "Готово.")
    events = sse(client.post("/api/chat", json={"vehicle_id": "B418HK98", "session_id": "S-0009",
                                                "message": "Заявку", "scenario": "ticket"}))
    draft = of_type(events, "draft")[0]
    ticket_id = client.post(f"/api/tickets/{draft['draft_id']}/confirm",
                            json={"hash": draft["content_hash"]}).json()["ticket_id"]

    tickets = client.get("/api/vehicles/B418HK98/journal?kind=ticket").json()["entries"]
    assert tickets[0]["title"].startswith(f"Заявка {ticket_id} создана")
    assert "Проверить зазоры" in tickets[0]["details"]
    assert tickets[0]["pages"] == [{"manual": "236-3902150-Б РЭ", "pdf_page": 50}]
    assert any(e["title"].startswith("Черновик заявки") for e in tickets)

    csv = client.get("/api/vehicles/B418HK98/journal.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert 'filename="journal_B418HK98.csv"' in csv.headers["content-disposition"]
    assert csv.content.startswith("﻿".encode("utf-8")) and ticket_id in csv.content.decode("utf-8")


def test_journal_api_errors(client_for):
    client, _, _ = client_for()
    assert client.get("/api/vehicles/b418hk98/journal").status_code == 404
    assert client.get("/api/vehicles/B418HK98/journal?kind=secret").status_code == 422
    assert client.get("/api/vehicles/B418HK98/journal?kind=check,warning").status_code == 200
