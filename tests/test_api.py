"""API интерфейса механика без LLM и сети: модель — сценарий, заглушки — в процессе,
поиск — синтетические фрагменты (как в test_agent). Проверяется слой HTTP: SSE-события,
404 на чужую проверку, подтверждение заявки по id и хэшу, одна проверка — одна заявка,
рендер страницы только из манифеста и с ограничением частоты."""
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.agent import messages as msg
from app.main import create_app
from app.policy import modes
from app.retrieval.pages import RateLimiter
from tests.conftest import kb_settings, needs_kb
from tests.test_agent import DIAG, MANUAL, PART, FakeSearch, ScriptedModel, chunk_id


@pytest.fixture
def client_for(settings, mock_transport):
    def factory(*rounds, **overrides):
        s = settings.model_copy(update={"auth_disabled": True, **overrides})
        model = ScriptedModel(*rounds)
        app = create_app(s, mock_transport, chat_model=model, search_fn=FakeSearch(),
                         manuals_fn=lambda fam: [MANUAL[fam]])
        return TestClient(app), model, app
    return factory


def sse(response) -> list[dict]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.split("\n") if line.startswith("data: ")]


def of_type(events, kind) -> list[dict]:
    return [e for e in events if e["type"] == kind]


def test_vehicles_list_and_card(client_for):
    client, _, _ = client_for()
    vehicles = client.get("/api/vehicles").json()
    b418 = next(v for v in vehicles if v["vehicle_id"] == "B418HK98")
    assert b418["engine_model"] == "ЯМЗ-236М2" and b418["last_check"]["session_id"] == "S-0009"
    card = client.get("/api/vehicles/B418HK98").json()
    assert [s["session_id"] for s in card["sessions"]][0] == "S-0009"      # новые сверху
    assert card["tickets"] == []
    assert "description" not in card["sessions"][0]                     # недоверенный текст не отдаём
    assert client.get("/api/vehicles/b418hk98").status_code == 404      # регистр важен


def test_diagnose_foreign_session_404(client_for):
    client, model, _ = client_for()
    r = client.post("/api/diagnose", json={"vehicle_id": "B418HK98", "session_id": "S-0003"})
    assert r.status_code == 404 and model.calls == []                    # модель даже не вызывалась


def test_c1_diagnose_then_ticket_confirmed_once(client_for, settings):
    cid = chunk_id("yamz-236")
    draft_args = {"conclusion": "Стук в клапанном механизме", "operations": ["Проверить и отрегулировать зазоры"],
                  "spare_parts": [{"part_number": PART, "name": "Клапан впускной", "chunk_id": cid}],
                  "chunk_ids": [cid]}
    client, model, _ = client_for(
        DIAG, [(modes.SEARCH_MANUAL, {"query": "зазоры клапанов"})], f"Отклонение F01.\nПроверьте зазоры [{cid}].\n",
        [(modes.SEARCH_MANUAL, {"query": "клапаны"})], [(modes.DRAFT_TICKET, draft_args)], "Черновик готов.")

    events = sse(client.post("/api/diagnose", json={"vehicle_id": "B418HK98", "session_id": "S-0009"}))
    assert events[0]["type"] == "conversation"
    conv_id = events[0]["conversation_id"]
    diag = of_type(events, "diagnosis")[0]
    assert diag["mode"] == modes.ANOMALY and diag["urgency"]["level"] == "high" and diag["warnings_ru"] == []
    done = of_type(events, "done")[0]
    assert "[236-3902150-Б РЭ, стр. 50]" in done["text"] and done["citations"][0]["doc_id"] == "yamz-236_re"
    assert model.calls[0]["tools"] == [modes.GET_DIAGNOSIS, modes.SEARCH_MANUAL]   # выбор — за моделью

    events = sse(client.post("/api/chat", json={"conversation_id": conv_id, "message": "Создай заявку",
                                                "scenario": "ticket"}))
    draft = of_type(events, "draft")[0]
    assert draft["ticket"]["anomaly_type"] == "F01" and len(draft["content_hash"]) == 64

    wrong = client.post(f"/api/tickets/{draft['draft_id']}/confirm", json={"hash": "0" * 64})
    assert wrong.status_code == 409
    ok = client.post(f"/api/tickets/{draft['draft_id']}/confirm", json={"hash": draft["content_hash"]})
    assert ok.status_code == 200 and ok.json()["ticket_id"].startswith("TKT")
    again = client.post(f"/api/tickets/{draft['draft_id']}/confirm", json={"hash": draft["content_hash"]})
    assert again.status_code == 409                                       # одна проверка — одна заявка
    tickets = client.get("/api/vehicles/B418HK98").json()["tickets"]
    assert [t["ticket_id"] for t in tickets] == [ok.json()["ticket_id"]]


def test_cancelled_draft_cannot_be_sent(client_for):
    cid = chunk_id("yamz-236")
    draft_args = {"conclusion": "Стук", "operations": ["Проверить зазоры"], "spare_parts": [], "chunk_ids": [cid]}
    client, _, _ = client_for(DIAG, [(modes.SEARCH_MANUAL, {"query": "клапаны"})],
                              [(modes.DRAFT_TICKET, draft_args)], "Готово.")
    events = sse(client.post("/api/chat", json={"vehicle_id": "B418HK98", "session_id": "S-0009",
                                                "message": "Заявку", "scenario": "ticket"}))
    draft = of_type(events, "draft")[0]
    assert client.post(f"/api/tickets/{draft['draft_id']}/cancel").status_code == 200
    r = client.post(f"/api/tickets/{draft['draft_id']}/confirm", json={"hash": draft["content_hash"]})
    assert r.status_code == 409 and "отмен" in r.json()["detail"]


def test_c3_check_failed_warning_first_no_ticket_tool(client_for):
    client, model, _ = client_for(DIAG, "Запись не удалась.")
    events = sse(client.post("/api/diagnose", json={"vehicle_id": "P456AT178", "session_id": "S-0003"}))
    assert of_type(events, "diagnosis")[0]["mode"] == modes.CHECK_FAILED
    assert of_type(events, "done")[0]["text"].startswith("⚠ " + msg.WARNINGS_RU[modes.W_CHECK_FAILED])
    assert modes.DRAFT_TICKET not in model.calls[1]["tools"]


def test_chat_starts_conversation_and_unknown_conversation_404(client_for):
    client, _, _ = client_for("Ответ без поиска.")
    events = sse(client.post("/api/chat", json={"vehicle_id": "B418HK98", "session_id": "S-0009",
                                                "message": "Можно ли своим ходом перегнать машину на базу?"}))
    assert events[0]["type"] == "conversation" and of_type(events, "done")
    assert client.post("/api/chat", json={"conversation_id": "нет-такого", "message": "?"}).status_code == 404
    assert client.post("/api/chat", json={"message": "?"}).status_code == 404
    assert client.post("/api/chat", json={"vehicle_id": "B418HK98", "session_id": "S-0009",
                                          "message": "?", "scenario": "indexing"}).status_code == 422


def test_llm_failure_streams_message_not_500(client_for):
    client, _, _ = client_for(RuntimeError("провайдер недоступен"))
    events = sse(client.post("/api/diagnose", json={"vehicle_id": "B418HK98", "session_id": "S-0009"}))
    done = of_type(events, "done")[0]
    assert done["llm_failed"] and msg.LLM_UNAVAILABLE in done["text"]


@pytest.mark.parametrize("vehicle_id,session_id", [("T792HO98", "S-0043"), ("H142OX78", "S-0113"),
                                                   ("E622KM47", "S-0195")])
def test_injection_flagged_in_verdict(client_for, vehicle_id, session_id):
    client, _, _ = client_for(DIAG, "Разбор проверки.")
    response = client.post("/api/diagnose", json={"vehicle_id": vehicle_id, "session_id": session_id})
    diag = of_type(sse(response), "diagnosis")[0]
    assert modes.W_INJECTION in diag["warnings"]
    assert msg.WARNINGS_RU[modes.W_INJECTION] in diag["warnings_ru"]
    raw = next(s for v in json.load(open(client.app.state.settings.vehicles_path, encoding="utf-8"))["vehicles"]
               if v["vehicle_id"] == vehicle_id for s in v["sessions"] if s["session_id"] == session_id)
    assert raw["description"] not in response.text            # исходный текст с инструкцией не уходит


def test_only_cited_pages_are_served(client_for):
    cid = chunk_id("yamz-236")
    client, _, app = client_for(DIAG, [(modes.SEARCH_MANUAL, {"query": "зазоры"})], f"Зазоры [{cid}].\n")
    services = app.state.services
    assert client.get("/api/manuals/yamz-236_re/pages/50.png").status_code == 404    # ещё не цитировалась
    sse(client.post("/api/diagnose", json={"vehicle_id": "B418HK98", "session_id": "S-0009"}))
    assert services.cited_pages == {("yamz-236_re", 50)}
    assert client.get("/api/manuals/yamz-236_re/pages/50.png").status_code == 503    # KB_DIR не задан
    assert client.get("/api/manuals/yamz-236_re/pages/51.png").status_code == 404    # соседняя — нет
    services.page_limiter = RateLimiter(1)
    client.get("/api/manuals/yamz-236_re/pages/50.png")
    assert client.get("/api/manuals/yamz-236_re/pages/50.png").status_code == 429


@needs_kb
def test_page_render_from_manifest(client_for, tmp_path):
    kb = kb_settings()
    client, _, app = client_for(kb_dir=kb.kb_dir, page_cache_dir=tmp_path / "pages")
    from app.retrieval.manuals import load_manuals
    manual = load_manuals(kb.kb_dir)[0]
    cited = app.state.services.cited_pages
    cited |= {(manual.doc_id, 2), (manual.doc_id, manual.pages + 1), (manual.doc_id, 0), ("../secret", 1)}
    r = client.get(f"/api/manuals/{manual.doc_id}/pages/2.png")
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/api/manuals/{manual.doc_id}/pages/{manual.pages + 1}.png").status_code == 404
    assert client.get("/api/manuals/..%2Fsecret/pages/1.png").status_code == 404
    assert client.get(f"/api/manuals/{manual.doc_id}/pages/0.png").status_code == 404


def test_api_requires_login(settings, mock_transport):
    s = settings.model_copy(update={"app_password": SecretStr("секрет")})
    client = TestClient(create_app(s, mock_transport))
    assert client.get("/api/vehicles").status_code == 401
    assert client.get("/api/manuals/yamz-236_re/pages/1.png").status_code == 401


def test_earlier_anomaly_shown_when_last_check_is_normal(client_for):
    """Ловушка данных: после отклонения пришла «норма». Отклонение не должно прятаться за последней проверкой."""
    client, _, _ = client_for()
    vehicles = {v["vehicle_id"]: v for v in client.get("/api/vehicles").json()}
    k354 = vehicles["K354KC98"]
    assert k354["last_check"]["session_id"] == "S-0088" and k354["last_check"]["status"] == "normal"
    assert k354["earlier_anomaly"]["session_id"] == "S-0087"
    assert k354["earlier_anomaly"]["anomaly_type"] == "F12" and k354["earlier_anomaly"]["ticket_id"] is None
    card = client.get("/api/vehicles/K354KC98").json()
    assert card["earlier_anomaly"]["session_id"] == "S-0087"
    # последняя проверка — сама отклонение: отдельной пометки нет, оно и так видно
    assert vehicles["B418HK98"]["earlier_anomaly"] is None
    # пометка — только показ: машины, где отклонений не было, без неё
    assert all(v["earlier_anomaly"] is None or v["earlier_anomaly"]["session_id"] != v["last_check"]["session_id"]
               for v in vehicles.values())


def test_trend_endpoint_as_of_selected_check(client_for):
    client, model, _ = client_for()
    latest = client.get("/api/vehicles/B418HK98/trend").json()
    assert latest["built"] and latest["label"] == "Оценка тенденции" and latest["n_points"] == 3
    # на момент первой проверки точек меньше трёх — прогноза нет, причина названа
    first = client.get("/api/vehicles/B418HK98").json()["sessions"][-1]["session_id"]
    early = client.get(f"/api/vehicles/B418HK98/trend?session_id={first}").json()
    assert early["built"] is False and "Мало проверок" in early["reason"]
    assert client.get("/api/vehicles/B418HK98/trend?session_id=S-0003").status_code == 404   # чужая проверка
    assert client.get("/api/vehicles/b418hk98/trend").status_code == 404
    assert model.calls == []                                                               # без модели


def test_gpt6_with_chat_api_fails_at_startup(tmp_path):
    """gpt-6* работают только через Responses API: с AGENT_API=chat приложение не стартует,
    в ошибке — что поправить в .env. По умолчанию AGENT_API=responses."""
    import pytest
    from app.config import Settings
    from app.main import create_app
    base = dict(_env_file=None, db_path=tmp_path / "t.sqlite", app_password="x")
    assert Settings(**base).agent_api == "responses"
    with pytest.raises(ValueError, match="AGENT_API=responses"):
        create_app(Settings(**base, agent_model="gpt-6.2-sol", agent_api="chat"))
    create_app(Settings(**base, agent_model="gpt-6.2-sol", agent_api="responses"))
    create_app(Settings(**base, agent_model="other-model", agent_api="chat"))   # другие провайдеры — можно
