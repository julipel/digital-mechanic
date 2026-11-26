"""Живой смоук-тест агента на С1–С3: настоящая модель (AGENT_MODEL из .env), настоящий индекс,
заглушка ML в процессе. Пропускается без ключа, модели или индекса. Проверяются только
инварианты кода — текст ответа модели не сравнивается."""
import asyncio

import pytest

from app.agent.loop import AgentDeps, make_chat_model, run_turn
from app.agent.state import ConversationStore
from app.integrations.fleet import load_fleet
from app.integrations.ml_client import HttpMlClient
from app.policy import modes
from app.retrieval.manuals import manuals_for_family
from tests.conftest import HAS_INDEX, ML_URL, kb_settings

_S = kb_settings()
live = pytest.mark.skipif(
    not (HAS_INDEX and _S.agent_model and _S.openai_api_key is not None and _S.openai_api_key.get_secret_value()),
    reason="нужны AGENT_MODEL, OPENAI_API_KEY и индекс в INDEX_DIR")


@pytest.fixture
def deps(mock_transport, tmp_path):
    settings = _S.model_copy(update={"db_path": tmp_path / "live.sqlite"})
    fleet = load_fleet(settings.vehicles_path, settings.config_dir, settings.fault_types_path)
    return AgentDeps(settings=settings, fleet=fleet, ml=HttpMlClient(ML_URL, transport=mock_transport),
                     model=make_chat_model(settings))


def turn(conv, deps, text, scenario):
    async def collect():
        return [e async for e in run_turn(conv, text, deps, scenario=scenario)]
    events = asyncio.run(collect())
    return events, next(e for e in events if e["type"] == "done")


def own_manuals(deps, vehicle_id):
    family = deps.fleet.get(vehicle_id).engine_family
    return {m.manual for m in manuals_for_family(deps.settings.kb_dir, family)}


@live
def test_live_c1_c2(deps):
    conv = ConversationStore().create("B418HK98", "S-0009")
    events, result = turn(conv, deps, "Получи результат диагностики и объясни его.", "explain")
    assert not result["llm_failed"] and result["text"]
    assert any(e["type"] == "diagnosis" for e in events)
    assert result["citations"], "ответ без единой ссылки на руководство"
    assert {c["manual"] for c in result["citations"]} <= own_manuals(deps, "B418HK98")

    _, result = turn(conv, deps, "Можно ли своим ходом перегнать машину на базу?", "question")
    assert not result["llm_failed"] and result["text"]
    assert {c["manual"] for c in result["citations"]} <= own_manuals(deps, "B418HK98")

    events, result = turn(conv, deps, "Создай заявку на дефектовку.", "ticket")
    assert not result["llm_failed"]
    draft = next((e for e in events if e["type"] == "draft"), None)
    if draft is not None:          # модель могла не уложиться в валидатор — это не падение
        assert {r["manual"] for r in draft["ticket"]["references"]} <= own_manuals(deps, "B418HK98")


@live
def test_live_c3_check_failed(deps):
    conv = ConversationStore().create("P456AT178", "S-0003")
    events, result = turn(conv, deps, "Получи результат диагностики и объясни его.", "explain")
    assert conv.mode.mode == modes.CHECK_FAILED
    assert result["text"].startswith("⚠")
    assert not any(e["type"] == "draft" for e in events)
