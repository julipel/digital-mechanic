"""Слой policy/: режимы, приоритет, ссылки, валидатор и подтверждение заявки.
Работает без LLM и без базы знаний: фрагменты руководств — синтетические, короткие."""
import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.config import PROJECT_ROOT
from app.integrations.diagnosis import CheckedDiagnosis
from app.integrations.errors import DeliveryUnknown, NotDelivered
from app.integrations.fleet import Session, Vehicle
from app.integrations.models import Diagnosis
from app.policy import modes
from app.policy.applicability import Applicability, check_applicability
from app.policy.citations import CitationError, CitationRegistry
from pydantic import ValidationError

from app.policy.ticket_validator import DraftInput, SparePartInput, part_number_in_text, validate_draft
from app.policy.tickets import TicketError, cancel_draft, confirm_draft, create_draft
from app.policy.urgency import compute_urgency
from app.storage import tickets as store
from app.storage.db import connect

CONFIG = PROJECT_ROOT / "config"
FAMILY = "yamz-236"


@dataclass
class Hit:
    chunk_id: str
    text: str
    pdf_page: int = 50
    doc_id: str = "yamz236_re"
    manual: str = "236-3902150-Б РЭ"
    engine_family: str = FAMILY


HIT_VALVES = Hit("yamz236_re-p050-1", "Регулировка зазоров клапанов. Клапан впускной 236-1007010-В2.")
HIT_ROCKER = Hit("yamz236_rk-p120-2", "Коромысло клапана 236–1007111 (с втулкой).", pdf_page=120,
                 doc_id="yamz236_rk", manual="236.01.01 РК")
HIT_FOREIGN = Hit("yamz536_re-p186-1", "Посторонний стук", pdf_page=186, doc_id="yamz536_re",
                  manual="536.3902150 РЭ", engine_family="yamz-536")


def diag(**over) -> Diagnosis:
    data = dict(vehicle_id="B418HK98", session_id="S-0009", status="anomaly", anomaly_type="F01",
                anomaly_title="Клапанный механизм", anomaly_score=0.7, confidence=0.8,
                signal_quality="good", description="стук", model_version="mock-v1")
    data.update(over)
    if data["status"] == "normal":
        data["anomaly_type"] = None
    return Diagnosis(**data)


def sess(sid: str, days: int, **over) -> Session:
    d = diag(session_id=sid, **over)
    return Session(**d.model_dump(), checked_at=datetime(2026, 11, 1, tzinfo=timezone.utc) + timedelta(days=days))


OK = Applicability(applicable=True)


MANUALS = ("236-3902150-Б РЭ", "236.01.01 РК")


def registry(*hits) -> CitationRegistry:
    reg = CitationRegistry(FAMILY, MANUALS)
    reg.register_all(hits)
    return reg


# --- Режимы и набор инструментов ---------------------------------------------------------

def decide(d: Diagnosis, applicability=OK, injection=False):
    return modes.decide_mode(d, applicability, injection, CONFIG)


def test_check_failed_has_no_diagnosis_and_ticket_tools():
    m = decide(diag(status="check_failed", anomaly_type=None, anomaly_score=None, confidence=None,
                    signal_quality="bad"))
    assert m.mode == modes.CHECK_FAILED
    assert not m.allows(modes.DRAFT_TICKET) and not m.allows(modes.GET_DIAGNOSIS)
    assert m.warnings[0] == modes.W_CHECK_FAILED


def test_bad_signal_overrides_anomaly():
    m = decide(diag(signal_quality="bad"))
    assert m.mode == modes.CHECK_FAILED and not m.allows(modes.DRAFT_TICKET)


@pytest.mark.parametrize("over,warning", [
    ({"signal_quality": "warning"}, modes.W_SIGNAL_WARNING),
    ({"confidence": 0.5}, modes.W_LOW_CONFIDENCE),
    ({"confidence": None}, modes.W_LOW_CONFIDENCE),
])
def test_low_confidence_or_noise_warns(over, warning):
    m = decide(diag(**over))
    assert m.mode == modes.ANOMALY_LOW_CONFIDENCE and warning in m.warnings
    assert m.allows(modes.DRAFT_TICKET)


def test_inconsistent_code_has_no_ticket_tool():
    m = decide(diag(anomaly_type="F11"), Applicability(applicable=False, reason="нет EGR"))
    assert m.mode == modes.ML_INCONSISTENT and not m.allows(modes.DRAFT_TICKET)
    assert m.warnings[0] == modes.W_ML_INCONSISTENT


def test_normal_has_no_ticket_tool():
    m = decide(diag(status="normal", anomaly_score=0.4))
    assert m.mode == modes.NORMAL and not m.allows(modes.DRAFT_TICKET) and not m.warnings


def test_injection_is_warning_not_mode_change():
    m = decide(diag(), injection=True)
    assert m.mode == modes.ANOMALY and m.warnings == (modes.W_INJECTION,)


def test_no_check_mode_offers_diagnosis():
    m = modes.no_check_mode()
    assert m.allows(modes.GET_DIAGNOSIS) and not m.allows(modes.DRAFT_TICKET)


def test_all_fleet_check_failed_and_traps(fleet, settings):
    """По всем проверкам парка: сбой записи → нет заявки; семь ловушек с неприменимым кодом → нет заявки."""
    traps = {("T801PO178", "S-0046"), ("P854BT78", "S-0021"), ("K354KC98", "S-0087"), ("M260EH47", "S-0079"),
             ("Y652CK98", "S-0132"), ("C808CM178", "S-0144"), ("C933KX98", "S-0163")}
    found = set()
    for v in fleet.list():
        for s in v.sessions:
            app = check_applicability(settings.config_dir, v.engine_family, s.anomaly_type)
            m = modes.decide_mode(s, app, s.injection_detected, settings.config_dir)
            if s.status == "check_failed":
                assert m.mode == modes.CHECK_FAILED
            if m.mode == modes.ML_INCONSISTENT:
                found.add((v.vehicle_id, s.session_id))
            if s.status != "anomaly" or not app.applicable:
                assert not m.allows(modes.DRAFT_TICKET), (v.vehicle_id, s.session_id)
    assert traps <= found


# --- Приоритет ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,code,level", [
    (0.55, "F04", "low"), (0.7, "F04", "medium"), (0.85, "F04", "high"),
    (0.55, "F02", "high"),            # низкое давление масла — сразу высокий
    (0.55, "F01", "medium"),          # минимум для клапанного механизма
])
def test_urgency_levels(score, code, level):
    r = compute_urgency(diag(anomaly_type=code, anomaly_score=score), [], CONFIG)
    assert r.level == level and r.reasons


def test_urgency_repeat_and_growth_raise_one_level():
    prev = [sess("S-1", 0, anomaly_type="F04", anomaly_score=0.66)]
    repeat = compute_urgency(diag(anomaly_type="F04", anomaly_score=0.7), prev, CONFIG)
    assert repeat.level == "high" and any("прошлой проверке" in r for r in repeat.reasons)
    prev = [sess("S-1", 0, anomaly_type="F04", anomaly_score=0.5)]
    growth = compute_urgency(diag(anomaly_type="F04", anomaly_score=0.62), prev, CONFIG)
    assert growth.level == "medium" and any("Рост" in r for r in growth.reasons)


def test_urgency_normal_after_anomaly_breaks_trend():
    """Ремонт или ложное срабатывание: normal между отклонениями — повтора нет."""
    prev = [sess("S-1", 0, anomaly_type="F04", anomaly_score=0.6),
            sess("S-2", 7, status="normal", anomaly_score=0.1)]
    assert compute_urgency(diag(anomaly_type="F04", anomaly_score=0.6), prev, CONFIG).level == "low"


def test_urgency_skips_failed_previous_check():
    prev = [sess("S-1", 0, anomaly_type="F04", anomaly_score=0.6),
            sess("S-2", 7, status="check_failed", anomaly_type=None, anomaly_score=None,
                 confidence=None, signal_quality="bad")]
    assert compute_urgency(diag(anomaly_type="F04", anomaly_score=0.6), prev, CONFIG).level == "medium"


def test_urgency_not_lowered_by_uncertainty():
    sure = compute_urgency(diag(anomaly_type="F04", anomaly_score=0.85), [], CONFIG)
    unsure = compute_urgency(diag(anomaly_type="F04", anomaly_score=0.85, confidence=0.4,
                                  signal_quality="warning"), [], CONFIG)
    assert sure.level == unsure.level == "high"
    assert len(unsure.reasons) == len(sure.reasons) + 2


def test_urgency_only_for_anomaly():
    with pytest.raises(ValueError):
        compute_urgency(diag(status="normal", anomaly_score=0.3), [], CONFIG)


# --- Ссылки ------------------------------------------------------------------------------

def test_citation_page_and_document_from_metadata():
    reg = registry(HIT_VALVES)
    text, used, dropped = reg.render("Зазор проверить по руководству [yamz236_re-p050-1].")
    assert text == "Зазор проверить по руководству [236-3902150-Б РЭ, стр. 50]."
    assert [c.chunk_id for c in used] == ["yamz236_re-p050-1"] and not dropped


def test_unknown_or_other_turn_chunk_dropped():
    reg = registry(HIT_VALVES)
    text, used, dropped = reg.render("Момент 200 Н·м [yamz236_re-p999-1]. Зазор [yamz236_re-p050-1, fake-1].")
    assert "p999" not in text and "fake" not in text and "стр. 999" not in text
    assert dropped == ["yamz236_re-p999-1", "fake-1"]
    assert text.endswith("[236-3902150-Б РЭ, стр. 50].")


@pytest.mark.parametrize("forged", [
    "[536.3902150 РЭ, стр. 186]", "(236-3902150-Б РЭ, стр. 12)", "см. стр. 77", "на странице 31",
    # обходы, найденные ревью
    "[236.01.01 РК], 145-я страница", "на страницах 50 и 51 РЭ", "cтp. 50", "236-3902150-Б РЭ, лист 50",
    "(236.01.01 РК, 145)", "[РЭ 2016: 50]", "стр.50-52", "page 12", "pp. 4-5",
    "См. с. 50", "см. с.50 руководства по ремонту", "страница: 50", "С.50", "p50",
    "Руководство 236.01.01 стр. 9", "по карте дефектации 236.01.01",
    "см. стр. — 145", "страница – 31", "лист -12",
])
def test_forged_page_reference_removed(forged):
    text, used, dropped = registry(HIT_VALVES).render(f"Проверить зазор {forged}.")
    assert not any(ch.isdigit() for ch in text), text
    assert "РЭ" not in text and "РК" not in text and not used and dropped


@pytest.mark.parametrize("plain", [
    "Прогреть двигатель 5 с. и заглушить.", "Зазор 0,25–0,30 мм на холодном двигателе.",
    "Момент затяжки 180–200 Н·м.", "Стоимость 50 р. за час.", "Выждать 10 с. 2 раза подряд.",
])
def test_plain_numbers_not_mistaken_for_references(plain):
    assert registry(HIT_VALVES).render(plain)[0] == plain


def test_placeholder_chars_from_model_removed():
    text, used, _ = registry(HIT_VALVES).render("Зазор  [yamz236_re-p050-1].")
    assert text == "Зазор [236-3902150-Б РЭ, стр. 50]." and len(used) == 1


@pytest.mark.parametrize("code,status", [(500, None), (409, None), (504, "unknown"), (502, "unknown")])
def test_toir_http_error_release_or_block(settings, code, status):
    from app.integrations.errors import ServiceUnavailable

    class Failing(FakeToir):
        async def create_ticket(self, ticket):
            raise ServiceUnavailable("toir", service="toir", status_code=code)

    draft = make_draft(settings).draft
    with pytest.raises((ServiceUnavailable, TicketError)):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, Failing(), db_path=settings.db_path))
    row = store.get_ticket(settings.db_path, "B418HK98", "S-0009")
    assert (row or {}).get("status") == status


def test_real_reference_survives_forged_cleanup():
    text, used, _ = registry(HIT_VALVES).render("Зазор [yamz236_re-p050-1], а не (см. стр. 70).")
    assert text == "Зазор [236-3902150-Б РЭ, стр. 50], а не." and len(used) == 1


def test_foreign_engine_fragment_is_error():
    with pytest.raises(CitationError):
        registry(HIT_FOREIGN)


# --- Валидатор заявки ---------------------------------------------------------------------

def good_draft(**over) -> DraftInput:
    data = dict(conclusion="Повышенный зазор в клапанном механизме",
                operations=["Проверить и отрегулировать зазоры клапанов"],
                spare_parts=[SparePartInput(part_number="236-1007010-В2", name="Клапан впускной",
                                            chunk_id=HIT_VALVES.chunk_id)],
                chunk_ids=[HIT_VALVES.chunk_id])
    data.update(over)
    return DraftInput(**data)


def validate(draft: DraftInput, d: Diagnosis | None = None, applicability=OK, reg=None, sessions=None):
    return validate_draft(draft, diagnosis=d or diag(), applicability=applicability, vehicle_id="B418HK98",
                          vehicle_sessions=sessions or {"S-0009"}, engine_family=FAMILY,
                          registry=reg or registry(HIT_VALVES, HIT_ROCKER))


def test_valid_draft_passes():
    r = validate(good_draft())
    assert r.ok, r.violations
    assert [c.pdf_page for c in r.citations] == [50]


@pytest.mark.parametrize("kwargs,needle", [
    ({"d": diag(status="normal", anomaly_score=0.3)}, "только при отклонении"),
    ({"d": diag(status="check_failed", anomaly_type=None, anomaly_score=None, confidence=None,
                signal_quality="bad")}, "Качество записи"),
    ({"d": diag(signal_quality="bad")}, "Качество записи"),
    ({"applicability": Applicability(applicable=False, reason="нет EGR")}, "не применим"),
    ({"d": diag(session_id="S-0099")}, "не принадлежит"),
    ({"d": diag(vehicle_id="P456AT178")}, "другой машине"),
])
def test_validator_rejects_wrong_check(kwargs, needle):
    r = validate(good_draft(), **kwargs)
    assert any(needle in v for v in r.violations), r.violations


@pytest.mark.parametrize("over,needle", [
    ({"operations": []}, "работ"),
    ({"operations": ["  "]}, "работ"),
    ({"conclusion": " "}, "вывод"),
    ({"chunk_ids": [], "spare_parts": []}, "Нет ни одной ссылки"),
    ({"chunk_ids": ["yamz236_re-p051-1"], "spare_parts": []}, "не из выдачи"),
])
def test_validator_rejects_bad_content(over, needle):
    r = validate(good_draft(**over))
    assert any(needle in v for v in r.violations), r.violations


@pytest.mark.parametrize("number,chunk,ok", [
    ("236-1007010-В2", HIT_VALVES.chunk_id, True),
    ("236-1007010", HIT_VALVES.chunk_id, False),      # обрубок более длинного номера
    ("236-1007010-В", HIT_VALVES.chunk_id, False),    # другая буква исполнения
    ("236-1007111", HIT_ROCKER.chunk_id, True),       # в тексте длинное тире — то же самое
    ("236-1007111", HIT_VALVES.chunk_id, False),      # номер есть, но в другом фрагменте
    ("123", HIT_VALVES.chunk_id, False),              # слишком короткий
])
def test_part_number_must_be_in_cited_fragment(number, chunk, ok):
    draft = good_draft(spare_parts=[SparePartInput(part_number=number, name="деталь", chunk_id=chunk)],
                       chunk_ids=[HIT_VALVES.chunk_id, HIT_ROCKER.chunk_id])
    assert validate(draft).ok is ok


def test_part_source_must_be_among_ticket_references():
    draft = good_draft(spare_parts=[SparePartInput(part_number="236-1007111", chunk_id=HIT_ROCKER.chunk_id)])
    r = validate(draft)
    assert any("не среди ссылок" in v for v in r.violations)


@pytest.mark.parametrize("over", [
    {"conclusion": "Износ, см. 236.01.01 РК, стр. 145"},
    {"operations": ["Заменить клапан по 236-3902150-Б РЭ"]},
    {"operations": ["Отрегулировать зазоры [yamz236_re-p050-1]"]},
    {"operations": ["Заменить вкладыш 238-1005170-А"]},                 # номер не из фрагментов
    {"operations": ["Заменить болт 201457-П29"]},
    {"operations": ["Установить ремкомплект 236 1007010"]},
    {"operations": ["Деталь 1007010"]},
    {"spare_parts": [SparePartInput(part_number="236-1007010-В2", name="клапан, также 238-1005170-А",
                                    chunk_id=HIT_VALVES.chunk_id)]},
])
def test_free_text_cannot_smuggle_references_or_numbers(over):
    assert not validate(good_draft(**over)).ok


def test_numbers_from_cited_fragment_allowed_in_text():
    r = validate(good_draft(operations=["Заменить клапан впускной 236-1007010-В2"]))
    assert r.ok, r.violations


def test_extra_fields_from_model_rejected():
    with pytest.raises(ValidationError):
        DraftInput(conclusion="x", operations=["y"], chunk_ids=[], urgency="low")
    with pytest.raises(ValidationError):
        SparePartInput(part_number="236-1007010-В2", chunk_id="c", pdf_page=5)


def test_missing_anomaly_score_is_violation():
    d = diag().model_copy(update={"anomaly_score": None})
    assert any("anomaly_score" in v for v in validate(good_draft(), d).violations)


def test_part_number_spelling_taken_from_fragment():
    hit = Hit("yamz236_re-p051-1", "Втулка 236–1007032 направляющая.", pdf_page=51)
    draft = good_draft(spare_parts=[SparePartInput(part_number="236-1007032", name="Втулка", chunk_id=hit.chunk_id)],
                       chunk_ids=[hit.chunk_id])
    r = validate(draft, reg=registry(hit))
    assert r.ok and r.spare_parts == ["236-1007032 Втулка"]


def test_part_number_boundaries():
    assert part_number_in_text("236-1007010-В2", "клапан 236-1007010-В2.")
    assert not part_number_in_text("36-1007010-В2", "клапан 236-1007010-В2.")


# --- Черновик и подтверждение ---------------------------------------------------------------

class FakeToir:
    def __init__(self, fail: type[Exception] | None = None, delay: float = 0.0):
        self.sent, self.fail, self.delay = [], fail, delay

    async def create_ticket(self, ticket):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail("toir: сбой", service="toir")
        self.sent.append(ticket)
        return f"TKT-{len(self.sent):04d}"


def checked(**over) -> CheckedDiagnosis:
    d = diag(**over)
    history = (sess("S-0005", 0, anomaly_type="F04", anomaly_score=0.5),
               Session(**d.model_dump(), checked_at=datetime(2026, 11, 20, tzinfo=timezone.utc)))
    vehicle = Vehicle(vehicle_id="B418HK98", engine_model="ЯМЗ-236М2", engine_family=FAMILY, sessions=history)
    return CheckedDiagnosis(vehicle=vehicle, diagnosis=d, injection_detected=False, applicability=OK)


def make_draft(settings, **over):
    return create_draft(good_draft(**over), checked(), registry(HIT_VALVES, HIT_ROCKER),
                        db_path=settings.db_path, config_dir=CONFIG)


def event_types(settings) -> list[str]:
    with connect(settings.db_path) as conn:
        return [r[0] for r in conn.execute("SELECT event_type FROM events ORDER BY id")]


def test_draft_built_from_code_data(settings):
    out = make_draft(settings)
    assert out.draft and not out.violations
    t = out.draft.ticket
    assert t.anomaly_type == "F01" and t.urgency == "medium"         # тип и приоритет — от кода
    assert t.references[0].manual == "236-3902150-Б РЭ" and t.references[0].pdf_page == 50
    assert t.spare_parts == ["236-1007010-В2 Клапан впускной"]
    assert out.draft.urgency_reasons
    assert "ticket_draft" in event_types(settings)


def test_rejected_draft_logged_not_stored(settings):
    out = make_draft(settings, operations=[])
    assert out.draft is None and out.violations
    assert event_types(settings)[-1] == "ticket_rejected"
    with connect(settings.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ticket_drafts").fetchone()[0] == 0


def test_confirm_sends_exactly_shown_draft_once(settings):
    draft = make_draft(settings).draft
    toir = FakeToir()
    ticket_id = asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path))
    assert ticket_id == "TKT-0001" and toir.sent == [draft.ticket]
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path))
    # второй черновик по той же проверке не создаётся
    again = make_draft(settings)
    assert again.draft is None and any("уже создана" in v for v in again.violations)
    assert len(toir.sent) == 1
    types = event_types(settings)
    assert types.index("ticket_confirmed") < types.index("ticket_created")


def test_two_drafts_one_ticket_concurrently(settings):
    """Два черновика одной проверки подтверждены одновременно — заявка одна."""
    d1, d2 = make_draft(settings).draft, make_draft(settings).draft
    toir = FakeToir(delay=0.05)

    async def both():
        return await asyncio.gather(
            confirm_draft(d1.draft_id, d1.content_hash, toir, db_path=settings.db_path),
            confirm_draft(d2.draft_id, d2.content_hash, toir, db_path=settings.db_path),
            return_exceptions=True)

    results = asyncio.run(both())
    assert sum(isinstance(r, str) for r in results) == 1
    assert sum(isinstance(r, TicketError) for r in results) == 1
    assert len(toir.sent) == 1


def test_confirm_with_wrong_hash_rejected(settings):
    draft = make_draft(settings).draft
    toir = FakeToir()
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft(draft.draft_id, "0" * 64, toir, db_path=settings.db_path))
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft("нет-такого", draft.content_hash, toir, db_path=settings.db_path))
    assert not toir.sent


def test_tampered_stored_draft_rejected(settings):
    """Содержимое черновика изменили в БД после показа (хэш прежний) — не отправляем."""
    draft = make_draft(settings).draft
    payload = draft.ticket.model_dump(mode="json") | {"urgency": "low"}
    with connect(settings.db_path) as conn:
        conn.execute("UPDATE ticket_drafts SET payload = ? WHERE draft_id = ?",
                     (json.dumps(payload, ensure_ascii=False), draft.draft_id))
    toir = FakeToir()
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path))
    assert not toir.sent


def test_cancelled_draft_not_sent(settings):
    draft = make_draft(settings).draft
    assert cancel_draft(draft.draft_id, db_path=settings.db_path)
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, FakeToir(), db_path=settings.db_path))


def test_toir_failure_allows_retry(settings):
    """Соединение не установлено — заявка точно не создана, повтор разрешён."""
    draft = make_draft(settings).draft
    with pytest.raises(NotDelivered):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, FakeToir(fail=NotDelivered),
                                  db_path=settings.db_path))
    assert store.get_ticket(settings.db_path, "B418HK98", "S-0009") is None
    assert "toir_error" in event_types(settings)
    ticket_id = asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, FakeToir(), db_path=settings.db_path))
    assert ticket_id == "TKT-0001"
    assert store.get_ticket(settings.db_path, "B418HK98", "S-0009")["ticket_id"] == "TKT-0001"


@pytest.mark.parametrize("failure", [DeliveryUnknown, asyncio.CancelledError])
def test_unknown_toir_outcome_blocks_retry(settings, failure):
    """Ответа нет (таймаут чтения, отмена запроса) — заявка могла создаться: повтор запрещён."""
    draft = make_draft(settings).draft
    toir = FakeToir(fail=(lambda *a, **k: failure()) if failure is asyncio.CancelledError else failure)
    with pytest.raises((TicketError, asyncio.CancelledError)):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path))
    assert store.get_ticket(settings.db_path, "B418HK98", "S-0009")["status"] == "unknown"
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, FakeToir(), db_path=settings.db_path))
    assert make_draft(settings).draft is None


def test_mock_toir_accepts_draft(settings, mock_transport):
    """Черновик в формате, который принимает заглушка ТОиР организаторов (201, TKT-…)."""
    from app.integrations.toir_client import HttpToirClient
    draft = make_draft(settings).draft
    client = HttpToirClient(settings.mock_toir_url, transport=mock_transport)
    ticket_id = asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, client, db_path=settings.db_path))
    assert ticket_id.startswith("TKT-")


# --- Граница модуля ----------------------------------------------------------------------

def test_policy_does_not_import_llm_client():
    code = ("import sys, importlib, pkgutil, app.policy as p\n"
            "[importlib.import_module(f'app.policy.{m.name}') for m in pkgutil.iter_modules(p.__path__)]\n"
            "bad = [m for m in sys.modules if m == 'openai' or m.startswith('openai.') or m == 'app.integrations.llm']\n"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_cancel_during_sending_is_refused(settings):
    """«Отмена», нажатая пока черновик уходит в ТОиР, не проходит: статус не станет «отменён»
    при созданной заявке."""
    draft = make_draft(settings).draft
    toir = FakeToir(delay=0.05)

    async def confirm_and_cancel():
        task = asyncio.create_task(confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path))
        await asyncio.sleep(0.01)                       # отправка началась, ответа ТОиР ещё нет
        cancelled = cancel_draft(draft.draft_id, db_path=settings.db_path)
        return cancelled, await task

    cancelled, ticket_id = asyncio.run(confirm_and_cancel())
    assert not cancelled
    assert ticket_id == "TKT-0001"
    assert store.get_draft(settings.db_path, draft.draft_id)["status"] == "sent"


def test_failed_send_returns_draft_and_allows_cancel(settings):
    """ТОиР точно не создал заявку — черновик снова обычный: его можно отменить."""
    draft = make_draft(settings).draft
    with pytest.raises(NotDelivered):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, FakeToir(fail=NotDelivered),
                                  db_path=settings.db_path))
    assert store.get_draft(settings.db_path, draft.draft_id)["status"] == "draft"
    assert cancel_draft(draft.draft_id, db_path=settings.db_path)


def test_unknown_outcome_draft_cannot_be_cancelled(settings):
    """Исход отправки неизвестен — черновик не отменяется (заявка могла быть создана)."""
    draft = make_draft(settings).draft
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, FakeToir(fail=DeliveryUnknown),
                                  db_path=settings.db_path))
    assert store.get_draft(settings.db_path, draft.draft_id)["status"] == "unknown"
    assert not cancel_draft(draft.draft_id, db_path=settings.db_path)


def test_same_draft_confirmed_twice_concurrently(settings):
    """Двойное нажатие / две вкладки: один черновик подтверждён дважды одновременно — заявка одна."""
    draft = make_draft(settings).draft
    toir = FakeToir(delay=0.05)

    async def twice():
        return await asyncio.gather(
            confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path),
            confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path),
            return_exceptions=True)

    results = asyncio.run(twice())
    assert sum(isinstance(r, str) for r in results) == 1
    assert sum(isinstance(r, TicketError) for r in results) == 1
    assert len(toir.sent) == 1


def test_losing_draft_returns_to_draft_status(settings):
    """По проверке заявка уже создана другим черновиком — проигравший не застревает в sending."""
    d1, d2 = make_draft(settings).draft, make_draft(settings).draft
    asyncio.run(confirm_draft(d1.draft_id, d1.content_hash, FakeToir(), db_path=settings.db_path))
    with pytest.raises(TicketError):
        asyncio.run(confirm_draft(d2.draft_id, d2.content_hash, FakeToir(), db_path=settings.db_path))
    assert store.get_draft(settings.db_path, d2.draft_id)["status"] == "draft"


def test_storage_failure_before_send_does_not_leave_sending(settings, monkeypatch):
    """SQLite упал при резерве (например, database is locked) — черновик снова draft, ТОиР не вызывался."""
    import sqlite3
    draft = make_draft(settings).draft
    toir = FakeToir()

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "reserve_ticket", locked)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(confirm_draft(draft.draft_id, draft.content_hash, toir, db_path=settings.db_path))
    assert not toir.sent
    assert store.get_draft(settings.db_path, draft.draft_id)["status"] == "draft"
