"""«Главный механик» (B3): статус машины — детерминированная функция по последней проверке,
приоритету (правило 9) и оценке тенденции. Сбой записи и код, невозможный для двигателя,
выводов о двигателе не дают (правила 4, 6); норма после отклонения статус не поднимает."""
from datetime import datetime, timezone

import pytest

from app.config import PROJECT_ROOT
from app.integrations.fleet import Vehicle
from app.policy.fleet_status import ATTENTION, NORMAL, QUARANTINE, status_history, status_since, vehicle_status
from tests.test_api import client_for  # noqa: F401 (фикстура)
from tests.test_trend import anomaly, failed, normal

CONFIG = PROJECT_ROOT / "config"
TITLES = {"F02": "Низкое давление масла", "F04": "Стук", "F12": "Зажигание"}


def car(*sessions, family="yamz-236") -> Vehicle:
    return Vehicle(vehicle_id="A001AA00", engine_model="ЯМЗ-236М2", engine_family=family,
                   sessions=tuple(s.model_copy(update={"vehicle_id": "A001AA00"}) for s in sessions))


def test_no_checks_needs_attention():
    st = vehicle_status(car(), CONFIG)
    assert st.status == ATTENTION and "Проверок нет" in st.reasons[0]


def test_normal():
    st = vehicle_status(car(normal("S-1", 0), normal("S-2", 14)), CONFIG)
    assert st.status == NORMAL and st.urgency is None


def test_high_urgency_is_quarantine_with_urgency_reasons():
    st = vehicle_status(car(normal("S-1", 0), anomaly("S-2", 14, 0.9)), CONFIG, fault_titles=TITLES)
    assert st.status == QUARANTINE and st.urgency == "high"
    assert any("приоритет «высокий»" in r for r in st.reasons)
    assert any("Выраженность отклонения 0.90" in r for r in st.reasons)      # причины из compute_urgency
    assert "Заявки по этому отклонению нет" in st.notes


def test_fault_minimum_high_is_quarantine_even_with_low_score():
    st = vehicle_status(car(anomaly("S-1", 0, 0.55, code="F02")), CONFIG)
    assert st.status == QUARANTINE


def test_medium_urgency_is_attention():
    st = vehicle_status(car(normal("S-1", 0), anomaly("S-2", 14, 0.7, code="F06")), CONFIG)
    assert st.status == ATTENTION and st.urgency == "medium"


def test_failed_last_check_gives_no_engine_conclusion():
    st = vehicle_status(car(normal("S-1", 0), failed("S-2", 14)), CONFIG)
    assert st.status == ATTENTION and "повторите запись" in st.reasons[0]
    assert st.basis_session_id == "S-1"


def test_failed_check_does_not_lift_quarantine():
    st = vehicle_status(car(anomaly("S-1", 0, 0.9), failed("S-2", 14)), CONFIG)
    assert st.status == QUARANTINE and "сбой записи" in st.reasons[0]


def test_inconsistent_code_is_not_quarantine():
    # F12 (зажигание) у дизеля ЯМЗ-536 невозможен: значение 0,95 недостоверно
    st = vehicle_status(car(normal("S-1", 0), anomaly("S-2", 14, 0.95, code="F12"), family="yamz-536"), CONFIG)
    assert st.status == ATTENTION and st.urgency is None
    assert "не подходит двигателю" in st.reasons[0]


def test_normal_after_anomaly_stays_normal_with_note():
    st = vehicle_status(car(anomaly("S-1", 0, 0.9), normal("S-2", 14)), CONFIG, fault_titles=TITLES,
                        ticket_ids={("A001AA00", "S-1"): "TKT-1"})
    assert st.status == NORMAL
    assert any("затем норма" in n and "TKT-1" in n for n in st.notes)


def test_rising_trend_raises_normal_to_attention():
    st = vehicle_status(car(normal("S-1", 0, 0.3), normal("S-2", 7, 0.4), normal("S-3", 14, 0.5)), CONFIG)
    assert st.status == ATTENTION and st.days_to_threshold is not None
    assert any("Оценка тенденции" in r for r in st.reasons)


def test_ticket_shown_for_basis_anomaly():
    st = vehicle_status(car(anomaly("S-1", 0, 0.9)), CONFIG, ticket_ids={("A001AA00", "S-1"): "TKT-7"})
    assert st.ticket_id == "TKT-7" and "Заявка TKT-7 создана" in st.notes


def test_history_and_since():
    v = car(normal("S-1", 0), anomaly("S-2", 14, 0.9), anomaly("S-3", 28, 0.92), normal("S-4", 42))
    h = status_history(v, CONFIG)
    assert [x["status"] for x in h] == [NORMAL, QUARANTINE, QUARANTINE, NORMAL]
    assert status_since(h[:3]) == h[1]["checked_at"]


def test_unknown_session_raises():
    with pytest.raises(KeyError):
        vehicle_status(car(normal("S-1", 0)), CONFIG, session_id="S-9")


@pytest.mark.parametrize("vehicle_id,session_id", [
    ("T801PO178", "S-0046"), ("P854BT78", "S-0021"), ("K354KC98", "S-0087"), ("M260EH47", "S-0079"),
    ("Y652CK98", "S-0132"), ("C808CM178", "S-0144"), ("C933KX98", "S-0163")])
def test_fleet_traps_inconsistent_code_never_quarantine(fleet, settings, vehicle_id, session_id):
    st = vehicle_status(fleet.get(vehicle_id), settings.config_dir, session_id=session_id)
    assert st.status == ATTENTION and st.urgency is None and "не подходит двигателю" in st.reasons[0]


def test_fleet_failed_check_never_judges_engine(fleet, settings):
    for v in fleet.list():
        for s in v.sessions:
            st = vehicle_status(v, settings.config_dir, session_id=s.session_id)
            if s.status == "check_failed" or s.signal_quality == "bad":
                assert st.status != NORMAL and "сбой записи" in st.reasons[0], (v.vehicle_id, s.session_id)
                assert st.basis_session_id != s.session_id


def test_api_fleet_status_sorted(client_for):
    client, model, _ = client_for()
    data = client.get("/api/fleet/status").json()
    order = {"quarantine": 0, "attention": 1, "normal": 2}
    ranks = [order[v["status"]] for v in data["vehicles"]]
    assert ranks == sorted(ranks) and sum(data["counts"].values()) == len(data["vehicles"])
    for v in data["vehicles"]:
        assert v["reasons"] and v["history"] and "description" not in v
    k354 = next(v for v in data["vehicles"] if v["vehicle_id"] == "K354KC98")
    assert k354["status"] == "normal"                                    # норма после F12 у дизеля
    assert model.calls == []                                             # модель не участвует


def test_inconsistent_check_does_not_break_repeat():
    """Правило 6 в приоритете: F12 у дизеля недостоверен и не обрывает повтор F04 (заявка и
    статус берут прошлые проверки одинаково — через previous_checks)."""
    from app.policy.urgency import compute_urgency, previous_checks
    v = car(anomaly("S-1", 0, 0.66), anomaly("S-2", 14, 0.9, code="F12"), anomaly("S-3", 28, 0.68),
            family="yamz-536")
    prev = previous_checks(v, "S-3", CONFIG)
    assert [s.session_id for s in prev] == ["S-1"]
    u = compute_urgency(v.session("S-3"), prev, CONFIG)
    assert u.level == "high" and any("То же отклонение F04" in r for r in u.reasons)
    assert vehicle_status(v, CONFIG).urgency == "high"
    assert previous_checks(v, "S-9", CONFIG) == []


def test_history_repeat_agrees_with_urgency_over_inconsistent_check():
    """Память по ТС и приоритет считают повтор одинаково: F12 у дизеля повтор F04 не обрывает."""
    from app.policy.history import vehicle_history
    v = car(anomaly("S-1", 0, 0.66), anomaly("S-2", 14, 0.9, code="F12"), anomaly("S-3", 28, 0.68),
            family="yamz-536")
    h = vehicle_history(v, "S-3", [], CONFIG)
    assert h.repeat_count == 2
    assert any("2-я проверка подряд" in o and "недостоверный результат ML" in o for o in h.observations)
