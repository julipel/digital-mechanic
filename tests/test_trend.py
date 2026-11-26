"""Оценка тенденции (B2): детерминированная функция, без модели. Рост, возврат в норму после
отклонения (ремонт или ложное срабатывание), меньше 3 точек, сбой записи в ряду, код,
противоречащий двигателю, нестабильный ряд."""
import pytest

from app.config import PROJECT_ROOT
from app.policy.trend import FALLING, RISING, STABLE, compute_trend, vehicle_trend
from tests.test_policy import sess

CONFIG = PROJECT_ROOT / "config"


def anomaly(sid, days, score, code="F04"):
    return sess(sid, days, anomaly_type=code, anomaly_score=score)


def normal(sid, days, score=0.1):
    return sess(sid, days, status="normal", anomaly_score=score)


def failed(sid, days):
    return sess(sid, days, status="check_failed", anomaly_type=None, anomaly_score=None,
                confidence=None, signal_quality="bad")


def test_growth_gives_days_to_threshold_with_caveats():
    t = compute_trend([normal("S-1", 0, 0.3), anomaly("S-2", 14, 0.45), anomaly("S-3", 28, 0.6)], CONFIG)
    assert t.built and t.direction == RISING and t.n_points == 3
    # 0,15 за 14 дней; от 0,60 до порога 0,80 — ещё ~19 дней
    assert t.days_to_threshold == 19 and not t.threshold_reached
    assert t.label == "Оценка тенденции" and any("не дата поломки" in c for c in t.caveats)
    assert any("Точек в расчёте: 3" in c for c in t.caveats)


def test_threshold_already_reached():
    t = compute_trend([normal("S-1", 0, 0.33), anomaly("S-2", 15, 0.67), anomaly("S-3", 33, 0.9)], CONFIG)
    assert t.built and t.threshold_reached and t.days_to_threshold is None


def test_slow_growth_beyond_horizon_has_no_date():
    t = compute_trend([anomaly("S-1", 0, 0.50), anomaly("S-2", 30, 0.56), anomaly("S-3", 60, 0.62)], CONFIG)
    assert t.built and t.direction == RISING and t.days_to_threshold is None
    assert f"{t.horizon_days} дн." in t.summary


def test_normal_after_anomaly_restarts_series():
    """Ловушка данных: после отклонения пришла норма — рост дальше не тянем."""
    checks = [normal("S-1", 0, 0.2), anomaly("S-2", 14, 0.6), anomaly("S-3", 28, 0.85), normal("S-4", 42, 0.1)]
    t = compute_trend(checks, CONFIG)
    assert not t.built and "Мало проверок" in t.reason and t.n_points == 1
    assert [p.used for p in t.points] == [False, False, False, True]
    # после ремонта набралось три нормальные проверки — тенденция есть, роста нет
    t = compute_trend(checks + [normal("S-5", 56, 0.12), normal("S-6", 70, 0.11)], CONFIG)
    assert t.built and t.direction == STABLE and t.days_to_threshold is None


def test_falling_series():
    t = compute_trend([anomaly("S-1", 0, 0.7), anomaly("S-2", 14, 0.6), anomaly("S-3", 28, 0.5)], CONFIG)
    assert t.built and t.direction == FALLING and t.days_to_threshold is None


def test_less_than_three_points():
    t = compute_trend([normal("S-1", 0), anomaly("S-2", 14, 0.7)], CONFIG)
    assert not t.built and "Мало проверок" in t.reason


def test_selected_check_failed_no_trend():
    t = compute_trend([normal("S-1", 0), anomaly("S-2", 14, 0.6), anomaly("S-3", 28, 0.7), failed("S-4", 42)],
                      CONFIG)
    assert not t.built and "сбой записи" in t.reason
    assert t.points[-1].anomaly_score is None          # на графике — отметка без значения


def test_failed_check_inside_series_breaks_it():
    checks = [normal("S-1", 0, 0.3), anomaly("S-2", 14, 0.6), failed("S-3", 28), anomaly("S-4", 42, 0.75)]
    t = compute_trend(checks, CONFIG)
    assert not t.built and t.n_points == 1 and "сбой записи" in t.reason
    # после сбоя набралось три точки — оценка по ним, а не через сбой
    t = compute_trend(checks + [anomaly("S-5", 56, 0.76), anomaly("S-6", 70, 0.78)], CONFIG)
    assert t.built and t.n_points == 3 and [p.session_id for p in t.points if p.used] == ["S-4", "S-5", "S-6"]


def test_inconsistent_code_is_not_trusted():
    checks = [normal("S-1", 0), normal("S-2", 14), anomaly("S-3", 28, 0.7, code="F12")]
    t = compute_trend(checks, CONFIG, inconsistent_ids={"S-3"})
    assert not t.built and "противоречит" in t.reason


def test_unstable_series():
    t = compute_trend([anomaly("S-1", 0, 0.5), anomaly("S-2", 14, 0.95), anomaly("S-3", 28, 0.55)], CONFIG)
    assert not t.built and "нестабилен" in t.reason


def test_window_takes_last_checks_only():
    t = compute_trend([normal(f"S-{i}", i * 14, 0.1) for i in range(8)], CONFIG)
    assert len(t.points) == 5 and t.points[0].session_id == "S-3"


@pytest.mark.parametrize("vehicle_id,session_id", [
    ("T801PO178", "S-0046"), ("P854BT78", "S-0021"), ("K354KC98", "S-0087"), ("M260EH47", "S-0079"),
    ("Y652CK98", "S-0132"), ("C808CM178", "S-0144"), ("C933KX98", "S-0163")])
def test_fleet_traps_no_trend_on_inconsistent_code(fleet, settings, vehicle_id, session_id):
    t = vehicle_trend(fleet.get(vehicle_id), session_id, settings.config_dir)
    assert not t.built and "противоречит" in t.reason


def test_fleet_trend_never_builds_on_failed_check(fleet, settings):
    for v in fleet.list():
        for s in v.sessions:
            t = vehicle_trend(v, s.session_id, settings.config_dir)
            if s.status == "check_failed" or s.signal_quality == "bad":
                assert not t.built, (v.vehicle_id, s.session_id)
            if t.built:
                assert t.n_points >= 3 and t.label and t.caveats


def test_as_dict_returns_copies_of_points():
    """get_trend убирает даты из точек для модели — это не должно портить точки для графика."""
    t = compute_trend([normal("S-1", 0, 0.3), anomaly("S-2", 14, 0.45), anomaly("S-3", 28, 0.6)], CONFIG)
    view = t.as_dict()
    for p in view["points"]:
        p.pop("checked_at")
    assert all("checked_at" in p for p in t.as_dict()["points"])
