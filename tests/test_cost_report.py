"""Расчёт стоимости сессии: средневзвешенная цена входа с кешем, вызовы на сессию, исключение
вызовов вне действия механика и индексации. Цифры выдуманы."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("cost_report", Path(__file__).resolve().parent.parent / "scripts" / "cost_report.py")
cost_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost_report)

CFG = {"currency": "$", "usd_rub": 100.0, "session": {"explain": 1, "question": 2, "ticket": 1},
       "prices": {"m": {"input": 2.0, "cached_input": 0.1, "output": 10.0}}}


def call(rid, scenario, step, tin, cached, tout):
    return {"request_id": rid, "scenario": scenario, "step": step, "model": "m",
            "input_tokens": tin, "cached_input_tokens": cached, "output_tokens": tout}


def test_effective_input_price_with_cache():
    assert cost_report.effective_input_price(CFG["prices"]["m"], 1000, 0) == 2.0
    assert abs(cost_report.effective_input_price(CFG["prices"]["m"], 1000, 500) - 1.05) < 1e-9


def test_session_rows_scale_by_session_and_skip_noise():
    calls = [
        call("q1", "question", "agent_final", 1_000_000, 0, 0),
        call("q2", "question", "agent_final", 1_000_000, 0, 0),
        call(None, "question", "agent_final", 9_000_000, 0, 0),        # прогон eval — не в счёт
        call(None, "indexing", "scan_ocr", 9_000_000, 0, 0),           # индексация — не в счёт
    ]
    rows = cost_report.session_rows(calls, CFG)
    assert len(rows) == 1
    r = rows[0]
    assert r["calls"] == 2.0                      # 1 вызов на вопрос × 2 вопроса в сессии
    assert abs(r["rub"] - 2 * 2.0 * 100) < 1e-9   # 2 вызова × $2 × 100 ₽


def test_fleet_month_cost():
    cfg = dict(CFG, fleet={"vehicles": 10, "check_interval_days": 10, "days_in_month": 30,
                           "status_share": {"normal": 0.5, "anomaly": 0.5},
                           "actions_by_status": {"normal": {"explain": 1}, "anomaly": {"explain": 1, "question": 2}},
                           "compare_models": ["gpt-6.2-sol"]})
    calls = [call("e1", "explain", "agent_final", 1_000_000, 0, 0),   # 200 ₽ за объяснение
             call("q1", "question", "agent_final", 500_000, 0, 0)]    # 100 ₽ за вопрос
    table = cost_report.fleet_table(calls, cfg, idx_total=0.0)
    month = next(r for r in table if r[0] == "Модели на парк в месяц")
    # 30 проверок: половина по 200 ₽, половина по 200 + 2×100 = 400 ₽ → 30 × 300 = 9000 ₽
    assert month[2] == "9 000,00"
