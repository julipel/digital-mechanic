"""Стоимость сессии проверки двигателя по журналу llm_calls — в формате калькулятора организаторов.

Запрос в калькуляторе = одна сессия проверки: состав (сколько объяснений, вопросов, заявок)
и цены моделей — в config/cost.json. По журналу для каждого (сценарий, шаг, модель) считается
среднее число вызовов на одно действие механика (request_id) и средние токены; вызовов на сессию =
вызовов на действие × число таких действий в сессии.

Кешированный вход дешевле: цена входа в строке — средневзвешенная по доле кешированных токенов
(input_tokens включает кешированные). Индексация и распознавание сканов (scenario=indexing) —
разовые затраты, в запрос не входят, печатаются отдельно.

    python scripts/cost_report.py                       # журнал из DB_PATH (.env)
    python scripts/cost_report.py --json calls.json     # выгрузка llm_calls со стенда (список строк)
    python scripts/cost_report.py --out cost.csv        # заполненный шаблон калькулятора
"""
import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCENARIOS = {"explain": "Объяснение проверки", "question": "Вопрос механика", "ticket": "Черновик заявки"}
STEPS = {"agent_tool_round": "раунд с инструментами", "agent_final": "итоговый ответ механику",
         "query_embedding": "эмбеддинг запроса к поиску"}
COLUMNS = ["ts", "request_id", "scenario", "step", "model", "input_tokens", "cached_input_tokens", "output_tokens"]


def load_calls(db: Path | None, json_file: Path | None) -> list[dict]:
    if json_file:
        return json.loads(json_file.read_text(encoding="utf-8"))
    if db is None:
        from app.config import get_settings
        db = get_settings().db_path
    con = sqlite3.connect(db)
    return [dict(zip(COLUMNS, r)) for r in con.execute(f"SELECT {', '.join(COLUMNS)} FROM llm_calls")]


def effective_input_price(price: dict, tokens: float, cached: float) -> float:
    """Цена входа за 1 млн с учётом доли кешированных токенов."""
    if tokens <= 0:
        return price["input"]
    return (price["input"] * (tokens - cached) + price["cached_input"] * cached) / tokens


def session_rows(calls: list[dict], cfg: dict) -> list[dict]:
    """Строки калькулятора: шаг агента на сессию проверки. Вызовы вне действия механика
    (без request_id — тесты и прогоны золотого набора) и индексация не учитываются."""
    work = [c for c in calls if c["scenario"] in SCENARIOS and c["request_id"]]
    actions = defaultdict(set)
    groups = defaultdict(list)
    for c in work:
        actions[c["scenario"]].add(c["request_id"])
        groups[(c["scenario"], c["step"], c["model"])].append(c)
    rows = []
    for scenario in SCENARIOS:
        for (sc, step, model), items in sorted(groups.items()):
            if sc != scenario or not actions[sc]:
                continue
            per_action = len(items) / len(actions[sc])
            n = cfg["session"].get(sc, 0)
            if n == 0:
                continue
            avg_in = sum(c["input_tokens"] for c in items) / len(items)
            avg_cached = sum(c["cached_input_tokens"] for c in items) / len(items)
            avg_out = sum(c["output_tokens"] for c in items) / len(items)
            price = cfg["prices"][model]
            price_in = effective_input_price(price, avg_in, avg_cached)
            calls_per_session = per_action * n
            rub = calls_per_session * (avg_in * price_in + avg_out * price["output"]) / 1e6 * cfg["usd_rub"]
            rows.append({
                "step": f"{SCENARIOS[sc]} ×{n}: {STEPS.get(step, step)}", "model": model,
                "calls": calls_per_session, "tokens_in": avg_in, "tokens_out": avg_out,
                "price_in": price_in, "price_out": price["output"], "rub": rub,
                "cached_share": avg_cached / avg_in if avg_in else 0.0,
                "actions": len(actions[sc]),
            })
    return rows


def indexing_cost(calls: list[dict], cfg: dict) -> list[tuple[str, str, int, int, int, float]]:
    out = []
    groups = defaultdict(list)
    for c in calls:
        if c["scenario"] == "indexing":
            groups[(c["step"], c["model"])].append(c)
    for (step, model), items in sorted(groups.items()):
        price = cfg["prices"].get(model)
        if price is None:
            continue
        tin = sum(c["input_tokens"] for c in items)
        tcached = sum(c["cached_input_tokens"] for c in items)
        tout = sum(c["output_tokens"] for c in items)
        usd = ((tin - tcached) * price["input"] + tcached * price["cached_input"] + tout * price["output"]) / 1e6
        out.append((step, model, len(items), tin, tout, usd * cfg["usd_rub"]))
    return out


def action_costs(calls: list[dict], cfg: dict, model: str | None = None) -> dict[str, float]:
    """₽ на одно действие механика каждого сценария (все шаги агента и эмбеддинги).
    model — пересчитать шаги агента по ценам другой модели на тех же токенах."""
    if model:
        calls = [dict(c, model=model) if c["step"].startswith("agent_") else c for c in calls]
    one = dict(cfg, session={sc: 1 for sc in SCENARIOS})
    out = {sc: 0.0 for sc in SCENARIOS}
    for r in session_rows(calls, one):
        sc = next(k for k, v in SCENARIOS.items() if r["step"].startswith(v))
        out[sc] += r["rub"]
    return out


def fleet_table(calls: list[dict], cfg: dict, idx_total: float) -> list[list[str]]:
    """Лист «Парк»: стоимость моделей в месяц и год на парк при допущениях из cfg["fleet"]."""
    f = cfg["fleet"]
    checks_month = f["vehicles"] * f["days_in_month"] / f["check_interval_days"]
    models = f["compare_models"]
    # первая модель — та, на которой собран журнал (AGENT_MODEL стенда); остальные — пересчёт по ценам
    costs = {m: action_costs(calls, cfg, None if m == models[0] else m) for m in models}
    blank = [""] * (2 + len(models))
    rows = [["Цифровой механик: стоимость моделей на парк"] + blank[1:],
            [f"Курс доллара, ₽ за 1 $ → {num(cfg['usd_rub'], 4)}"] + blank[1:], blank,
            ["Допущение", "Значение"] + [""] * len(models),
            ["Машин в парке", str(f["vehicles"])] + [""] * len(models),
            ["Проверка каждой машины раз в, дней", str(f["check_interval_days"])] + [""] * len(models),
            ["Проверок в месяц на парк", num(checks_month, 1)] + [""] * len(models)]
    names = {"normal": "норма", "anomaly": "отклонение", "check_failed": "сбой записи"}
    for st, share in f["status_share"].items():
        acts = ", ".join(f"{SCENARIOS[k].lower()} ×{v}" for k, v in f["actions_by_status"][st].items())
        rows.append([f"Доля проверок «{names[st]}» (механик: {acts})", f"{share:.0%}"] + [""] * len(models))
    rows += [blank, ["Стоимость, ₽", ""] + models]
    for sc in SCENARIOS:
        rows.append([f"Одно действие: {SCENARIOS[sc].lower()}", ""] + [num(costs[m][sc], 4) for m in models])
    per_check = {}
    for st in f["status_share"]:
        line = [f"Одна проверка «{names[st]}»", ""]
        for m in models:
            v = sum(costs[m][sc] * n for sc, n in f["actions_by_status"][st].items())
            per_check.setdefault(m, {})[st] = v
            line.append(num(v, 4))
        rows.append(line)
    avg = {m: sum(per_check[m][st] * sh for st, sh in f["status_share"].items()) for m in models}
    month = {m: avg[m] * checks_month for m in models}
    rows += [["Средняя проверка (с учётом долей)", ""] + [num(avg[m], 4) for m in models],
             ["Модели на парк в месяц", ""] + [num(month[m], 2) for m in models],
             ["Модели на парк в год", ""] + [num(month[m] * 12, 2) for m in models],
             ["На одну машину в месяц", ""] + [num(month[m] / f["vehicles"], 2) for m in models],
             blank,
             ["Разово: индексация руководств и распознавание сканов (gpt-6.2-astra, эмбеддинги)",
              num(idx_total, 2)] + [""] * len(models),
             blank,
             ["Комментарий: интервал проверок и доли статусов — по данным заглушки (50 машин, 195 проверок), "
              "у реального парка доля отклонений, вероятно, ниже — тогда дешевле. Режим «Главный механик» "
              f"считается кодом без модели. {', '.join(models[1:]) or '—'}: пересчёт по ценам на токенах {models[0]} "
              "(до замера качества). Сервер и хранение не включены."] + blank[1:]]
    return rows


def num(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def write_template(path: Path, rows: list[dict], cfg: dict, comment: list[str]) -> None:
    """Заполненный лист «Расчёт» калькулятора организаторов (CSV, как выгружен шаблон)."""
    total = sum(r["rub"] for r in rows)
    lines = [
        ["ИИкузница 2026 — калькулятор экономики решения"] + [""] * 8,
        ["Цифровой механик: стоимость одной сессии проверки двигателя"] + [""] * 8,
        [f"Курс доллара, ₽ за 1 $ → {num(cfg['usd_rub'], 4)}"] + [""] * 8,
        [""] * 9,
        ["Модели на один запрос"] + [""] * 8,
        ["Запрос = одна сессия проверки: " + ", ".join(f"{SCENARIOS[k].lower()} ×{v}" for k, v in cfg["session"].items())]
        + [""] * 8,
        ["Шаг агента", "Модель", "Вызовов на запрос", "Токенов на входе", "Токенов на выходе",
         "Цена входа за 1 млн", "Цена выхода за 1 млн", "Валюта", "₽ на 1 запрос"],
    ]
    for r in rows:
        lines.append([r["step"], r["model"], num(r["calls"]), num(r["tokens_in"], 0), num(r["tokens_out"], 0),
                      num(r["price_in"], 4), num(r["price_out"], 2), cfg["currency"], num(r["rub"], 4)])
    lines.append(["Итого модели на 1 запрос"] + [""] * 7 + [num(total, 4)])
    lines.append([""] * 9)
    lines.append(["Комментарий: " + " ".join(comment)] + [""] * 8)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path)
    parser.add_argument("--json", type=Path, help="выгрузка llm_calls (список словарей)")
    parser.add_argument("--indexing-db", type=Path, help="журнал с вызовами индексации (по умолчанию тот же)")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "cost.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fleet-out", type=Path, help="лист «Парк»: стоимость на парк в месяц/год (CSV)")
    parser.add_argument("--as-model", help="оценка: цены этой модели вместо модели агента на тех же токенах "
                                           "(эмбеддинги не меняются); настоящие токены другой модели будут иными")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    calls = load_calls(args.db, args.json)
    if args.as_model:
        if args.as_model not in cfg["prices"]:
            print(f"нет цены для {args.as_model} в {args.config}")
            return 2
        calls = [dict(c, model=args.as_model) if c["step"].startswith("agent_") else c for c in calls]
    rows = session_rows(calls, cfg)
    if not rows:
        print("В журнале нет действий механика (request_id) — считать не по чему")
        return 2
    total = sum(r["rub"] for r in rows)
    print(f"{'Шаг агента':55} {'модель':24} {'выз.':>5} {'вход':>7} {'кеш%':>5} {'выход':>6} {'$/1М вх':>8} {'₽':>8}")
    for r in rows:
        print(f"{r['step']:55} {r['model']:24} {r['calls']:5.2f} {r['tokens_in']:7.0f} {r['cached_share']*100:5.0f}"
              f" {r['tokens_out']:6.0f} {r['price_in']:8.4f} {r['rub']:8.4f}")
    print(f"Итого на сессию проверки: {num(total, 4)} ₽ (≈ ${total / cfg['usd_rub']:.4f})")
    print("действий в журнале:", {sc: next((r['actions'] for r in rows if r['step'].startswith(SCENARIOS[sc])), 0)
                                  for sc in SCENARIOS})
    idx_calls = load_calls(args.indexing_db, None) if args.indexing_db else calls
    idx = indexing_cost(idx_calls, cfg)
    idx_total = sum(x[5] for x in idx)
    if idx:
        print("\nРазово (индексация и распознавание, в запрос не входит):")
        for step, model, n, tin, tout, rub in idx:
            print(f"  {step:18} {model:24} вызовов {n:4}  вход {tin:>10,}  выход {tout:>8,}  {num(rub)} ₽")
        print(f"  всего {num(idx_total)} ₽")
    if args.out:
        comment = [
            f"Токены — средние по журналу демо-стенда; цены OpenAI Standard (developers.openai.com/api/docs/pricing),"
            f" цена входа средневзвешенная с учётом кешированных токенов (кеш дешевле в 20 раз).",
            f"Разово, вне запроса: индексация руководств и распознавание сканов ≈ {num(idx_total)} ₽."
            if idx else "",
            "Цифры растут с длиной диалога (история в контексте) и числом поисков; модель меняется в .env без кода.",
        ]
        write_template(args.out, rows, cfg, [c for c in comment if c])
        print(f"\nшаблон заполнен: {args.out}")
    if args.fleet_out:
        table = fleet_table(calls, cfg, idx_total)
        with args.fleet_out.open("w", encoding="utf-8-sig", newline="") as fh:
            csv.writer(fh).writerows(table)
        for line in table[3:-2]:
            if any(line):
                print("  " + " | ".join(x for x in line if x))
        print(f"лист «Парк»: {args.fleet_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
