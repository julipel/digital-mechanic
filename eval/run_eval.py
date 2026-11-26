"""Оценка поиска на золотом наборе eval/golden.jsonl.

Запуск из корня проекта:  python -m eval.run_eval  [--k 5] [--bm25-only] [--no-history]

Для каждого вопроса семейство двигателя берётся из карточки ТС (как в приложении), а не
из файла набора: так проверяется и сопоставление машины с руководствами.
Метрики:
- hit@1, hit@5 — среди k лучших есть фрагмент с ожидаемыми (документ, страница PDF);
- foreign — фрагменты документов ДРУГОГО двигателя в выдаче (должно быть 0 всегда);
- вопросы с expected=refuse в hit не входят: поиск всегда что-то вернёт, отказ — забота
  агента; для них печатается лучший фрагмент, чтобы видеть, насколько он «похож».
Итог дописывается в eval/history.csv.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.integrations.fleet import load_fleet
from app.retrieval.manuals import manuals_by_id
from app.retrieval.search import get_index, search

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden.jsonl"
HISTORY = EVAL_DIR / "history.csv"
HISTORY_FIELDS = ["ts", "index_built_at", "mode", "k", "questions", "hit1", "hit5", "mrr", "foreign", "misses"]


def load_golden(path: Path = GOLDEN) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_history() -> dict | None:
    if not HISTORY.exists():
        return None
    with HISTORY.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bm25-only", action="store_true", help="без эмбеддингов запроса")
    parser.add_argument("--no-history", action="store_true", help="не дописывать eval/history.csv")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.bm25_only:
        settings = settings.model_copy(update={"embedding_model": ""})
    fleet = load_fleet(settings.vehicles_path, settings.config_dir, settings.fault_types_path)
    manuals = manuals_by_id(settings.require_kb_dir())
    index = get_index(settings)
    golden = load_golden()

    scored = hit1 = hit5 = foreign = 0
    rr_sum = 0.0
    misses: list[str] = []
    mode = "?"
    for q in golden:
        vehicle = fleet.get(q["vehicle_id"])
        if vehicle is None or vehicle.engine_family is None:
            print(f"[{q['id']}] машина {q['vehicle_id']} не найдена или двигатель не сопоставлен")
            return 2
        if vehicle.engine_family != q["engine_family"]:
            print(f"[{q['id']}] в наборе {q['engine_family']}, у машины {vehicle.engine_family}")
            return 2
        result = search(q["question"], vehicle.engine_family, settings, k=args.k,
                        scenario="question", index=index, vehicle_id=q["vehicle_id"])
        mode = result.mode
        bad = [h for h in result.hits if manuals[h.doc_id].engine_family != vehicle.engine_family]
        foreign += len(bad)
        got = [(h.doc_id, h.pdf_page) for h in result.hits]
        top3 = ", ".join(f"{d} p{p}" for d, p in got[:3])
        if q["expected"] == "refuse":
            best = result.hits[0] if result.hits else None
            print(f"[{q['id']}] refuse  лучший: {top3}  bm25_rank={best.bm25_rank if best else None}")
            continue
        scored += 1
        expected = {(e["doc_id"], e["pdf_page"]) for e in q["expected"]}
        rank = next((i for i, g in enumerate(got, start=1) if g in expected), None)
        hit1 += rank == 1
        hit5 += rank is not None and rank <= 5
        rr_sum += 1 / rank if rank else 0.0
        status = f"rank {rank}" if rank else "MISS"
        print(f"[{q['id']}] {status:7} {q['question'][:60]}  → {top3}")
        if not rank:
            misses.append(q["id"])

    h1, h5, mrr = hit1 / scored, hit5 / scored, rr_sum / scored
    print()
    print(f"режим поиска: {mode}; вопросов с ответом: {scored}, вне базы: {len(golden) - scored}")
    print(f"hit@1 = {h1:.2f} ({hit1}/{scored})   hit@{args.k} = {h5:.2f} ({hit5}/{scored})   MRR = {mrr:.2f}")
    print(f"фрагментов чужого двигателя в выдаче: {foreign}")
    if misses:
        print("промахи: " + ", ".join(misses))
    prev = last_history()
    if prev:
        print(f"прошлый прогон ({prev['ts']}, {prev['mode']}): hit@1 = {prev['hit1']}, hit@5 = {prev['hit5']}")

    if not args.no_history:
        manifest = json.loads((settings.require_index_dir() / "manifest.json").read_text(encoding="utf-8"))
        new_file = not HISTORY.exists()
        with HISTORY.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "index_built_at": manifest.get("built_at"), "mode": mode, "k": args.k,
                "questions": scored, "hit1": f"{h1:.3f}", "hit5": f"{h5:.3f}", "mrr": f"{mrr:.3f}",
                "foreign": foreign, "misses": " ".join(misses),
            })
    return 1 if foreign else 0


if __name__ == "__main__":
    sys.exit(main())
