"""Применимость типа неисправности (F01…F14) к семейству двигателя.

Таблица — в config/fault_applicability.json, не догадка модели. Если ML-сервис вернул код,
которого не может быть у этого двигателя (например, зажигание у дизеля), результат
считается противоречивым: агент сообщает об этом механику, заявка не создаётся.
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

APPLICABILITY_FILE = "fault_applicability.json"


@dataclass(frozen=True)
class Applicability:
    applicable: bool
    reason: str = ""
    confirmed: bool = False                       # подтверждено страницами руководства
    evidence: tuple[dict, ...] = field(default_factory=tuple)


@lru_cache
def load_table(config_dir: Path) -> dict[str, dict[str, dict]]:
    with (Path(config_dir) / APPLICABILITY_FILE).open(encoding="utf-8") as f:
        return json.load(f)["not_applicable"]


def check_applicability(config_dir: Path, engine_family: str | None, anomaly_type: str | None) -> Applicability:
    """Нет кода неисправности — проверять нечего (применимо). Неизвестное семейство
    двигателя — применимость не определить, и это тоже противоречие: документов нет."""
    if anomaly_type is None:
        return Applicability(applicable=True)
    if engine_family is None:
        return Applicability(applicable=False, reason="Модель двигателя не сопоставлена с руководствами")
    entry = load_table(Path(config_dir)).get(engine_family, {}).get(anomaly_type)
    if entry is None:
        return Applicability(applicable=True)
    return Applicability(
        applicable=False,
        reason=entry["reason"],
        confirmed=bool(entry.get("confirmed")),
        evidence=tuple(entry.get("evidence", [])),
    )
