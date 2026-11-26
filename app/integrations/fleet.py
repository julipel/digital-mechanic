"""Парк ТС и история проверок из vehicles.json (заглушка ML списка машин не отдаёт).

Семейство двигателя определяется по engine_model через config/engines.json — по нему код
выбирает документы для поиска. Описания проверок очищаются от внедрённых инструкций
сразу при загрузке, чтобы история (память по ТС, график, журнал) не несла их дальше.
"""
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.integrations.models import Diagnosis
from app.policy.sanitize import sanitize_description

ENGINES_FILE = "engines.json"


class Session(Diagnosis):
    """Проверка из истории: те же поля, что у ответа /diagnose, плюс время."""
    checked_at: datetime
    description: str = ""
    model_version: str = ""
    injection_detected: bool = False


class Vehicle(BaseModel):
    model_config = ConfigDict(frozen=True)

    vehicle_id: str
    garage_no: str | None = None
    make: str | None = None
    model: str | None = None
    vehicle_type: str | None = None
    engine_model: str
    fuel: str | None = None
    engine_family: str | None = None  # None — двигатель не сопоставлен с руководствами
    sessions: tuple[Session, ...] = ()

    @property
    def last_session(self) -> Session | None:
        return self.sessions[-1] if self.sessions else None

    def session(self, session_id: str) -> Session | None:
        return next((s for s in self.sessions if s.session_id == session_id), None)


class Fleet:
    def __init__(self, vehicles: list[Vehicle]):
        self._by_id = {v.vehicle_id: v for v in vehicles}

    def list(self) -> list[Vehicle]:
        return list(self._by_id.values())

    def get(self, vehicle_id: str) -> Vehicle | None:
        """Госномер латиницей, регистр важен — как в заглушке, без нормализации."""
        return self._by_id.get(vehicle_id)


def load_engine_families(config_dir: Path) -> dict[str, str]:
    """engine_model → семейство."""
    with (Path(config_dir) / ENGINES_FILE).open(encoding="utf-8") as f:
        families = json.load(f)["families"]
    return {model: family for family, spec in families.items() for model in spec["engine_models"]}


@lru_cache
def fault_titles(path: Path) -> dict[str, str]:
    """Справочник F01…F14 → название (fault_types.json). Название неисправности везде берётся
    отсюда по коду, а не текстом из данных ML."""
    with Path(path).open(encoding="utf-8") as f:
        return {item["code"]: item["title"] for item in json.load(f)}


def load_fleet(vehicles_path: Path, config_dir: Path, fault_types_path: Path) -> Fleet:
    with Path(vehicles_path).open(encoding="utf-8") as f:
        raw = json.load(f)
    family_by_model = load_engine_families(config_dir)
    titles = fault_titles(fault_types_path)
    vehicles = []
    for v in raw["vehicles"]:
        sessions = []
        for s in v.get("sessions", []):
            clean = sanitize_description(s.get("description"))
            sessions.append(Session.model_validate(
                {**s, "vehicle_id": v["vehicle_id"], "description": clean.text,
                 "anomaly_title": titles.get(s["anomaly_type"]) if s.get("anomaly_type") else None,
                 "injection_detected": clean.injection_detected}))
        sessions.sort(key=lambda s: s.checked_at)
        vehicles.append(Vehicle.model_validate({
            **{k: val for k, val in v.items() if k != "sessions"},
            "engine_family": family_by_model.get(v["engine_model"]),
            "sessions": tuple(sessions),
        }))
    return Fleet(vehicles)


@lru_cache
def get_fleet(vehicles_path: Path, config_dir: Path, fault_types_path: Path) -> Fleet:
    return load_fleet(vehicles_path, config_dir, fault_types_path)
