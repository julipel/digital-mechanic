"""Пороги политик из config/policy.json — не в коде, чтобы заказчик мог их менять."""
import json
from functools import lru_cache
from pathlib import Path

POLICY_FILE = "policy.json"


@lru_cache
def load_policy(config_dir: Path) -> dict:
    with (Path(config_dir) / POLICY_FILE).open(encoding="utf-8") as f:
        return json.load(f)
