"""Справочник руководств из config/manuals.json.

Это единственный источник обозначений документов (references[].manual) и их принадлежности
семейству двигателя. Фильтр поиска и проверка ссылок берут семейство отсюда, а не от модели.
"""
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

MANUALS_FILE = "manuals.json"


class Manual(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    manual: str           # обозначение как в ТЗ — идёт в заявку
    file: str
    engine_family: str
    kind: str
    pages: int
    edition: str | None = None
    year: int | None = None
    # правила извлечения (см. _comment в manuals.json)
    page_number_offset: int | None = 0
    strip_patterns: tuple[str, ...] = ()
    title_band: tuple[float, float] | None = None
    scan_pages: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        """Подпись для интерфейса: обозначение + издание и год."""
        extra = ", ".join(x for x in (self.edition, str(self.year) if self.year else None) if x)
        return f"{self.manual} ({extra})" if extra else self.manual


@lru_cache
def load_manuals(config_dir: Path) -> tuple[Manual, ...]:
    with (Path(config_dir) / MANUALS_FILE).open(encoding="utf-8") as f:
        return tuple(Manual.model_validate(m) for m in json.load(f)["manuals"])


def manuals_by_id(config_dir: Path) -> dict[str, Manual]:
    return {m.doc_id: m for m in load_manuals(config_dir)}


def manuals_for_family(config_dir: Path, engine_family: str) -> list[Manual]:
    return [m for m in load_manuals(config_dir) if m.engine_family == engine_family]
