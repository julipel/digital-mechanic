"""Справочник руководств из манифеста базы знаний KB_DIR/kb.yaml (структура — kb.example.yaml).

Это единственный источник обозначений документов (references[].manual) и их принадлежности
семейству двигателя. Фильтр поиска и проверка ссылок берут семейство отсюда, а не от модели.
База знаний внешняя: манифест лежит рядом с PDF, вне репозитория.
"""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

MANIFEST_FILE = "kb.yaml"


class Manual(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    manual: str           # обозначение как в ТЗ — идёт в заявку
    file: str
    engine_family: str
    kind: str
    pages: int
    sha256: str | None = None
    edition: str | None = None
    year: int | None = None
    # правила извлечения (см. kb.example.yaml)
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
def load_manuals(kb_dir: Path) -> tuple[Manual, ...]:
    path = Path(kb_dir) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Нет манифеста базы знаний {path} (структура — kb.example.yaml)")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    manuals = tuple(Manual.model_validate(m) for m in data.get("manuals") or [])
    ids = [m.doc_id for m in manuals]
    if not manuals or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: пустой список manuals или повторяющиеся doc_id")
    return manuals


def check_families_known(manuals: tuple[Manual, ...], known_families: set[str]) -> None:
    """Манифест внешний: опечатка в engine_family молча выключила бы документ из поиска."""
    unknown = sorted({m.engine_family for m in manuals} - known_families)
    if unknown:
        raise ValueError(f"kb.yaml: семейства {unknown} нет в config/engines.json")


def manuals_by_id(kb_dir: Path) -> dict[str, Manual]:
    return {m.doc_id: m for m in load_manuals(kb_dir)}


def manuals_for_family(kb_dir: Path, engine_family: str) -> list[Manual]:
    return [m for m in load_manuals(kb_dir) if m.engine_family == engine_family]
