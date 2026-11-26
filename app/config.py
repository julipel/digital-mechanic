"""Настройки приложения из переменных окружения и файла .env.

Все адреса сервисов, ключи и имена моделей берутся только отсюда — в коде их нет
(аргумент «модель в контуре партнёра»: сменить провайдера = поменять .env).
Относительные пути считаются от корня проекта, чтобы запуск не зависел от текущей папки.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Внешние сервисы (имена — как в .env.example заглушек организаторов)
    mock_ml_url: str = "http://127.0.0.1:8001"
    mock_toir_url: str = "http://127.0.0.1:8002"
    http_timeout_s: float = 10.0

    # LLM: OpenAI-совместимый провайдер. Модели не имеют значений по умолчанию —
    # пустая строка означает «не настроено», ошибка будет при первом обращении к модели.
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    agent_model: str = ""
    embedding_model: str = ""
    vision_model: str = ""

    # Данные и хранилища
    manuals_dir: Path = Path("data/manuals")
    index_dir: Path = Path("index")
    db_path: Path = Path("var/mechanic.sqlite")
    vehicles_path: Path = Path("mocks/data/vehicles.json")
    fault_types_path: Path = Path("mocks/data/fault_types.json")

    @field_validator("manuals_dir", "index_dir", "db_path", "vehicles_path", "fault_types_path")
    @classmethod
    def _resolve_from_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("mock_ml_url", "mock_toir_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("openai_base_url")
    @classmethod
    def _empty_to_none(cls, value: str | None) -> str | None:
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
