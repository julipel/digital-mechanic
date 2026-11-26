"""Настройки приложения из переменных окружения и файла .env.

Все адреса сервисов, ключи и имена моделей берутся только отсюда — в коде их нет
(аргумент «модель в контуре партнёра»: сменить провайдера = поменять .env).
Относительные пути считаются от корня проекта, чтобы запуск не зависел от текущей папки.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    # Агент: лимиты цикла инструментов (не модель — её выбор только в AGENT_MODEL)
    # reasoning_effort для Chat Completions; пусто — не передаётся. Некоторые модели принимают
    # инструменты в Chat Completions только с «none» (ответ провайдера 400 подскажет).
    agent_reasoning_effort: str | None = None
    # chat — Chat Completions (любой OpenAI-совместимый провайдер); responses — Responses API OpenAI.
    agent_api: Literal["chat", "responses"] = "chat"
    agent_max_tool_rounds: int = 6       # раундов с вызовом инструментов на одно действие механика
    agent_search_k: int = 5              # фрагментов в выдаче search_manual
    agent_chunk_chars: int = 2500        # сколько символов фрагмента видит модель
    agent_history_messages: int = 12     # сколько прошлых реплик диалога передавать модели

    # Доступ к интерфейсу и API. Пароль не задан → всё, кроме /health, отвечает 503,
    # если явно не включён AUTH_DISABLED=1 (только для локальной разработки).
    app_password: SecretStr | None = None
    auth_disabled: bool = False

    # База знаний — внешняя (CLAUDE.md, «База знаний — внешняя»): каталоги вне репозитория
    # и образа, значений по умолчанию нет. KB_DIR — PDF + манифест kb.yaml; INDEX_DIR — индекс.
    kb_dir: Path | None = None
    index_dir: Path | None = None

    # Данные и хранилища
    db_path: Path = Path("var/mechanic.sqlite")
    vehicles_path: Path = Path("mocks/data/vehicles.json")
    fault_types_path: Path = Path("mocks/data/fault_types.json")
    config_dir: Path = Path("config")  # engines.json, fault_applicability.json

    @field_validator("kb_dir", "index_dir", "db_path", "vehicles_path", "fault_types_path", "config_dir",
                     mode="before")
    @classmethod
    def _resolve_from_root(cls, value):
        if value is None or value == "":
            return None
        value = Path(value)
        return value if value.is_absolute() else PROJECT_ROOT / value

    def require_kb_dir(self) -> Path:
        if self.kb_dir is None:
            raise RuntimeError("KB_DIR не задан в .env: каталог базы знаний (PDF + kb.yaml)")
        return self.kb_dir

    def require_index_dir(self) -> Path:
        if self.index_dir is None:
            raise RuntimeError("INDEX_DIR не задан в .env: каталог поискового индекса")
        return self.index_dir

    @field_validator("mock_ml_url", "mock_toir_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("openai_base_url", "agent_reasoning_effort")
    @classmethod
    def _empty_to_none(cls, value: str | None) -> str | None:
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
