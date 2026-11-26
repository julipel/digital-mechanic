"""OpenAI-совместимый клиент и учёт вызовов.

Ключ, base_url и имена моделей — только из настроек (.env). Каждый вызов эмбеддингов
пишется в llm_calls через app.storage.events.log_llm_call.
"""
import time

from openai import OpenAI

from app.config import Settings
from app.storage.events import log_llm_call


class LLMNotConfigured(RuntimeError):
    pass


def get_client(settings: Settings) -> OpenAI:
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise LLMNotConfigured("OPENAI_API_KEY не задан в .env")
    return OpenAI(api_key=settings.openai_api_key.get_secret_value(), base_url=settings.openai_base_url)


def embed(client: OpenAI, settings: Settings, texts: list[str], *, scenario: str, step: str) -> list[list[float]]:
    """Эмбеддинги одной пачки текстов, с записью вызова в llm_calls."""
    if not settings.embedding_model:
        raise LLMNotConfigured("EMBEDDING_MODEL не задан в .env")
    started = time.perf_counter()
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    log_llm_call(
        settings.db_path, scenario=scenario, step=step, model=settings.embedding_model,
        input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
