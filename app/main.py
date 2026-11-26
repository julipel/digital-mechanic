"""FastAPI-приложение: middleware request_id и доступа по паролю, /health, вход, API
интерфейса механика (app/api.py) и отдача фронтенда (app/web/). Бизнес-логики здесь нет.
"""
import asyncio
from pathlib import Path
from typing import Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.loop import ChatModel, make_chat_model
from app.api import Services, build_router
from app.auth import auth_guard, login_form, login_submit, logout
from app.config import Settings, get_settings
from app.integrations.fleet import get_fleet
from app.integrations.ml_client import HttpMlClient
from app.integrations.toir_client import HttpToirClient
from app.retrieval.pages import RateLimiter
from app.retrieval.search import search as default_search
from app.storage.events import new_request_id, request_id_var

WEB_DIR = Path(__file__).resolve().parent / "web"
REQUEST_ID_HEADER = "X-Request-ID"


def create_app(settings: Settings | None = None, http_transport: httpx.AsyncBaseTransport | None = None,
               *, chat_model: ChatModel | None = None, search_fn: Callable | None = None,
               manuals_fn: Callable[[str], list[str]] | None = None) -> FastAPI:
    """Фабрика приложения. В тестах подменяются http_transport (заглушки в процессе),
    модель, поиск и список документов — сеть и LLM не нужны."""
    settings = settings or get_settings()
    app = FastAPI(title="Цифровой механик", version="0.2.0")
    app.state.settings = settings
    app.state.http_transport = http_transport
    services = Services(
        settings=settings,
        fleet_fn=lambda: get_fleet(settings.vehicles_path, settings.config_dir, settings.fault_types_path),
        ml=HttpMlClient(settings.mock_ml_url, settings.http_timeout_s, transport=http_transport),
        toir=HttpToirClient(settings.mock_toir_url, settings.http_timeout_s, transport=http_transport),
        model_fn=(lambda: chat_model) if chat_model is not None else (lambda: make_chat_model(settings)),
        search_fn=search_fn or default_search,
        manuals_fn=manuals_fn,
        page_limiter=RateLimiter(settings.page_rate_per_minute),
    )
    app.state.services = services
    app.include_router(build_router(services))

    # Порядок: middleware, объявленный позже, — внешний. Проверка доступа внутри request_id,
    # чтобы и отказ получил заголовок X-Request-ID.
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        denied = await auth_guard(request, settings)
        return denied if denied is not None else await call_next(request)

    @app.get("/login", include_in_schema=False)
    async def login_page():
        return login_form()

    @app.post("/login", include_in_schema=False)
    async def login(request: Request):
        return await login_submit(request, settings)

    @app.get("/logout", include_in_schema=False)
    async def logout_route():
        return logout()

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        # Одно действие механика = один request_id; клиент может передать свой.
        rid = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

    @app.get("/health")
    async def health():
        """Доступность приложения и обеих заглушек. 503, если хотя бы одна недоступна."""
        targets = {"mock_ml": settings.mock_ml_url, "mock_toir": settings.mock_toir_url}
        # trust_env=False: заглушки и ТОиР — внутренние сервисы, системный HTTP_PROXY к ним не применяем.
        async with httpx.AsyncClient(timeout=3.0, transport=app.state.http_transport, trust_env=False) as client:
            results = await asyncio.gather(*(_ping(client, url) for url in targets.values()))
        services = dict(zip(targets, results))
        ok = all(s["ok"] for s in services.values())
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ok" if ok else "degraded", "services": services},
        )

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


async def _ping(client: httpx.AsyncClient, base_url: str) -> dict:
    try:
        r = await client.get(f"{base_url}/health")
        return {"ok": r.status_code == 200, "status_code": r.status_code}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": type(exc).__name__}


app = create_app()
