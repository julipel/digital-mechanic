"""FastAPI-приложение: middleware request_id, /health, отдача фронтенда.

Бизнес-логики здесь нет — роуты сценариев появятся на этапе 6.
"""
import asyncio
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.storage.events import new_request_id, request_id_var

WEB_DIR = Path(__file__).resolve().parent / "web"
REQUEST_ID_HEADER = "X-Request-ID"


def create_app(settings: Settings | None = None, http_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Фабрика приложения. http_transport подменяется в тестах, чтобы не поднимать заглушки."""
    settings = settings or get_settings()
    app = FastAPI(title="Цифровой механик", version="0.1.0")
    app.state.settings = settings
    app.state.http_transport = http_transport

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
