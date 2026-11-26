"""Общие фикстуры: настройки во временной папке и заглушки организаторов в процессе
(httpx.ASGITransport) — сеть и запущенные сервисы для тестов не нужны."""
import sys

import httpx
import pytest

from app.config import PROJECT_ROOT, Settings
from app.integrations.fleet import load_fleet

MOCKS_DIR = PROJECT_ROOT / "mocks"
ML_URL = "http://mock-ml.test"
TOIR_URL = "http://mock-toir.test"
HAS_DATA = (MOCKS_DIR / "data" / "vehicles.json").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="нет mocks/data/vehicles.json (не публикуется в git)")


class RoutingTransport(httpx.AsyncBaseTransport):
    """Направляет запрос в нужное ASGI-приложение по хосту."""

    def __init__(self, routes: dict[str, httpx.AsyncBaseTransport]):
        self.routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self.routes.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError("нет маршрута", request=request)
        return await transport.handle_async_request(request)


class DownTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("сервис недоступен", request=request)


def fixed_response(status_code: int, body) -> httpx.MockTransport:
    """Транспорт, всегда отвечающий заданным кодом и телом (для кодов, которые наш
    клиент сам не спровоцирует: 400, 418, некорректный ответ)."""
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, (dict, list)):
            return httpx.Response(status_code, json=body)
        return httpx.Response(status_code, text=body)
    return httpx.MockTransport(handler)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, mock_ml_url=ML_URL, mock_toir_url=TOIR_URL,
                    db_path=tmp_path / "test.sqlite")


@pytest.fixture
def mock_transport() -> RoutingTransport:
    if not HAS_DATA:
        pytest.skip("нет mocks/data/vehicles.json (не публикуется в git)")
    if str(MOCKS_DIR) not in sys.path:
        sys.path.insert(0, str(MOCKS_DIR))
    from mock_ml.app import app as ml_app
    from mock_toir.app import app as toir_app
    return RoutingTransport({
        "mock-ml.test": httpx.ASGITransport(app=ml_app),
        "mock-toir.test": httpx.ASGITransport(app=toir_app),
    })


@pytest.fixture
def fleet(settings):
    if not HAS_DATA:
        pytest.skip("нет mocks/data/vehicles.json (не публикуется в git)")
    return load_fleet(settings.vehicles_path, settings.config_dir, settings.fault_types_path)
