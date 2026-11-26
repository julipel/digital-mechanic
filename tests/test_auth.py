"""Доступ по паролю: /health открыт, остальное — только после входа; без пароля — закрыто."""
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.auth as auth
from app.auth import COOKIE_NAME, session_token
from app.main import create_app
from tests.conftest import DownTransport


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    monkeypatch.setattr(auth, "FAILED_LOGIN_DELAY_S", 0)


def _client(settings, **overrides):
    if isinstance(overrides.get("app_password"), str):
        overrides["app_password"] = SecretStr(overrides["app_password"])
    s = settings.model_copy(update=overrides)
    return TestClient(create_app(s, http_transport=DownTransport()), follow_redirects=False)


def test_health_open_without_login(settings):
    for overrides in ({}, {"app_password": "secret"}):
        r = _client(settings, **overrides).get("/health")
        assert r.status_code == 503 and "services" in r.json()  # заглушки «лежат», но доступ открыт


def test_closed_when_password_not_set(settings):
    client = _client(settings)
    assert client.get("/").status_code == 503
    assert client.get("/api/vehicles").status_code == 503
    assert client.post("/login", data={"password": ""}).status_code == 503


def test_auth_disabled_flag_opens_everything(settings):
    assert _client(settings, auth_disabled=True).get("/").status_code == 200


def test_redirect_and_401_without_session(settings):
    client = _client(settings, app_password="secret")
    r = client.get("/")
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/api/vehicles").status_code == 401
    assert client.get("/login").status_code == 200


def test_wrong_password_rejected(settings):
    client = _client(settings, app_password="secret")
    r = client.post("/login", data={"password": "nope"})
    assert r.status_code == 401 and COOKIE_NAME not in r.cookies
    assert client.get("/").status_code == 303


def test_login_sets_httponly_cookie_and_opens_app(settings):
    client = _client(settings, app_password="secret")
    r = client.post("/login", data={"password": "secret"})
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert "secret" not in r.headers["set-cookie"]          # в cookie не пароль, а HMAC
    assert client.get("/").status_code == 200


def test_forged_or_stale_cookie_rejected(settings):
    client = _client(settings, app_password="new-secret")
    client.cookies.set(COOKIE_NAME, session_token("old-secret"))
    assert client.get("/").status_code == 303
    client.cookies.set(COOKIE_NAME, "0" * 64)
    assert client.get("/api/x").status_code == 401


def test_secure_cookie_behind_https_proxy(settings):
    client = _client(settings, app_password="secret")
    r = client.post("/login", data={"password": "secret"}, headers={"X-Forwarded-Proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_logout_clears_session(settings):
    client = _client(settings, app_password="secret")
    client.post("/login", data={"password": "secret"})
    r = client.get("/logout")
    assert r.status_code == 303
    client.cookies.clear()
    assert client.get("/").status_code == 303


def test_non_ascii_cookie_does_not_crash(settings):
    client = _client(settings, app_password="secret")
    # сырые байты UTF-8 в заголовке — как их может прислать браузер или сканер
    raw = {"Cookie": f"{COOKIE_NAME}=ключ-ё".encode("utf-8")}
    assert client.get("/", headers=raw).status_code == 303
    assert client.get("/api/x", headers=raw).status_code == 401
