"""Доступ к интерфейсу и API по паролю (APP_PASSWORD). /health открыт всегда.

- Пароль задан: без cookie сессии /api/* → 401, остальные страницы → редирект на /login.
- Пароль не задан: всё, кроме /health, → 503 («закрыто по умолчанию»), чтобы стенд нельзя было
  случайно поднять открытым. Для локальной разработки — явный AUTH_DISABLED=1.
- Cookie сессии = HMAC-SHA256(пароль, метка версии): без хранилища сессий, без новых
  зависимостей; смена пароля разлогинивает всех. Сравнение — за постоянное время.
- Неверный пароль → пауза перед ответом (замедляет подбор), событие в журнал не пишется:
  к машине оно не относится, а сам пароль логировать нельзя.
"""
import asyncio
import hashlib
import hmac
from html import escape
from urllib.parse import parse_qs

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

from app.config import Settings

COOKIE_NAME = "dm_session"
COOKIE_MAX_AGE = 7 * 24 * 3600
PUBLIC_PATHS = frozenset({"/health", "/login", "/logout"})
FAILED_LOGIN_DELAY_S = 1.0
_SESSION_LABEL = b"digital-mechanic-session-v1"


def session_token(password: str) -> str:
    return hmac.new(password.encode("utf-8"), _SESSION_LABEL, hashlib.sha256).hexdigest()


def _password(settings: Settings) -> str | None:
    if settings.app_password is None:
        return None
    return settings.app_password.get_secret_value() or None


def is_authenticated(request: Request, settings: Settings) -> bool:
    if settings.auth_disabled:
        return True
    password = _password(settings)
    cookie = request.cookies.get(COOKIE_NAME)
    if not (password and cookie):
        return False
    # байты, а не str: compare_digest на str с не-ASCII символами бросает TypeError (→ 500)
    return hmac.compare_digest(cookie.encode("utf-8"), session_token(password).encode("utf-8"))


async def auth_guard(request: Request, settings: Settings) -> Response | None:
    """None — пропустить запрос дальше; иначе — готовый ответ-отказ."""
    path = request.url.path
    if path == "/health" or settings.auth_disabled:
        return None
    if _password(settings) is None:
        return PlainTextResponse("Доступ не настроен: задайте APP_PASSWORD", status_code=503)
    if path in PUBLIC_PATHS or is_authenticated(request, settings):
        return None
    if path.startswith("/api/"):
        return JSONResponse({"detail": "Требуется вход"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


def _is_https(request: Request) -> bool:
    # за Caddy приложение видит http; исходная схема — в X-Forwarded-Proto
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


LOGIN_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Цифровой механик — вход</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;padding:24px 16px;background:#f4f5f7;color:#1c1f24}}
form{{max-width:360px;margin:15vh auto 0;background:#fff;padding:24px;border-radius:12px;
box-shadow:0 1px 4px rgba(0,0,0,.08)}}
h1{{font-size:20px;margin:0 0 16px}}
input,button{{width:100%;box-sizing:border-box;font-size:18px;padding:12px;border-radius:8px}}
input{{border:1px solid #c4c8cf;margin-bottom:12px}}
button{{border:0;background:#1f6feb;color:#fff}}
.err{{color:#b42318;margin:0 0 12px}}
</style></head><body>
<form method="post" action="/login">
<h1>Цифровой механик</h1>
{error}
<input type="password" name="password" placeholder="Пароль" autocomplete="current-password" autofocus required>
<button type="submit">Войти</button>
</form></body></html>"""


def login_form(error: str | None = None, status_code: int = 200) -> HTMLResponse:
    block = f'<p class="err">{escape(error)}</p>' if error else ""
    return HTMLResponse(LOGIN_PAGE.format(error=block), status_code=status_code)


async def login_submit(request: Request, settings: Settings) -> Response:
    password = _password(settings)
    form = parse_qs((await request.body()).decode("utf-8", errors="replace"))
    given = (form.get("password") or [""])[0]
    if password is None:
        return PlainTextResponse("Доступ не настроен: задайте APP_PASSWORD", status_code=503)
    if not hmac.compare_digest(given.encode("utf-8"), password.encode("utf-8")):
        await asyncio.sleep(FAILED_LOGIN_DELAY_S)
        return login_form("Неверный пароль", status_code=401)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(COOKIE_NAME, session_token(password), max_age=COOKIE_MAX_AGE, httponly=True,
                        samesite="lax", secure=_is_https(request))
    return response


def logout() -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response
