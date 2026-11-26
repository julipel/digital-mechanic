"""Общий HTTP-вызов для клиентов заглушек: JSON в UTF-8, перевод ошибок в доменные."""
import httpx

from app.integrations.errors import (BadRequest, DeliveryUnknown, LegacySchema, NotDelivered, NotFound,
                                     ServiceUnavailable)


async def post_json(
    *, service: str, base_url: str, path: str, payload: dict, timeout: float,
    transport: httpx.AsyncBaseTransport | None,
) -> httpx.Response:
    """POST с телом в UTF-8. Возвращает ответ 2xx, иначе бросает доменную ошибку."""
    # trust_env=False: внутренние сервисы, системный HTTP_PROXY к ним не применяем.
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout,
                                     transport=transport, trust_env=False) as client:
            r = await client.post(path, json=payload)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise NotDelivered(f"{service}: нет соединения ({type(exc).__name__})", service=service) from exc
    except httpx.HTTPError as exc:
        raise DeliveryUnknown(f"{service}: нет ответа ({type(exc).__name__})", service=service) from exc

    if r.is_success:
        return r
    detail = _detail(r)
    code = r.status_code
    if code == 404:
        raise NotFound(f"{service}: не найдено", service=service, status_code=code, detail=detail)
    if code == 418:
        raise LegacySchema(f"{service}: устаревшая схема полей", service=service, status_code=code, detail=detail)
    if code in (400, 422):
        raise BadRequest(f"{service}: запрос отклонён ({code})", service=service, status_code=code, detail=detail)
    raise ServiceUnavailable(f"{service}: ошибка сервиса ({code})", service=service, status_code=code, detail=detail)


def _detail(r: httpx.Response):
    try:
        return r.json()
    except ValueError:
        return r.text[:500]
