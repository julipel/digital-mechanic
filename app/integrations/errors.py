"""Доменные ошибки интеграций: вызывающий код решает, что сказать механику, а не ловит httpx."""


class IntegrationError(Exception):
    """Базовая ошибка внешнего сервиса."""

    def __init__(self, message: str, *, service: str, status_code: int | None = None, detail=None):
        super().__init__(message)
        self.service = service
        self.status_code = status_code
        self.detail = detail


class ServiceUnavailable(IntegrationError):
    """Сервис не отвечает, таймаут или 5xx — можно повторить позже."""


class NotDelivered(ServiceUnavailable):
    """Соединение не установлено — запрос точно не дошёл до сервиса, повтор безопасен."""


class DeliveryUnknown(ServiceUnavailable):
    """Запрос мог дойти, но ответа нет (таймаут чтения, обрыв). Для заявки повтор опасен:
    в ТОиР могла создаться заявка."""


class NotFound(IntegrationError):
    """404: неизвестная машина или проверка (в т.ч. проверка чужой машины)."""


class BadRequest(IntegrationError):
    """400 / 422: запрос не разобран или не по схеме — ошибка на нашей стороне."""


class LegacySchema(BadRequest):
    """418 от ТОиР: заявка собрана по старой схеме полей."""


class InvalidResponse(IntegrationError):
    """Сервис ответил, но ответ не соответствует контракту — выводов по нему не делаем."""
