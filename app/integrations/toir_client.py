"""Клиент системы ТОиР: POST {MOCK_TOIR_URL}/tickets по контракту ТЗ v6.

Клиент только отправляет уже проверенную заявку. Смысловые проверки (валидатор правила 8,
подтверждение механиком, одна проверка — одна заявка) — в policy/ до вызова клиента.
"""
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import ValidationError

from app.integrations.errors import IntegrationError, InvalidResponse
from app.integrations.http import post_json
from app.integrations.models import TicketRequest
from app.storage.events import log_event

SERVICE = "toir"


class TicketService(Protocol):
    async def create_ticket(self, ticket: TicketRequest) -> str: ...


class HttpToirClient:
    def __init__(self, base_url: str, timeout: float = 10.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url
        self.timeout = timeout
        self.transport = transport

    async def create_ticket(self, ticket: TicketRequest) -> str:
        """Возвращает ticket_id. 201 — успех; иначе доменная ошибка."""
        r = await post_json(service=SERVICE, base_url=self.base_url, path="/tickets",
                            payload=ticket.model_dump(mode="json"),
                            timeout=self.timeout, transport=self.transport)
        try:
            body = r.json()
            ticket_id = body["ticket_id"]
            if r.status_code != 201 or body.get("status") != "created" or not isinstance(ticket_id, str):
                raise ValueError(body)
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise InvalidResponse(f"toir: ответ не по контракту ({r.status_code})", service=SERVICE,
                                  status_code=r.status_code) from exc
        return ticket_id


async def submit_ticket(service: TicketService, ticket: TicketRequest, db_path: Path) -> str:
    """Отправка с записью в журнал: ticket_created или toir_error (ошибка пробрасывается)."""
    try:
        ticket_id = await service.create_ticket(ticket)
    except IntegrationError as exc:
        log_event(db_path, "toir_error", vehicle_id=ticket.vehicle_id, session_id=ticket.session_id,
                  data={"error": type(exc).__name__, "status_code": exc.status_code, "detail": exc.detail})
        raise
    log_event(db_path, "ticket_created", vehicle_id=ticket.vehicle_id, session_id=ticket.session_id,
              data={"ticket_id": ticket_id, "anomaly_type": ticket.anomaly_type, "urgency": ticket.urgency,
                    "references": [r.model_dump() for r in ticket.references]})
    return ticket_id
