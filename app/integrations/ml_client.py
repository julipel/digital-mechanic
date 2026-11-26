"""Клиент ML-сервиса акустической диагностики: POST {MOCK_ML_URL}/diagnose.

За интерфейсом DiagnosticsService: рабочий сервис подключается новым адаптером
без изменений в агенте. Ответ проверяется по контракту; несоответствие — InvalidResponse.
"""
from typing import Protocol

import httpx
from pydantic import ValidationError

from app.integrations.errors import InvalidResponse
from app.integrations.http import post_json
from app.integrations.models import Diagnosis

SERVICE = "ml"


class DiagnosticsService(Protocol):
    async def diagnose(self, vehicle_id: str, session_id: str) -> Diagnosis: ...


class HttpMlClient:
    def __init__(self, base_url: str, timeout: float = 10.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url
        self.timeout = timeout
        self.transport = transport

    async def diagnose(self, vehicle_id: str, session_id: str) -> Diagnosis:
        r = await post_json(service=SERVICE, base_url=self.base_url, path="/diagnose",
                            payload={"vehicle_id": vehicle_id, "session_id": session_id},
                            timeout=self.timeout, transport=self.transport)
        try:
            diagnosis = Diagnosis.model_validate(r.json())
        except ValidationError as exc:
            # Только поля и типы нарушений: значения (в т.ч. сырой description) в журнал не идут.
            problems = [{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()]
            raise InvalidResponse("ml: ответ не по контракту", service=SERVICE,
                                  status_code=r.status_code, detail=problems) from None
        except ValueError:
            raise InvalidResponse("ml: ответ не JSON", service=SERVICE, status_code=r.status_code) from None
        # Ответ про другую машину или проверку — не наш результат, выводов не делаем.
        if (diagnosis.vehicle_id, diagnosis.session_id) != (vehicle_id, session_id):
            raise InvalidResponse("ml: ответ относится к другой машине или проверке", service=SERVICE,
                                  status_code=r.status_code,
                                  detail={"vehicle_id": diagnosis.vehicle_id, "session_id": diagnosis.session_id})
        return diagnosis
