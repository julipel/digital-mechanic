"""Модели данных по контракту ТЗ v6 (раздел 5): ответ /diagnose и заявка в ТОиР."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FAULT_CODE_PATTERN = r"^F(0[1-9]|1[0-4])$"

Status = Literal["normal", "anomaly", "check_failed"]
SignalQuality = Literal["good", "warning", "bad"]
Urgency = Literal["low", "medium", "high"]


class Diagnosis(BaseModel):
    """Ответ POST /diagnose. Лишние поля игнорируются, обязательные — строго по схеме.

    Сверх схемы проверяются инварианты из описания заглушки: при anomaly код неисправности
    есть, при normal — нет. Ответ, нарушающий их, считается некорректным (выводов не делаем).
    """
    model_config = ConfigDict(extra="ignore", frozen=True)

    vehicle_id: str
    session_id: str
    status: Status
    anomaly_type: str | None = Field(default=None, pattern=FAULT_CODE_PATTERN)
    anomaly_title: str | None = None
    anomaly_score: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    signal_quality: SignalQuality
    description: str
    model_version: str

    @model_validator(mode="after")
    def _status_invariants(self):
        if self.status == "anomaly" and self.anomaly_type is None:
            raise ValueError("status=anomaly без anomaly_type")
        if self.status == "normal" and self.anomaly_type is not None:
            raise ValueError("status=normal с anomaly_type")
        return self


class Reference(BaseModel):
    manual: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)


class TicketRequest(BaseModel):
    """Тело POST /tickets, имена полей — ровно как в ТЗ v6 (старые имена дают 418)."""
    model_config = ConfigDict(extra="forbid")

    vehicle_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    anomaly_type: str = Field(pattern=FAULT_CODE_PATTERN)
    conclusion: str = Field(min_length=1)
    urgency: Urgency
    operations: list[str] = Field(min_length=1)
    spare_parts: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(min_length=1)
