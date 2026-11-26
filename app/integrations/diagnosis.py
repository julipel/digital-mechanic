"""Получение результата проверки с защитой от ловушек в данных ML-сервиса.

Цепочка: проверить, что проверка принадлежит машине → вызвать ML → очистить description
→ сверить код неисправности с двигателем. Каждый шаг пишет событие в журнал.
На выходе — CheckedDiagnosis: структурные поля ML без изменений + флаги для режима
(policy/modes, этап 4) и для пометок механику. Сырой description дальше не передаётся.
"""
import hashlib
from dataclasses import dataclass

from app.config import Settings
from app.integrations.errors import IntegrationError, NotFound
from app.integrations.fleet import Fleet, Vehicle, fault_titles
from app.integrations.ml_client import DiagnosticsService
from app.integrations.models import Diagnosis
from app.policy.applicability import Applicability, check_applicability
from app.policy.sanitize import sanitize_description
from app.storage.events import log_event


@dataclass(frozen=True)
class CheckedDiagnosis:
    vehicle: Vehicle
    diagnosis: Diagnosis           # description очищен, anomaly_title — из справочника; числа и коды — как от ML
    injection_detected: bool       # в описании был посторонний текст — показать пометку
    applicability: Applicability   # applicable=False → результат ML противоречит двигателю

    @property
    def ml_inconsistent(self) -> bool:
        return not self.applicability.applicable


async def get_checked_diagnosis(
    ml: DiagnosticsService, fleet: Fleet, vehicle_id: str, session_id: str,
    settings: Settings,
) -> CheckedDiagnosis:
    db_path = settings.db_path
    vehicle = fleet.get(vehicle_id)
    problem = ("Машина не найдена в парке" if vehicle is None
               else "Проверка не принадлежит этой машине" if vehicle.session(session_id) is None else None)
    if problem:
        log_event(db_path, "error", vehicle_id=vehicle_id, session_id=session_id,
                  data={"where": "diagnose", "error": "NotFound", "message": problem})
        raise NotFound(problem, service="fleet")

    log_event(db_path, "diagnose_called", vehicle_id=vehicle_id, session_id=session_id)
    try:
        raw = await ml.diagnose(vehicle_id, session_id)
    except IntegrationError as exc:
        log_event(db_path, "error", vehicle_id=vehicle_id, session_id=session_id,
                  data={"where": "diagnose", "error": type(exc).__name__,
                        "status_code": exc.status_code, "message": str(exc)})
        raise

    clean = sanitize_description(raw.description)
    # Название типа — из нашего справочника по коду, а не текстом от ML.
    title = fault_titles(settings.fault_types_path).get(raw.anomaly_type) if raw.anomaly_type else None
    diagnosis = raw.model_copy(update={"description": clean.text, "anomaly_title": title})
    applicability = check_applicability(settings.config_dir, vehicle.engine_family, diagnosis.anomaly_type)

    log_event(db_path, "diagnose_result", vehicle_id=vehicle_id, session_id=session_id, data={
        "status": diagnosis.status, "anomaly_type": diagnosis.anomaly_type,
        "anomaly_score": diagnosis.anomaly_score, "confidence": diagnosis.confidence,
        "signal_quality": diagnosis.signal_quality, "model_version": diagnosis.model_version,
        "ml_title_differs": (raw.anomaly_title or None) != title,
    })
    if clean.injection_detected:
        log_event(db_path, "injection_detected", vehicle_id=vehicle_id, session_id=session_id,
                  data={"removed_chars": len(clean.removed),
                        "removed_sha256": hashlib.sha256(clean.removed.encode("utf-8")).hexdigest()[:16]})
    if not applicability.applicable:
        log_event(db_path, "ml_inconsistent", vehicle_id=vehicle_id, session_id=session_id, data={
            "anomaly_type": diagnosis.anomaly_type, "engine_model": vehicle.engine_model,
            "engine_family": vehicle.engine_family, "reason": applicability.reason,
        })
    return CheckedDiagnosis(vehicle=vehicle, diagnosis=diagnosis,
                            injection_detected=clean.injection_detected, applicability=applicability)

