"""Режим проверки и набор инструментов, доступных агенту (правила 4–6 CLAUDE.md).

Это ОГРАНИЧЕНИЕ набора инструментов, а не маршрутизация сценария (F5): код решает, каких
инструментов у модели нет (например, при сбое записи — ни диагноза, ни заявки), а какой
из доступных вызвать — решает модель.

Режимы:
- no_check — проверка ещё не запрошена: можно получить результат и искать в руководстве;
- check_failed — сбой записи или плохое качество сигнала: выводов о двигателе нет,
  инструментов диагноза и заявки нет (правило 4);
- ml_inconsistent — код неисправности невозможен для этого двигателя: заявки нет (правило 6);
- normal — отклонений нет: заявка не нужна (валидатор её всё равно отклонит);
- anomaly / anomaly_low_confidence — отклонение; при низкой уверенности или помехах
  заявка возможна, но агент первой фразой предупреждает (правило 5).

warnings — то, что агент обязан сказать первой фразой; интерфейс показывает их плашками.
"""
from dataclasses import dataclass
from pathlib import Path

from app.integrations.models import Diagnosis
from app.policy.applicability import Applicability
from app.policy.settings import load_policy

# Имена инструментов агента (реализация — app/agent, этап 5).
GET_DIAGNOSIS = "get_diagnosis"
SEARCH_MANUAL = "search_manual"
DRAFT_TICKET = "draft_ticket"
GET_TREND = "get_trend"

NO_CHECK = "no_check"
CHECK_FAILED = "check_failed"
ML_INCONSISTENT = "ml_inconsistent"
NORMAL = "normal"
ANOMALY = "anomaly"
ANOMALY_LOW_CONFIDENCE = "anomaly_low_confidence"

# Предупреждения — коды; тексты для механика — в интерфейсе/промпте.
W_CHECK_FAILED = "check_failed"
W_SIGNAL_WARNING = "signal_warning"
W_LOW_CONFIDENCE = "low_confidence"
W_INJECTION = "injection_detected"
W_ML_INCONSISTENT = "ml_inconsistent"

# get_trend (оценка тенденции, B2) — вывод о двигателе по ряду проверок, поэтому его нет
# при сбое записи (правило 4) и при коде, противоречащем двигателю (значение недостоверно).
_TOOLS = {
    NO_CHECK: (GET_DIAGNOSIS, SEARCH_MANUAL),
    CHECK_FAILED: (SEARCH_MANUAL,),
    ML_INCONSISTENT: (SEARCH_MANUAL,),
    NORMAL: (SEARCH_MANUAL, GET_TREND),
    ANOMALY: (SEARCH_MANUAL, GET_TREND, DRAFT_TICKET),
    ANOMALY_LOW_CONFIDENCE: (SEARCH_MANUAL, GET_TREND, DRAFT_TICKET),
}


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    tools: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def allows(self, tool: str) -> bool:
        return tool in self.tools


def no_check_mode() -> ModeDecision:
    return ModeDecision(mode=NO_CHECK, tools=_TOOLS[NO_CHECK])


def decide_mode(diagnosis: Diagnosis, applicability: Applicability, injection_detected: bool,
                config_dir: Path) -> ModeDecision:
    """Режим по структурным полям ответа ML (текст description не участвует)."""
    threshold = load_policy(Path(config_dir))["modes"]["low_confidence_below"]
    warnings: list[str] = []
    if injection_detected:
        warnings.append(W_INJECTION)

    # Правило 4: сбой записи или плохой сигнал — сильнее любых других признаков.
    if diagnosis.status == "check_failed" or diagnosis.signal_quality == "bad":
        return _decision(CHECK_FAILED, [W_CHECK_FAILED, *warnings])

    if diagnosis.signal_quality == "warning":
        warnings.append(W_SIGNAL_WARNING)
    low_confidence = diagnosis.confidence is None or diagnosis.confidence < threshold
    if low_confidence:
        warnings.append(W_LOW_CONFIDENCE)

    if not applicability.applicable:
        return _decision(ML_INCONSISTENT, [W_ML_INCONSISTENT, *warnings])
    if diagnosis.status == "normal":
        return _decision(NORMAL, warnings)
    mode = ANOMALY_LOW_CONFIDENCE if (low_confidence or W_SIGNAL_WARNING in warnings) else ANOMALY
    return _decision(mode, warnings)


def _decision(mode: str, warnings: list[str]) -> ModeDecision:
    return ModeDecision(mode=mode, tools=_TOOLS[mode], warnings=tuple(dict.fromkeys(warnings)))
