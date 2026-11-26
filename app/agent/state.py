"""Состояние диалога механика с агентом по одной машине и одной проверке.

Что переживает ход (одно действие механика):
- машина и проверка — их выбрал механик в интерфейсе, модель их не меняет;
- результат диагностики (CheckedDiagnosis) и режим — после первого вызова get_diagnosis;
- реплики механика и ИТОГОВЫЕ ответы агента (уже с подписями ссылок от кода).
Что НЕ переживает ход: выдача поиска и chunk_id — правило 1 требует ссылаться только на
фрагменты, найденные в этом ходе; результаты инструментов прошлых ходов модели не передаются.

Хранилище — в памяти процесса: диалог — рабочая сессия механика, журнал событий и заявки
лежат в SQLite отдельно и рестарт переживают.
"""
import uuid
from dataclasses import dataclass, field

from app.integrations.diagnosis import CheckedDiagnosis
from app.policy import modes

MAX_CONVERSATIONS = 500


@dataclass
class Conversation:
    vehicle_id: str
    session_id: str
    conversation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    checked: CheckedDiagnosis | None = None
    mode: modes.ModeDecision = field(default_factory=modes.no_check_mode)
    history: list[dict] = field(default_factory=list)   # {"role": "user"|"assistant", "content": str}


class ConversationStore:
    def __init__(self, limit: int = MAX_CONVERSATIONS):
        self._items: dict[str, Conversation] = {}
        self.limit = limit

    def create(self, vehicle_id: str, session_id: str) -> Conversation:
        if len(self._items) >= self.limit:                 # самый старый диалог уходит
            self._items.pop(next(iter(self._items)))
        conv = Conversation(vehicle_id=vehicle_id, session_id=session_id)
        self._items[conv.conversation_id] = conv
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        return self._items.get(conversation_id)
