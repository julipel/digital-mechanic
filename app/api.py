"""HTTP API интерфейса механика. Тонкий слой: правила — в policy/, сценарий — у агента.

Роуты:
- GET  /api/vehicles                    — парк: госномер, модель, двигатель, последняя проверка;
- GET  /api/vehicles/{id}               — карточка, история проверок (новые сверху), наши заявки;
- GET  /api/vehicles/{id}/trend?session_id= — оценка тенденции anomaly_score на момент проверки
                                          (policy/trend, без модели) — для графика в карточке;
- POST /api/diagnose                    — кнопка «Получить результат диагностики»: новый диалог
                                          и ход агента (SSE). Что вызвать — решает агент (F5);
- POST /api/chat                        — сообщение механика в диалоге (SSE);
- POST /api/tickets/{draft_id}/confirm  — кнопка «Отправить в ТОиР»: id + хэш показанного черновика;
- POST /api/tickets/{draft_id}/cancel   — «Отмена»;
- GET  /api/manuals/{doc_id}/pages/{n}.png — картинка одной страницы: только из ссылок, выданных
                                          кодом в ходах агента, с ограничением частоты.

SSE: каждое событие агента — строка «data: {json}» (типы — в app/agent/loop.py), первым идёт
{"type": "conversation", "conversation_id"}. Поток не падает: ошибка → событие "error".
"""
import json
from dataclasses import dataclass, field
from typing import Callable, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent import messages as msg
from app.agent.loop import AgentDeps, ChatModel, run_turn
from app.agent.state import Conversation, ConversationStore
from app.auth import COOKIE_NAME
from app.config import Settings
from app.integrations.errors import BadRequest, IntegrationError, NotDelivered, NotFound
from app.integrations.fleet import Fleet, Vehicle
from app.integrations.ml_client import DiagnosticsService
from app.integrations.toir_client import TicketService
from app.policy.tickets import TicketError, cancel_draft, confirm_draft
from app.policy.trend import vehicle_trend
from app.retrieval.pages import PageNotFound, RateLimiter, render_page
from app.retrieval.search import search as default_search
from app.storage import tickets as ticket_store
from app.storage.events import log_event, request_id_var

# Текст хода для кнопки «Получить результат диагностики». Это просьба механика, а не команда
# коду: инструменты выбирает модель из набора, разрешённого режимом.
DIAGNOSE_REQUEST = "Покажи результат выбранной проверки и объясни его."


@dataclass
class Services:
    settings: Settings
    fleet_fn: Callable[[], Fleet]
    ml: DiagnosticsService
    toir: TicketService
    model_fn: Callable[[], ChatModel]
    search_fn: Callable = default_search
    manuals_fn: Callable[[str], list[str]] | None = None
    conversations: ConversationStore = field(default_factory=ConversationStore)
    page_limiter: RateLimiter | None = None
    busy: set[str] = field(default_factory=set)       # диалоги, по которым идёт ход
    # (doc_id, pdf_page) из ссылок и источников, выданных кодом; в памяти процесса
    cited_pages: set[tuple[str, int]] = field(default_factory=set)

    def deps(self) -> AgentDeps:
        return AgentDeps(settings=self.settings, fleet=self.fleet(), ml=self.ml, model=self.model_fn(),
                         search_fn=self.search_fn, manuals_fn=self.manuals_fn)

    def fleet(self) -> Fleet:
        try:
            return self.fleet_fn()
        except FileNotFoundError:
            raise HTTPException(503, "Нет данных парка (vehicles.json)") from None


class DiagnoseIn(BaseModel):
    vehicle_id: str = Field(min_length=1, max_length=32)
    session_id: str = Field(min_length=1, max_length=32)


class ChatIn(BaseModel):
    # Диалог начинается кнопкой диагностики или первым вопросом: тогда conversation_id нет,
    # а машина и проверка — те, что механик выбрал на экране.
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    vehicle_id: str | None = Field(default=None, min_length=1, max_length=32)
    session_id: str | None = Field(default=None, min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=2000)
    # Только для учёта стоимости по сценариям (llm_calls); на набор инструментов не влияет.
    scenario: Literal["question", "ticket"] = "question"


class ConfirmIn(BaseModel):
    hash: str = Field(min_length=64, max_length=64)


def _session_view(s) -> dict:
    return {"session_id": s.session_id, "checked_at": s.checked_at.isoformat(), "status": s.status,
            "anomaly_type": s.anomaly_type, "anomaly_title": s.anomaly_title, "anomaly_score": s.anomaly_score,
            "confidence": s.confidence, "signal_quality": s.signal_quality}


def _earlier_anomaly(v: Vehicle, ticket_ids: dict[tuple[str, str], str]) -> dict | None:
    """Последнее отклонение, после которого были ещё проверки (ловушка данных: после отклонения
    приходит «норма» — ремонт или ложное срабатывание). Без этого отклонение прячется за
    последней проверкой. Если последняя проверка сама — отклонение, оно и так видно: None.
    Только показ: на режим и набор инструментов агента не влияет."""
    if not v.sessions or v.sessions[-1].status == "anomaly":
        return None
    for s in reversed(v.sessions[:-1]):
        if s.status == "anomaly":
            return {"session_id": s.session_id, "checked_at": s.checked_at.isoformat(),
                    "anomaly_type": s.anomaly_type, "anomaly_title": s.anomaly_title,
                    "ticket_id": ticket_ids.get((v.vehicle_id, s.session_id))}
    return None


def _vehicle_summary(v: Vehicle, ticket_ids: dict[tuple[str, str], str]) -> dict:
    last = v.last_session
    return {"vehicle_id": v.vehicle_id, "garage_no": v.garage_no, "make": v.make, "model": v.model,
            "vehicle_type": v.vehicle_type, "engine_model": v.engine_model, "fuel": v.fuel,
            "manuals_available": v.engine_family is not None,
            "last_check": _session_view(last) if last else None,
            "earlier_anomaly": _earlier_anomaly(v, ticket_ids)}


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


def build_router(services: Services) -> APIRouter:
    router = APIRouter(prefix="/api")
    s = services.settings

    @router.get("/vehicles")
    def vehicles():
        ticket_ids = ticket_store.created_ticket_ids(s.db_path)
        return [_vehicle_summary(v, ticket_ids) for v in services.fleet().list()]

    @router.get("/vehicles/{vehicle_id}")
    def vehicle(vehicle_id: str):
        v = services.fleet().get(vehicle_id)
        if v is None:
            raise HTTPException(404, "Машина не найдена")
        tickets = [{"ticket_id": t["ticket_id"], "session_id": t["session_id"], "created_at": t["created_at"],
                    "urgency": t["payload"].get("urgency"), "anomaly_type": t["payload"].get("anomaly_type")}
                   for t in ticket_store.list_tickets(s.db_path, vehicle_id)]
        return {**_vehicle_summary(v, ticket_store.created_ticket_ids(s.db_path)), "sessions": [_session_view(x) for x in reversed(v.sessions)],
                "tickets": tickets}

    @router.get("/vehicles/{vehicle_id}/trend")
    def trend(vehicle_id: str, session_id: str | None = None):
        v = services.fleet().get(vehicle_id)
        if v is None:
            raise HTTPException(404, "Машина не найдена")
        try:
            return vehicle_trend(v, session_id, s.config_dir).as_dict()
        except KeyError:
            raise HTTPException(404, "Проверка не найдена у этой машины") from None

    def stream(conv: Conversation, text: str, scenario: str) -> StreamingResponse:
        if conv.conversation_id in services.busy:
            raise HTTPException(409, "Предыдущий запрос ещё выполняется")
        deps = services.deps()
        request_id = request_id_var.get()

        async def events():
            # Метку «занят» ставим внутри генератора: если клиент оборвёт связь до начала потока,
            # генератор не стартует и метка не останется навсегда.
            if conv.conversation_id in services.busy:
                yield _sse({"type": "error", "message": "Предыдущий запрос ещё выполняется"})
                return
            services.busy.add(conv.conversation_id)
            # поток читается уже после выхода из middleware — request_id действия ставим сами
            token = request_id_var.set(request_id)
            try:
                yield _sse({"type": "conversation", "conversation_id": conv.conversation_id,
                            "vehicle_id": conv.vehicle_id, "session_id": conv.session_id})
                async for event in run_turn(conv, text, deps, scenario=scenario, request_id=request_id):
                    if event["type"] == "diagnosis":
                        event = {**event, "warnings_ru": [msg.WARNINGS_RU[w] for w in event["warnings"]]}
                    # страницы, которые можно открыть картинкой, — только те, на что сослался код
                    for ref in event.get("citations") or event.get("sources") or ():
                        services.cited_pages.add((ref["doc_id"], ref["pdf_page"]))
                    yield _sse(event)
            except Exception as exc:              # поток не должен обрываться молча
                log_event(s.db_path, "error", vehicle_id=conv.vehicle_id, session_id=conv.session_id,
                          data={"where": "api_stream", "error": type(exc).__name__})
                yield _sse({"type": "error", "message": "Внутренняя ошибка — повторите запрос"})
            finally:
                services.busy.discard(conv.conversation_id)
                try:
                    request_id_var.reset(token)
                except ValueError:                # генератор закрыт из другого контекста (обрыв связи)
                    pass

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def new_conversation(vehicle_id: str | None, session_id: str | None) -> Conversation:
        v = services.fleet().get(vehicle_id) if vehicle_id else None
        if v is None or not session_id or v.session(session_id) is None:
            raise HTTPException(404, "Машина или проверка не найдены")
        return services.conversations.create(vehicle_id, session_id)

    @router.post("/diagnose")
    def diagnose(body: DiagnoseIn):
        return stream(new_conversation(body.vehicle_id, body.session_id), DIAGNOSE_REQUEST, "explain")

    @router.post("/chat")
    def chat(body: ChatIn):
        if body.conversation_id is None:
            return stream(new_conversation(body.vehicle_id, body.session_id), body.message, body.scenario)
        conv = services.conversations.get(body.conversation_id)
        if conv is None:
            raise HTTPException(404, "Диалог не найден — выберите машину заново")
        return stream(conv, body.message, body.scenario)

    @router.post("/tickets/{draft_id}/confirm")
    async def confirm(draft_id: str, body: ConfirmIn):
        try:
            ticket_id = await confirm_draft(draft_id, body.hash, services.toir, db_path=s.db_path)
        except TicketError as exc:
            raise HTTPException(409, str(exc)) from None
        except NotDelivered:
            raise HTTPException(502, "ТОиР недоступен — заявка не создана, повторите позже") from None
        except NotFound:
            raise HTTPException(502, "ТОиР не нашёл машину или проверку — заявка не создана") from None
        except BadRequest:
            raise HTTPException(502, "ТОиР не принял формат заявки — заявка не создана") from None
        except IntegrationError:
            raise HTTPException(502, "ТОиР ответил ошибкой — проверьте в ТОиР, создана ли заявка") from None
        return {"ticket_id": ticket_id, "status": "created"}

    @router.post("/tickets/{draft_id}/cancel")
    def cancel(draft_id: str):
        if not cancel_draft(draft_id, db_path=s.db_path):
            raise HTTPException(409, "Черновик уже отправляется, отправлен или отменён")
        return {"status": "cancelled"}

    @router.get("/manuals/{doc_id}/pages/{pdf_page}.png")
    def manual_page(doc_id: str, pdf_page: int, request: Request):
        limiter = services.page_limiter
        if limiter is not None and not limiter.allow(_client_key(request)):
            raise HTTPException(429, "Слишком много запросов страниц — подождите минуту")
        # Только страницы из ссылок, выданных кодом: руководство нельзя выкачать перебором номеров.
        if (doc_id, pdf_page) not in services.cited_pages:
            raise HTTPException(404, "Страница не найдена")
        try:
            path = render_page(s, doc_id, pdf_page)
        except PageNotFound:
            raise HTTPException(404, "Страница не найдена") from None
        except Exception as exc:                  # KB не подключена, PDF повреждён и т.п.
            log_event(s.db_path, "error", data={"where": "manual_page", "doc_id": doc_id, "pdf_page": pdf_page,
                                                "error": type(exc).__name__})
            raise HTTPException(503, "База знаний недоступна") from None
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    return router


def _client_key(request: Request) -> str:
    # За Caddy адрес клиента — в X-Forwarded-For (Caddy подставляет его сам).
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "-")
    return f"{ip}|{request.cookies.get(COOKIE_NAME, '')[:16]}"
