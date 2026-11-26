"""Цикл агента на OpenAI tool calling (Chat Completions — работает у любого OpenAI-совместимого
провайдера; модель, ключ и base_url — из .env).

Один ход = одно действие механика (сообщение или кнопка):
1. Сообщения модели: системный промпт, состояние (машина, проверка, режим, доступные
   инструменты), последние реплики диалога, новое сообщение механика.
2. Раунд: модель получает ТОЛЬКО инструменты текущего режима (policy/modes) и либо вызывает
   их, либо отвечает текстом. После get_diagnosis режим меняется — со следующего раунда
   набор инструментов другой (например, при сбое записи заявки нет). Раундов с инструментами
   не больше AGENT_MAX_TOOL_ROUNDS; после — последний раунд без инструментов.
3. Ответ стримится построчно. Каждая строка проходит через CitationRegistry.render: маркеры
   [chunk_id] из выдачи ЭТОГО хода → «[документ, стр. N]» из метаданных, остальное похожее
   на ссылку вырезается. Строки, повторяющие системный промпт, не выводятся.
4. Первой строкой ответа код ставит предупреждения режима (правила 4–5) — не модель.
5. Каждый вызов модели → llm_calls (request_id, scenario, step, модель, токены, задержка).

run_turn — асинхронный генератор событий для интерфейса (этап 6 превращает их в SSE):
  {"type": "tool", "name", "step"}          — ход работы агента;
  {"type": "diagnosis", ...}                — карточка вердикта;
  {"type": "draft", ...}                    — карточка черновика заявки;
  {"type": "delta", "text"}                 — кусок ответа (готовые строки, со ссылками от кода);
  {"type": "done", "text", "citations", "tools", "mode", "warnings", ...}.
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Protocol

from app.agent import messages as msg
from app.agent.prompts import SYSTEM_PROMPT, state_message
from app.agent.state import Conversation
from app.agent.tools import ToolContext, allowed_tools, diagnosis_view, execute, tool_schemas
from app.config import Settings
from app.integrations.fleet import Fleet
from app.integrations.ml_client import DiagnosticsService
from app.policy import modes
from app.policy.citations import _HOMOGLYPHS, Citation, CitationRegistry
from app.retrieval.search import search as default_search
from app.storage.events import log_event, log_llm_call, request_id_var

SCENARIOS = ("explain", "question", "ticket")


# --- Модель за интерфейсом: в тестах подменяется сценарием без сети --------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class RoundResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class ChatModel(Protocol):
    def stream(self, messages: list[dict], tools: list[dict] | None) -> AsyncIterator:
        """Отдаёт ("text", кусок) по мере генерации и в конце ("end", RoundResult)."""


class OpenAIChatModel:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def stream(self, messages: list[dict], tools: list[dict] | None):
        from app.integrations.llm import get_async_client
        client = get_async_client(self.settings)
        kwargs = {"model": self.settings.agent_model, "messages": messages, "stream": True,
                  "stream_options": {"include_usage": True}}
        if tools:
            kwargs["tools"] = tools
        if self.settings.agent_reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.agent_reasoning_effort
        result = RoundResult(model=self.settings.agent_model)
        calls: dict[int, dict] = {}
        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            if chunk.usage is not None:
                result.input_tokens = chunk.usage.prompt_tokens or 0
                result.output_tokens = chunk.usage.completion_tokens or 0
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                result.cached_input_tokens = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                result.content += delta.content
                yield "text", delta.content
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        result.tool_calls = [ToolCall(**calls[i]) for i in sorted(calls)]
        yield "end", result


class OpenAIResponsesModel:
    """Тот же интерфейс поверх Responses API — для моделей, которые принимают инструменты
    только там (gpt-6.2-sol). Сообщения цикла (формат Chat Completions) переводятся во
    входные элементы Responses; ответ не сохраняется у провайдера (store=False)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def to_input(messages: list[dict]) -> list[dict]:
        items: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
                continue
            if m.get("content"):
                items.append({"role": m["role"], "content": m["content"]})
            for c in m.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                              "arguments": c["function"]["arguments"]})
        return items

    async def stream(self, messages: list[dict], tools: list[dict] | None):
        from app.integrations.llm import get_async_client
        client = get_async_client(self.settings)
        kwargs = {"model": self.settings.agent_model, "input": self.to_input(messages), "stream": True,
                  "store": False}
        if tools:
            kwargs["tools"] = [{"type": "function", **t["function"]} for t in tools]
        if self.settings.agent_reasoning_effort:
            kwargs["reasoning"] = {"effort": self.settings.agent_reasoning_effort}
        result = RoundResult(model=self.settings.agent_model)
        response = await client.responses.create(**kwargs)
        async for event in response:
            if event.type == "response.output_text.delta":
                result.content += event.delta
                yield "text", event.delta
            elif event.type == "response.output_item.done" and event.item.type == "function_call":
                result.tool_calls.append(ToolCall(id=event.item.call_id, name=event.item.name,
                                                  arguments=event.item.arguments or ""))
            elif event.type in ("response.completed", "response.incomplete") and event.response.usage:
                usage = event.response.usage
                result.input_tokens = usage.input_tokens or 0
                result.output_tokens = usage.output_tokens or 0
                details = getattr(usage, "input_tokens_details", None)
                result.cached_input_tokens = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            elif event.type in ("response.failed", "error"):
                raise RuntimeError(f"responses: {event.type}")
        yield "end", result


def make_chat_model(settings: Settings) -> ChatModel:
    """AGENT_API=chat (по умолчанию, любой OpenAI-совместимый провайдер) или responses."""
    return OpenAIResponsesModel(settings) if settings.agent_api == "responses" else OpenAIChatModel(settings)


@dataclass
class AgentDeps:
    settings: Settings
    fleet: Fleet
    ml: DiagnosticsService
    model: ChatModel
    search_fn: Callable = default_search
    manuals_fn: Callable[[str], list[str]] | None = None    # семейство → обозначения документов


def _manual_names(deps: AgentDeps, family: str) -> list[str]:
    if deps.manuals_fn is not None:
        return list(deps.manuals_fn(family))
    if deps.settings.kb_dir is None:
        return []
    from app.retrieval.manuals import manuals_for_family
    return [m.manual for m in manuals_for_family(deps.settings.kb_dir, family)]


# --- Защита от вывода системного промпта ---------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


_PROMPT_PIECES = [p for p in (_norm(re.sub(r"^\s*(?:[-•]|\d+\))\s*", "", x))
                              for x in re.split(r"(?<=[.:!?])\s+|\n", SYSTEM_PROMPT)) if len(p) >= 40]


def leaks_prompt(text: str) -> bool:
    """Строка ответа содержит предложение системного промпта (или его заметную часть)."""
    t = _norm(text)
    return any(piece[:60] in t for piece in _PROMPT_PIECES)


# Хвост строки, после которого может идти продолжение ссылки (страница, номер, скобка, цифра).
_OPEN_TAIL = re.compile(r"(?:стр\w*\.?|страниц\w*|лист\w*|\bс\.|\bсм\.|\bр{1,2}\.?|№|[(\[]|\d(?:\s*-?\s*[а-яё]{1,2})?)\s*$", re.IGNORECASE)


def _open_tail(line: str) -> bool:
    return bool(_OPEN_TAIL.search(line.translate(_HOMOGLYPHS)))


# --- Ход ------------------------------------------------------------------------------------

class _Renderer:
    """Построчная подстановка ссылок: строка выводится, только когда она закончена —
    маркер [chunk_id] не может разорваться между кусками стрима."""

    def __init__(self, registry: CitationRegistry | None):
        self.registry = registry
        self.buffer = ""
        self.text = ""
        self.citations: dict[str, Citation] = {}
        self.dropped: list[str] = []
        self.leaked = False

    def feed(self, piece: str) -> str:
        if self.leaked:
            return ""
        self.buffer += piece
        if "\n" not in self.buffer:
            return ""
        ready, rest = self.buffer.rsplit("\n", 1)
        # Ссылку-подделку можно разорвать переводом строки («см. стр.\n145», «145-я\nстраница»):
        # строка, которая заканчивается началом такой конструкции, ждёт следующую и рендерится с ней.
        while ready and _open_tail(ready.rsplit("\n", 1)[-1]):
            if "\n" not in ready:
                return ""                       # всё ещё ждём продолжения — буфер не трогаем
            ready, held = ready.rsplit("\n", 1)
            rest = held + "\n" + rest
        self.buffer = rest
        return self._render(ready + "\n")

    def flush(self) -> str:
        if self.leaked:
            return ""
        ready, self.buffer = self.buffer, ""
        return self._render(ready) if ready else ""

    def _render(self, text: str) -> str:
        if self.registry is not None:
            text, used, dropped = self.registry.render(text)
            self.citations.update({c.chunk_id: c for c in used})
            self.dropped += dropped
        else:                         # поиска не было — ссылок быть не может, режем всё похожее
            text, _, dropped = CitationRegistry("-", ()).render(text)
            self.dropped += dropped
        # окно через границу строк: предложение промпта, разорванное переводом строки, тоже ловится
        if leaks_prompt(self.text[-300:] + text):
            self.leaked = True
            text = msg.PROMPT_LEAK
        self.text += text
        return text


def _state(conv: Conversation, deps: AgentDeps, tools: tuple[str, ...]) -> dict:
    vehicle = deps.fleet.get(conv.vehicle_id)
    session = vehicle.session(conv.session_id) if vehicle else None
    state = {
        "vehicle": None if vehicle is None else {
            "vehicle_id": vehicle.vehicle_id, "garage_no": vehicle.garage_no, "model": vehicle.model,
            "engine_model": vehicle.engine_model, "fuel": vehicle.fuel,
            "manuals_available": vehicle.engine_family is not None},
        "selected_check": {"session_id": conv.session_id,
                           "checked_at": session.checked_at.isoformat() if session else None},
        "mode": conv.mode.mode,
        "warnings_already_shown": [msg.WARNINGS_RU[w] for w in conv.mode.warnings],
        "available_tools": list(tools),
    }
    if conv.checked is not None:
        state["diagnosis"] = diagnosis_view(conv.checked, conv.mode, deps.settings)
    else:
        state["diagnosis"] = "ещё не запрошен (инструмент get_diagnosis)"
    return state


async def run_turn(conv: Conversation, user_text: str, deps: AgentDeps, *, scenario: str = "question",
                   request_id: str | None = None) -> AsyncIterator[dict]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Неизвестный сценарий: {scenario}")
    token = request_id_var.set(request_id) if request_id else None
    try:
        async for event in _turn(conv, user_text, deps, scenario):
            yield event
    finally:
        if token is not None:
            request_id_var.reset(token)


async def _turn(conv: Conversation, user_text: str, deps: AgentDeps, scenario: str) -> AsyncIterator[dict]:
    s = deps.settings
    vehicle = deps.fleet.get(conv.vehicle_id)
    family = vehicle.engine_family if vehicle else None
    registry = CitationRegistry(family, _manual_names(deps, family)) if family else None
    ctx = ToolContext(settings=s, fleet=deps.fleet, ml=deps.ml, conv=conv, registry=registry,
                      scenario=scenario, search_fn=deps.search_fn)
    user_text = user_text.strip()[:4000]
    history = conv.history[-s.agent_history_messages:] if s.agent_history_messages > 0 else []
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, None,  # состояние — перед каждым раундом
                            *history, {"role": "user", "content": user_text}]
    renderer = _Renderer(registry)
    steps: list[str] = []
    prefix_sent = False
    failed = False

    def prefix() -> str:
        nonlocal prefix_sent
        if prefix_sent:
            return ""
        prefix_sent = True
        line = msg.warning_line(conv.mode.warnings)
        if line:
            renderer.text = line + "\n\n" + renderer.text    # строка уже отрисована — ставим перед ней
            return line + "\n\n"
        return ""

    for round_no in range(s.agent_max_tool_rounds + 1):
        tools = allowed_tools(conv, vehicle) if round_no < s.agent_max_tool_rounds else ()
        messages[1] = {"role": "system", "content": state_message(_state(conv, deps, tools))}
        # Пока результата проверки нет, текст раунда не стримим: если модель потом вызовет
        # get_diagnosis, предупреждения должны встать первой строкой, до любого текста.
        hold = conv.mode.mode == modes.NO_CHECK and modes.GET_DIAGNOSIS in tools
        started = time.perf_counter()
        result: RoundResult | None = None
        held: list[str] = []
        try:
            async for kind, payload in deps.model.stream(messages, tool_schemas(tools) or None):
                if kind == "end":
                    result = payload
                elif hold:
                    held.append(payload)
                else:
                    out = renderer.feed(payload)
                    if out:
                        yield {"type": "delta", "text": prefix() + out}
        except Exception as exc:              # сеть, ключ, лимиты провайдера — ход не падает
            log_event(s.db_path, "error", vehicle_id=conv.vehicle_id, session_id=conv.session_id,
                      data={"where": "agent_llm", "error": type(exc).__name__,
                            "status_code": getattr(exc, "status_code", None),
                            "round": round_no})
            failed = True
            break
        if result is None:
            failed = True
            break
        log_llm_call(s.db_path, scenario=scenario,
                     step="agent_tool_round" if result.tool_calls else "agent_final",
                     model=result.model or s.agent_model, input_tokens=result.input_tokens,
                     cached_input_tokens=result.cached_input_tokens, output_tokens=result.output_tokens,
                     latency_ms=int((time.perf_counter() - started) * 1000))

        if not result.tool_calls:
            if held:                           # ответ без инструментов в режиме no_check
                out = renderer.feed("".join(held))
                if out:
                    yield {"type": "delta", "text": prefix() + out}
            break

        messages.append({"role": "assistant", "content": result.content or None, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in result.tool_calls]})
        for call in result.tool_calls:
            step = msg.TOOL_STEPS_RU.get(call.name, call.name)
            steps.append(step)
            yield {"type": "tool", "name": call.name, "step": step}
            output = await execute(call.name, call.arguments, ctx, offered=tools)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(output, ensure_ascii=False)})
            for ui_event in ctx.ui_events:
                yield ui_event
            ctx.ui_events.clear()

    tail = renderer.flush()
    if failed:
        tail += ("\n" if renderer.text else "") + msg.LLM_UNAVAILABLE
        renderer.text += msg.LLM_UNAVAILABLE
    if tail or not prefix_sent:
        yield {"type": "delta", "text": prefix() + tail}
    answer = renderer.text.strip()

    citations = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "manual": c.manual, "pdf_page": c.pdf_page,
                  "label": c.label} for c in renderer.citations.values()]
    log_event(s.db_path, "answer_sent", vehicle_id=conv.vehicle_id, session_id=conv.session_id, data={
        "scenario": scenario, "chars": len(answer), "mode": conv.mode.mode, "tools": steps,
        "citations": [{"doc_id": c["doc_id"], "pdf_page": c["pdf_page"]} for c in citations],
        "dropped_refs": len(renderer.dropped), "prompt_leak_blocked": renderer.leaked, "llm_failed": failed})
    if not failed:
        conv.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
    yield {"type": "done", "text": answer, "citations": citations, "tools": steps, "mode": conv.mode.mode,
           "warnings": list(conv.mode.warnings), "llm_failed": failed}
