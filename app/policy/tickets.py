"""Черновик заявки и её отправка по подтверждению механика (правила 7–9 CLAUDE.md).

create_draft — то, что вызывает инструмент агента draft_ticket: собирает заявку из
предложения модели и данных кода, прогоняет валидатор, считает приоритет, сохраняет
черновик с id и хэшем содержимого. В ТОиР ничего не отправляет.

confirm_draft — то, что вызывает кнопка «Отправить в ТОиР»: отправляет ровно тот черновик,
который видел механик (совпадают id и хэш), один раз на проверку. Модель этот путь
не вызывает: у агента нет такого инструмента.
"""
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.integrations.diagnosis import CheckedDiagnosis
from app.integrations.errors import BadRequest, NotDelivered, NotFound, ServiceUnavailable
from app.integrations.models import Reference, TicketRequest
from app.integrations.toir_client import TicketService, submit_ticket
from app.policy.citations import CitationRegistry
from app.policy.settings import load_policy
from app.policy.ticket_validator import DraftInput, validate_draft
from app.policy.urgency import compute_urgency
from app.storage import tickets as store
from app.storage.events import log_event


class TicketError(Exception):
    """Подтверждение отклонено; текст — для механика."""


@dataclass(frozen=True)
class TicketDraft:
    draft_id: str
    content_hash: str
    ticket: TicketRequest
    urgency_reasons: tuple[str, ...]
    sources: tuple[dict, ...]           # chunk_id → документ и страница, для чипов интерфейса


@dataclass
class DraftOutcome:
    draft: TicketDraft | None = None
    violations: list[str] = field(default_factory=list)


def content_hash(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_draft(proposal: DraftInput, checked: CheckedDiagnosis, registry: CitationRegistry, *,
                 db_path: Path, config_dir: Path, fault_titles: dict[str, str] | None = None) -> DraftOutcome:
    diagnosis, vehicle = checked.diagnosis, checked.vehicle
    vid, sid = vehicle.vehicle_id, diagnosis.session_id
    cfg = load_policy(Path(config_dir))["ticket"]

    result = validate_draft(
        proposal, diagnosis=diagnosis, applicability=checked.applicability, vehicle_id=vid,
        vehicle_sessions={s.session_id for s in vehicle.sessions}, engine_family=vehicle.engine_family,
        registry=registry, part_number_min_len=cfg["part_number_min_len"])
    existing = store.get_ticket(db_path, vid, sid)
    if existing is not None:
        result.violations.append(
            "По этой проверке уже отправлялась заявка, ответ ТОиР не получен — сверьте с ТОиР"
            if existing["status"] == "unknown" else
            "По этой проверке заявка уже создана" + (f": {existing['ticket_id']}" if existing.get("ticket_id") else ""))
    if not result.ok:
        log_event(db_path, "ticket_rejected", vehicle_id=vid, session_id=sid,
                  data={"violations": result.violations, "chunk_ids": proposal.chunk_ids[:20]})
        return DraftOutcome(violations=result.violations)

    current = vehicle.session(sid)
    previous = [s for s in vehicle.sessions if current is not None and s.checked_at < current.checked_at]
    urgency = compute_urgency(diagnosis, previous, config_dir, fault_titles)

    references, seen = [], set()
    for c in result.citations:                       # одна страница документа — одна ссылка
        if (c.manual, c.pdf_page) not in seen:
            seen.add((c.manual, c.pdf_page))
            references.append(Reference(manual=c.manual, pdf_page=c.pdf_page))
    ticket = TicketRequest(
        vehicle_id=vid, session_id=sid, anomaly_type=diagnosis.anomaly_type,
        conclusion=proposal.conclusion.strip(), urgency=urgency.level,
        operations=[op.strip() for op in proposal.operations],
        spare_parts=result.spare_parts,           # номер — в написании из фрагмента руководства
        references=references)
    payload = ticket.model_dump(mode="json")
    digest = content_hash(payload)
    draft_id = uuid.uuid4().hex
    sources = tuple({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "manual": c.manual, "pdf_page": c.pdf_page}
                    for c in result.citations)
    store.insert_draft(db_path, draft_id=draft_id, vehicle_id=vid, session_id=sid, payload=payload,
                       content_hash=digest,
                       extra={"urgency_reasons": list(urgency.reasons), "sources": list(sources)})
    log_event(db_path, "ticket_draft", vehicle_id=vid, session_id=sid, data={
        "draft_id": draft_id, "anomaly_type": ticket.anomaly_type, "urgency": ticket.urgency,
        "operations": len(ticket.operations), "spare_parts": len(ticket.spare_parts),
        "references": [r.model_dump() for r in references]})
    return DraftOutcome(draft=TicketDraft(draft_id=draft_id, content_hash=digest, ticket=ticket,
                                          urgency_reasons=urgency.reasons, sources=sources))


async def confirm_draft(draft_id: str, shown_hash: str, service: TicketService, *, db_path: Path) -> str:
    """Отправка черновика по кнопке механика. Возвращает ticket_id или TicketError."""
    row = store.get_draft(db_path, draft_id)
    if row is None:
        raise TicketError("Черновик не найден — подготовьте заявку заново")
    vid, sid = row["vehicle_id"], row["session_id"]

    def reject(message: str):
        log_event(db_path, "ticket_rejected", vehicle_id=vid, session_id=sid,
                  data={"draft_id": draft_id, "violations": [message], "stage": "confirm"})
        raise TicketError(message)

    if row["status"] == "cancelled":
        reject("Черновик отменён")
    if row["status"] == "sent":
        reject("Этот черновик уже отправлен")
    # Хэш, показанный механику, должен совпасть и с сохранённым, и с пересчитанным по
    # содержимому: отправляется ровно то, что механик видел на экране.
    if shown_hash != row["content_hash"] or content_hash(row["payload"]) != row["content_hash"]:
        reject("Черновик изменился после показа — проверьте заявку ещё раз")
    ticket = TicketRequest.model_validate(row["payload"])
    if (ticket.vehicle_id, ticket.session_id) != (vid, sid):
        reject("Черновик повреждён: машина или проверка не совпадают")
    if not store.reserve_ticket(db_path, vid, sid, draft_id):
        existing = store.get_ticket(db_path, vid, sid) or {}
        reject("По этой проверке заявка уже создана"
               + (f": {existing['ticket_id']}" if existing.get("ticket_id") else " или отправляется"))

    log_event(db_path, "ticket_confirmed", vehicle_id=vid, session_id=sid,
              data={"draft_id": draft_id, "content_hash": row["content_hash"][:16]})
    try:
        ticket_id = await submit_ticket(service, ticket, db_path)
    except (NotDelivered, NotFound, BadRequest) as exc:
        store.release_ticket(db_path, vid, sid)          # заявка точно не создана — можно повторить
        raise exc
    except ServiceUnavailable as exc:
        # Сервис сам ответил ошибкой — заявка не создана. 502/504 приходят от прокси:
        # до ТОиР запрос мог дойти, исход неизвестен.
        if exc.status_code is not None and exc.status_code not in (502, 504):
            store.release_ticket(db_path, vid, sid)
            raise
        store.mark_ticket_unknown(db_path, vid, sid)
        raise TicketError("ТОиР не ответил — заявка могла быть создана. Проверьте в ТОиР, "
                          "повторная отправка по этой проверке заблокирована") from exc
    except BaseException:
        # таймаут без ответа, некорректный ответ, отмена запроса — исход неизвестен
        store.mark_ticket_unknown(db_path, vid, sid)
        raise
    store.complete_ticket(db_path, vid, sid, ticket_id)
    store.set_draft_status(db_path, draft_id, "sent")
    return ticket_id


def cancel_draft(draft_id: str, *, db_path: Path) -> bool:
    return store.set_draft_status(db_path, draft_id, "cancelled")
