// Интерфейс механика: одна страница без сборки. Все строки — в словаре T (на будущее для языков).
// Текст от сервера вставляется только через textContent или после экранирования.
"use strict";

const T = {
  appTitle: "Цифровой механик",
  logout: "Выйти",
  searchPlaceholder: "Госномер, модель или двигатель",
  nothingFound: "Ничего не найдено",
  noManuals: "Для двигателя этой машины нет руководств в базе знаний — ответы со ссылками недоступны.",
  checkLabel: "Проверка",
  diagnose: "Получить результат диагностики",
  qCheck: "Что проверить?",
  qDrive: "Можно ехать своим ходом?",
  qTicket: "Создать заявку",
  qCheckText: "Что проверить в первую очередь?",
  qDriveText: "Можно ли своим ходом перегнать машину на базу?",
  qTicketText: "Подготовь заявку на дефектовку по этой проверке.",
  askPlaceholder: "Вопрос по машине или результату проверки",
  send: "Спросить",
  status: { normal: "норма", anomaly: "отклонение", check_failed: "сбой проверки" },
  noChecks: "проверок нет",
  earlierAnomaly: "{date}: было отклонение {code}",
  earlierTicket: "заявка {id}",
  earlierNoTicket: "заявки нет",
  earlierOpen: "Открыть проверку {id}",
  verdict: {
    normal: "Норма — значимых отклонений нет, эксплуатацию можно продолжать",
    anomaly: "Отклонение: {title}",
    anomaly_low_confidence: "Возможное отклонение: {title} — подтвердите ручной проверкой",
    check_failed: "Сбой проверки — запись непригодна, выводов о двигателе нет. Повторите запись",
    ml_inconsistent: "Результат ML противоречит устройству двигателя ({title}) — повторите запись или проверьте вручную",
  },
  score: "Выраженность отклонения",
  confidence: "уверенность",
  signal: { good: "сигнал хороший", warning: "помехи при записи", bad: "плохой сигнал" },
  priority: "Приоритет",
  priorityWhy: "почему",
  modelVersion: "Версия ML-модели",
  agentSteps: "Агент",
  thinking: "Агент работает",
  draftTitle: "Черновик заявки на дефектовку",
  dVehicle: "Машина и проверка", dFault: "Неисправность", dConclusion: "Вывод",
  dOperations: "Работы", dParts: "Запчасти", dRefs: "Источники", noParts: "не требуются по руководству",
  sendToir: "Отправить в ТОиР", cancel: "Отмена", sending: "Отправка…",
  ticketCreated: "Заявка создана: {id}", draftCancelled: "Черновик отменён",
  urgency: { low: "низкий", medium: "средний", high: "высокий" },
  errNetwork: "Нет связи с сервером — повторите",
  errGeneric: "Ошибка: {msg}",
  page: "{manual}, стр. {page}",
  trendTitle: "Выраженность отклонения по проверкам",
  trendLabel: "{label} · точек в расчёте: {n}",
  trendNotBuilt: "Оценка тенденции не строится: {reason}",
  trendThreshold: "порог карантина {t}",
  trendFailed: "сбой",
};

const $ = (id) => document.getElementById(id);
const fmt = (s, vars) => s.replace(/\{(\w+)\}/g, (_, k) => (vars[k] ?? ""));
const num = (x) => (x == null ? "—" : Number(x).toFixed(2).replace(".", ","));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const day = (iso) => new Date(iso).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });
// «31.10: было отклонение F12 · заявки нет» — отклонение, после которого пришли другие проверки
const earlierText = (e) => fmt(T.earlierAnomaly, { date: day(e.checked_at), code: e.anomaly_type || "" })
  + " · " + (e.ticket_id ? fmt(T.earlierTicket, { id: e.ticket_id }) : T.earlierNoTicket);
const when = (iso) => new Date(iso).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });

const state = { vehicles: [], vehicle: null, conversationId: null, busy: false };

// --- строки интерфейса из словаря ---------------------------------------------------------
document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = T[el.dataset.t]; });
document.querySelectorAll("[data-t-placeholder]").forEach((el) => { el.placeholder = T[el.dataset.tPlaceholder]; });

// --- сеть ----------------------------------------------------------------------------------
async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (res.status === 401) { location.href = "/login"; throw new Error("auth"); }
  return res;
}

async function errorText(res) {
  try { const body = await res.json(); return typeof body.detail === "string" ? body.detail : res.statusText; }
  catch { return res.statusText; }
}

// POST с ответом SSE: события «data: {json}» разделены пустой строкой.
async function streamPost(path, body, onEvent) {
  const res = await api(path, { method: "POST", body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await errorText(res));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const data = block.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

// --- экран парка ---------------------------------------------------------------------------
async function loadFleet() {
  const res = await api("/api/vehicles");
  if (!res.ok) { $("fleet-empty").textContent = await errorText(res); $("fleet-empty").hidden = false; return; }
  state.vehicles = await res.json();
  renderFleet();
}

function renderFleet() {
  const q = $("search").value.trim().toLowerCase();
  const list = state.vehicles.filter((v) =>
    !q || [v.vehicle_id, v.garage_no, v.make, v.model, v.engine_model].some((x) => (x || "").toLowerCase().includes(q)));
  const ul = $("vehicles");
  ul.replaceChildren(...list.map((v) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    const last = v.last_check;
    const status = last ? last.status : "";
    b.innerHTML = `<span class="dot ${esc(status)}"></span><span class="plate">${esc(v.vehicle_id)}</span>`
      + `<span class="st">${esc(last ? (T.status[status] || status) + (last.anomaly_type ? " " + last.anomaly_type : "") : T.noChecks)}</span>`
      + `<span class="sub">${esc([v.make, v.model].filter(Boolean).join(" "))} · ${esc(v.engine_model)}</span>`
      + (v.earlier_anomaly ? `<span class="earlier">${esc(earlierText(v.earlier_anomaly))}</span>` : "");
    b.onclick = () => { location.hash = "#/v/" + encodeURIComponent(v.vehicle_id); };
    li.append(b);
    return li;
  }));
  $("fleet-empty").textContent = T.nothingFound;
  $("fleet-empty").hidden = list.length > 0;
}

// --- экран машины --------------------------------------------------------------------------
async function openVehicle(id) {
  const res = await api("/api/vehicles/" + encodeURIComponent(id));
  if (!res.ok) { location.hash = ""; return; }
  const v = await res.json();
  state.vehicle = v;
  resetConversation();
  $("title").textContent = v.vehicle_id;
  $("v-plate").textContent = v.vehicle_id;
  $("v-model").textContent = [[v.make, v.model].filter(Boolean).join(" "), v.engine_model, v.fuel].filter(Boolean).join(" · ");
  $("v-no-manuals").hidden = v.manuals_available;
  const sel = $("session");
  sel.replaceChildren(...v.sessions.map((s) => {
    const o = document.createElement("option");
    o.value = s.session_id;
    o.textContent = `${when(s.checked_at)} — ${T.status[s.status] || s.status}${s.anomaly_type ? " " + s.anomaly_type : ""} (${s.session_id})`;
    return o;
  }));
  $("diagnose").disabled = v.sessions.length === 0;
  const ea = v.earlier_anomaly, eb = $("v-earlier");
  eb.hidden = !ea;
  if (ea) {
    eb.innerHTML = `<span>${esc(earlierText(ea))}${ea.anomaly_title ? " — " + esc(ea.anomaly_title) : ""}</span>`
      + `<span class="go">${esc(fmt(T.earlierOpen, { id: ea.session_id }))} →</span>`;
    eb.onclick = () => {
      if (sel.value !== ea.session_id) { sel.value = ea.session_id; resetConversation(); loadTrend(); }
      $("diagnose").focus();
    };
  }
  show("vehicle");
  loadTrend();
}

// --- оценка тенденции (B2): считает сервер (policy/trend), здесь только рисуем ----------------
async function loadTrend() {
  const v = state.vehicle, sid = $("session").value;
  $("trend").hidden = true;
  if (!v || !sid) return;
  const res = await api(`/api/vehicles/${encodeURIComponent(v.vehicle_id)}/trend?session_id=${encodeURIComponent(sid)}`);
  if (!res.ok || state.vehicle !== v || $("session").value !== sid) return;   // пока ждали, выбрали другое
  renderTrend(await res.json());
}

function renderTrend(t) {
  if (!t.points.length) return;
  $("trend-chart").innerHTML = trendSvg(t);
  $("trend-summary").textContent = t.built ? t.summary : fmt(T.trendNotBuilt, { reason: t.reason });
  $("trend-summary").className = "trend-summary" + (t.built && (t.threshold_reached || t.days_to_threshold != null) ? " hot" : "");
  $("trend-label").textContent = fmt(T.trendLabel, { label: t.label, n: t.n_points });
  // главная оговорка («это не дата поломки») — всегда на виду, остальное — под «подробнее»
  $("trend-caveat").textContent = t.built ? t.caveats[0] : "";
  $("trend-caveat").hidden = !t.built;
  const notes = t.built ? t.caveats.slice(1) : t.points.filter((p) => p.note).map((p) => `${day(p.checked_at)}: ${p.note}.`);
  $("trend-caveats").replaceChildren(...notes.map((c) => { const li = document.createElement("li"); li.textContent = c; return li; }));
  $("trend-more").hidden = notes.length === 0 && !t.built;
  $("trend").hidden = false;
}

// SVG: ось X — даты (по времени), Y — 0…1. Точки в расчёте соединены линией; разрывы ряда — бледнее.
function trendSvg(t) {
  const W = 320, H = 150, L = 30, R = 10, TOP = 10, B = 26;
  const pts = t.points, ts = pts.map((p) => new Date(p.checked_at).getTime());
  const t0 = Math.min(...ts), span = Math.max(...ts) - t0 || 1;
  const x = (ms) => pts.length === 1 ? (L + W - R) / 2 : L + (ms - t0) / span * (W - L - R);
  const y = (v) => TOP + (1 - v) * (H - TOP - B);
  let g = "";
  for (const v of [0, 0.5, 1]) g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="ax" x="${L - 4}" y="${y(v) + 4}" text-anchor="end">${num(v).replace(",00", "")}</text>`;
  g += `<line class="thr" x1="${L}" x2="${W - R}" y1="${y(t.threshold)}" y2="${y(t.threshold)}"/>`
    + `<text class="thr-t" x="${W - R}" y="${y(t.threshold) - 4}" text-anchor="end">${esc(fmt(T.trendThreshold, { t: num(t.threshold) }))}</text>`;
  const used = pts.map((p, i) => [p, ts[i]]).filter(([p]) => p.used && p.anomaly_score != null);
  if (used.length > 1) g += `<polyline class="line" points="${used.map(([p, ms]) => `${x(ms)},${y(p.anomaly_score)}`).join(" ")}"/>`;
  pts.forEach((p, i) => {
    const cx = x(ts[i]), cls = `pt ${esc(p.status)}${p.used ? "" : " off"}`;
    g += p.anomaly_score == null
      ? `<text class="${cls}" x="${cx}" y="${y(0) - 3}" text-anchor="middle">×</text>`
      : `<circle class="${cls}" cx="${cx}" cy="${y(p.anomaly_score)}" r="5"><title>${esc(day(p.checked_at))}: ${num(p.anomaly_score)}</title></circle>`;
    g += `<text class="ax" x="${cx}" y="${H - 8}" text-anchor="middle">${esc(day(p.checked_at))}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(T.trendTitle)}">${g}</svg>`;
}

function resetConversation() {
  state.conversationId = null;
  $("verdict").hidden = true;
  $("verdict").replaceChildren();
  $("chat").replaceChildren();
}

function show(view) {
  $("fleet-view").hidden = view !== "fleet";
  $("vehicle-view").hidden = view !== "vehicle";
  $("back").hidden = view === "fleet";
  if (view === "fleet") $("title").textContent = T.appTitle;
  window.scrollTo(0, 0);
}

function route() {
  const m = location.hash.match(/^#\/v\/(.+)$/);
  if (m) openVehicle(decodeURIComponent(m[1]));
  else { state.vehicle = null; show("fleet"); }
}

// --- вердикт -------------------------------------------------------------------------------
function renderVerdict(d) {
  const card = $("verdict");
  card.className = "verdict " + d.mode;
  const title = d.anomaly_title ? `${d.anomaly_title} (${d.anomaly_type})` : (d.anomaly_type || "");
  const headline = fmt(T.verdict[d.mode] || T.verdict.anomaly, { title });
  const metrics = d.mode === "check_failed" ? T.signal[d.signal_quality] || ""
    : `${T.score} ${num(d.anomaly_score)} · ${T.confidence} ${num(d.confidence)} · ${T.signal[d.signal_quality] || d.signal_quality}`;
  let html = `<p class="headline">${esc(headline)}</p><p class="metrics">${esc(metrics)}</p>`;
  if (d.warnings_ru && d.warnings_ru.length) {
    html += `<div class="badges">${d.warnings_ru.map((w) => `<span class="badge">⚠ ${esc(w)}</span>`).join("")}</div>`;
  }
  if (d.urgency) {
    html += `<details class="urgency"><summary>${esc(T.priority)}: <span class="lvl-${esc(d.urgency.level)}">${esc(d.urgency.level_ru)}</span> — ${esc(T.priorityWhy)}</summary>`
      + `<ul>${d.urgency.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></details>`;
  }
  html += `<div class="version">${esc(T.modelVersion)}: ${esc(d.model_version)}</div>`;
  card.innerHTML = html;
  card.hidden = false;
}

// --- чат -----------------------------------------------------------------------------------
function addMessage(kind, text) {
  const div = document.createElement("div");
  div.className = "msg " + kind;
  div.textContent = text || "";
  $("chat").append(div);
  div.scrollIntoView({ block: "nearest" });
  return div;
}

// Ответ агента: экранирование → **жирный** → подписи ссылок от кода в чипы (только те,
// что пришли в списке citations этого хода; остальное остаётся текстом).
function renderAnswer(div, text, citations) {
  const byLabel = new Map(citations.map((c) => [c.label, c]));
  let html = esc(text).replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");
  html = html.replace(/\[([^\[\]\n]+)\]/g, (whole, inner) => {
    const parts = inner.split("; ");
    const found = parts.map((p) => byLabel.get(unesc(p)));
    if (found.some((c) => !c)) return whole;
    return found.map((c) => chipHtml(c)).join(" ");
  });
  div.innerHTML = html;
  bindChips(div);
}

function unesc(s) {
  return s.replace(/&(amp|lt|gt|quot|#39);/g, (_, e) => ({ amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'" }[e]));
}

function chipHtml(c) {
  return `<button type="button" class="chip" data-doc="${esc(c.doc_id)}" data-page="${esc(c.pdf_page)}" data-manual="${esc(c.manual)}">`
    + `${esc(fmt(T.page, { manual: c.manual, page: c.pdf_page }))}</button>`;
}

function bindChips(root) {
  root.querySelectorAll(".chip").forEach((b) => {
    b.onclick = () => openPage(b.dataset.doc, b.dataset.page, b.dataset.manual);
  });
}

function openPage(docId, page, manual) {
  $("page-title").textContent = fmt(T.page, { manual, page });
  $("page-img").src = `/api/manuals/${encodeURIComponent(docId)}/pages/${encodeURIComponent(page)}.png`;
  $("page-img").alt = $("page-title").textContent;
  $("page-dialog").showModal();
}

function addSteps(div, steps) {
  if (!steps.length) return;
  const d = document.createElement("details");
  d.className = "steps";
  const s = document.createElement("summary");
  s.textContent = `${T.agentSteps}: ${steps.join(" → ")}`;
  d.append(s);
  div.append(d);
}

// --- черновик заявки -----------------------------------------------------------------------
function renderDraft(ev) {
  const t = ev.ticket;
  const card = document.createElement("div");
  card.className = "draft";
  const sources = (ev.sources || []).map((c) => chipHtml(c)).join(" ");
  card.innerHTML = `<h3>${esc(T.draftTitle)}</h3><dl>`
    + `<dt>${esc(T.dVehicle)}</dt><dd>${esc(t.vehicle_id)} · ${esc(t.session_id)}</dd>`
    + `<dt>${esc(T.dFault)}</dt><dd>${esc(t.anomaly_type)}</dd>`
    + `<dt>${esc(T.priority)}</dt><dd><details class="urgency"><summary><span class="lvl-${esc(t.urgency)}">${esc(T.urgency[t.urgency] || t.urgency)}</span> — ${esc(T.priorityWhy)}</summary>`
    + `<ul>${(ev.urgency_reasons || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul></details></dd>`
    + `<dt>${esc(T.dConclusion)}</dt><dd>${esc(t.conclusion)}</dd>`
    + `<dt>${esc(T.dOperations)}</dt><dd><ol>${t.operations.map((o) => `<li>${esc(o)}</li>`).join("")}</ol></dd>`
    + `<dt>${esc(T.dParts)}</dt><dd>${t.spare_parts.length ? `<ul>${t.spare_parts.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : esc(T.noParts)}</dd>`
    + `<dt>${esc(T.dRefs)}</dt><dd>${sources}</dd></dl>`
    + `<div class="actions"><button class="primary send">${esc(T.sendToir)}</button><button class="secondary cancel">${esc(T.cancel)}</button></div>`
    + `<div class="result" hidden></div>`;
  bindChips(card);
  const sendBtn = card.querySelector(".send");
  const cancelBtn = card.querySelector(".cancel");
  const result = card.querySelector(".result");
  const finish = (text, ok) => {
    result.textContent = text; result.className = "result " + (ok ? "ok" : "bad"); result.hidden = false;
  };
  sendBtn.onclick = async () => {
    sendBtn.disabled = cancelBtn.disabled = true;
    sendBtn.textContent = T.sending;
    try {
      // подтверждение привязано к id и хэшу ровно этого показанного черновика
      const res = await api(`/api/tickets/${encodeURIComponent(ev.draft_id)}/confirm`,
        { method: "POST", body: JSON.stringify({ hash: ev.content_hash }) });
      if (res.ok) {
        const body = await res.json();
        finish(fmt(T.ticketCreated, { id: body.ticket_id }), true);
        card.querySelector(".actions").hidden = true;
      } else {
        finish(await errorText(res), false);
        sendBtn.textContent = T.sendToir;
        sendBtn.disabled = cancelBtn.disabled = res.status === 409;   // 409 — повтор бессмыслен
      }
    } catch (e) {
      finish(T.errNetwork, false);
      sendBtn.textContent = T.sendToir;
      sendBtn.disabled = cancelBtn.disabled = false;
    }
  };
  cancelBtn.onclick = async () => {
    sendBtn.disabled = cancelBtn.disabled = true;
    try {
      const res = await api(`/api/tickets/${encodeURIComponent(ev.draft_id)}/cancel`, { method: "POST" });
      finish(res.ok ? T.draftCancelled : await errorText(res), res.ok);
      card.querySelector(".actions").hidden = true;
    } catch (e) {
      finish(T.errNetwork, false);
      sendBtn.disabled = cancelBtn.disabled = false;
    }
  };
  $("chat").append(card);
  card.scrollIntoView({ block: "nearest" });
}

// --- ход агента ----------------------------------------------------------------------------
async function turn(path, body, userText) {
  if (state.busy) return;
  state.busy = true;
  setBusy(true);
  if (userText) addMessage("user", userText);
  const answer = addMessage("agent pending", "");
  let text = "";
  try {
    await streamPost(path, body, (ev) => {
      if (ev.type === "conversation") state.conversationId = ev.conversation_id;
      else if (ev.type === "diagnosis") renderVerdict(ev);
      else if (ev.type === "draft") renderDraft(ev);
      else if (ev.type === "trend") renderTrend(ev);
      else if (ev.type === "delta") { text += ev.text; answer.textContent = text; }
      else if (ev.type === "done") {
        answer.classList.remove("pending");
        renderAnswer(answer, ev.text, ev.citations || []);
        addSteps(answer, ev.tools || []);
        $("chat").append(answer);               // ответ — после карточек этого хода
      } else if (ev.type === "error") {
        answer.className = "msg error"; answer.textContent = ev.message;
      }
    });
  } catch (e) {
    answer.className = "msg error";
    answer.textContent = e.message === "auth" ? "" : fmt(T.errGeneric, { msg: e.message || T.errNetwork });
  } finally {
    answer.classList.remove("pending");
    state.busy = false;
    setBusy(false);
  }
}

function setBusy(on) {
  $("diagnose").disabled = on || !state.vehicle || state.vehicle.sessions.length === 0;
  document.querySelectorAll("#quick button, #ask button").forEach((b) => { b.disabled = on; });
}

function ask(text, scenario = "question") {
  text = text.trim();
  if (!text || !state.vehicle) return;
  const body = state.conversationId
    ? { conversation_id: state.conversationId, message: text, scenario }
    : { vehicle_id: state.vehicle.vehicle_id, session_id: $("session").value, message: text, scenario };
  turn("/api/chat", body, text);
}

// --- обработчики ---------------------------------------------------------------------------
$("search").addEventListener("input", renderFleet);
$("back").onclick = () => { location.hash = ""; };
$("session").onchange = () => { resetConversation(); loadTrend(); };   // другая проверка — новый диалог
$("diagnose").onclick = () => {
  resetConversation();
  turn("/api/diagnose", { vehicle_id: state.vehicle.vehicle_id, session_id: $("session").value }, T.diagnose);
};
$("quick").addEventListener("click", (e) => {
  const kind = e.target.dataset && e.target.dataset.quick;
  if (kind === "check") ask(T.qCheckText);
  if (kind === "drive") ask(T.qDriveText);
  if (kind === "ticket") ask(T.qTicketText, "ticket");
});
$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("message").value;
  $("message").value = "";
  ask(text);
});
$("message").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("ask").requestSubmit(); }
});
$("page-close").onclick = () => $("page-dialog").close();
$("page-dialog").addEventListener("click", (e) => { if (e.target === $("page-dialog")) $("page-dialog").close(); });
window.addEventListener("hashchange", route);

loadFleet();
route();
