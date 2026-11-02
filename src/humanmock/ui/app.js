// The page. It draws everything from what the server declares in /v1/catalog, so a new
// organ, drug, illness, or patient parameter appears here without any change to this file.

import { ApiError, api } from "./api.js";
import { partLabel, partOrder, drawBody } from "./body.js";
import { domainFor, drawAxis, drawDoseRow, drawStrip, timeTicks, timeToX } from "./chart.js";

const $ = (id) => document.getElementById(id);

const SHORT_UNITS = {
  "degrees C": "°C",
  "beats per minute": "beats per minute",
  "breaths per minute": "breaths per minute",
  "score 0 to 10": "out of 10",
  "index 0 to 1": "out of 1",
  "mg per litre": "mg/L",
  "thousand cells per microlitre": "thousand per µL",
  "log10 colony forming units": "log10 colonies",
};

// What each hypothetical drug field means. Fields not listed here keep their defaults.
const DRUG_FIELDS = [
  ["ka", "Absorption rate, per hour", 1],
  ["v_l_per_kg", "Volume of distribution, L/kg", 1],
  ["cl_ref_l_h", "Clearance at 70 kg, L/h", 10],
  ["kill_emax", "Strongest germ killing, per hour", 0],
  ["kill_ec50", "Level for half that killing, mg/L", 1],
  ["reservoir_penetration", "Reach into hidden germs, 0 to 1", 0],
  ["antipyretic_emax", "Strongest fever reduction, 0 to 1", 0],
  ["analgesic_emax", "Strongest pain relief, 0 to 1", 0],
];

const MINIMUM_CHART_HOURS = 24;
const MAX_STEPS_PER_MOVE = 400;
const NOTHING = { p10: 0, p50: 0, p90: 0 };

const state = {
  catalog: null,
  model: null,
  sessionId: null,
  session: null,
  history: [],
  queue: [],
  saved: [],
  compareId: "",
  compareHistory: [],
  shown: new Set(),
  notices: [],
  outsideTraining: [],
  hover: null,
  selected: null,
  running: false,
  stopping: false,
  frame: null,
  valueNodes: new Map(),
};

// ---- Small helpers ----

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else if (value === true) node.setAttribute(name, "");
    else if (value !== false && value !== undefined && value !== null) node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

function remember(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Remembering is only a convenience, so the page carries on without it.
  }
}

function recall(key) {
  try {
    return JSON.parse(localStorage.getItem(key));
  } catch {
    return null;
  }
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function elapsed(hours) {
  if (hours < 1e-6) return "the start";
  const rounded = Math.round(hours * 4) / 4;
  const days = Math.floor(rounded / 24);
  const rest = rounded - days * 24;
  const parts = [];
  if (days) parts.push(plural(days, "day"));
  if (rest >= 1 || (rest > 0 && !days)) {
    parts.push(rest === Math.floor(rest) ? plural(rest, "hour") : `${rest} hours`);
  }
  return parts.join(" ") || "the start";
}

function capitalise(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function drugLabel(name) {
  const known = state.catalog.drugs[name];
  return known ? known.label : name;
}

function illnessLabel(name) {
  return state.catalog.conditions[name] || name;
}

function now() {
  const last = state.history[state.history.length - 1];
  return last ? last.time : 0;
}

function decimalsFor(noise) {
  if (noise >= 5) return 0;
  if (noise >= 0.2) return 1;
  return 2;
}

function describePerson(patient = {}, conditions = []) {
  const who = [];
  if (patient.sex) who.push(capitalise(patient.sex === "female" ? "woman" : "man"));
  if (patient.age_years) who.push(`${who.length ? "" : "Age "}${patient.age_years}${who.length ? " years" : ""}`);
  if (patient.weight_kg) who.push(`${patient.weight_kg} kg`);
  const person = who.length ? who.join(", ") : "Someone not described";
  const illness = conditions.length
    ? conditions.map((c) => illnessLabel(c.name)).join(" and ")
    : "not ill";
  return `${person}, ${illness}`;
}

function showNotice(message, action) {
  const text = capitalise(message);
  if (!state.notices.some((n) => n.text === text)) state.notices.push({ text, action });
  renderNotices();
}

function clearNotices() {
  state.notices = [];
  renderNotices();
}

function fail(error) {
  if (error instanceof ApiError && error.status === 409) {
    showNotice(
      "This session was made with a different model, so it cannot continue as it is. Drawing new virtual people keeps the description and forgets what happened so far.",
      { label: "Draw new virtual people", run: redrawPeople },
    );
  } else {
    showNotice(error.message || String(error));
  }
}

// A button that forgets something asks once more before acting, without a browser popup.
function askFirst(button, question, action) {
  const original = button.textContent;
  let armed = false;
  let timer = null;
  button.addEventListener("click", async () => {
    if (!armed) {
      armed = true;
      button.textContent = question;
      timer = setTimeout(() => {
        armed = false;
        button.textContent = original;
      }, 4000);
      return;
    }
    clearTimeout(timer);
    armed = false;
    button.textContent = original;
    await action();
  });
}

// ---- Setting up a person ----

function buildSetupForm() {
  const { conditions, patient } = state.catalog;
  const illness = $("setup-illness");
  illness.replaceChildren(
    ...Object.entries(conditions).map(([name, label]) => el("option", { value: name }, label)),
    el("option", { value: "" }, "None, a healthy person"),
  );

  const also = $("setup-also");
  for (const [name, label] of Object.entries(conditions)) {
    also.append(el("label", { class: "check", "data-also-row": name }, el("input", { type: "checkbox", value: name, "data-also": true }), label));
  }
  // The illness already chosen above is not offered again.
  const offerOthers = () => {
    for (const row of document.querySelectorAll("[data-also-row]")) {
      const chosen = row.dataset.alsoRow === illness.value;
      row.hidden = chosen;
      if (chosen) row.querySelector("input").checked = false;
    }
  };
  illness.addEventListener("change", offerOthers);
  offerOthers();

  drawBody($("welcome-body"), {
    parts: new Map([["throat", { heat: 0.85 }], ["skin", { heat: 0.5 }], ["lungs", { heat: 0.15 }]]),
    germs: 0.7,
    drug: 0.6,
  });

  const measures = $("setup-measures");
  for (const [name, entry] of Object.entries(patient)) {
    if (name === "age_years" || name === "weight_kg") continue;
    measures.append(
      el("label", { class: "field" }, entry.note,
        el("input", { type: "number", step: "any", min: entry.min, max: entry.max, "data-measure": name, placeholder: "Not sure" })),
    );
  }

  const drug = $("setup-drug");
  for (const [name, label, usual] of DRUG_FIELDS) {
    if (!state.catalog.custom_drug_parameters.includes(name)) continue;
    drug.append(el("label", { class: "field" }, label,
      el("input", { type: "number", step: "any", min: 0, "data-drug": name, placeholder: String(usual) })));
  }
}

function readNumber(input) {
  const value = input.value.trim();
  return value === "" ? undefined : Number(value);
}

function readSetup() {
  const patient = {};
  const age = readNumber($("setup-age"));
  const weight = readNumber($("setup-weight"));
  if (age !== undefined) patient.age_years = age;
  if (weight !== undefined) patient.weight_kg = weight;
  if ($("setup-sex").value) patient.sex = $("setup-sex").value;
  for (const input of document.querySelectorAll("[data-measure]")) {
    const value = readNumber(input);
    if (value !== undefined) patient[input.dataset.measure] = value;
  }
  const allergies = $("setup-allergies").value.split(",").map((a) => a.trim()).filter(Boolean);
  if (allergies.length) patient.allergies = allergies;

  const onset = readNumber($("setup-onset")) ?? 1;
  const names = new Set();
  if ($("setup-illness").value) names.add($("setup-illness").value);
  for (const box of document.querySelectorAll("[data-also]:checked")) names.add(box.value);
  const conditions = [...names].map((name) => ({ name, days_since_symptom_onset: onset }));

  const body = { patient, conditions, people: readNumber($("setup-people")) ?? 64, include_internal: true };
  const seed = readNumber($("setup-seed"));
  if (seed !== undefined) body.seed = seed;

  const drugName = $("drug-name").value.trim();
  if (drugName) {
    const drug = { name: drugName };
    for (const input of document.querySelectorAll("[data-drug]")) {
      const value = readNumber(input);
      if (value !== undefined) drug[input.dataset.drug] = value;
    }
    body.custom_drugs = [drug];
  }
  return body;
}

async function startSession(body) {
  const button = $("setup-start");
  const error = $("setup-error");
  error.hidden = true;
  button.disabled = true;
  button.textContent = "Starting";
  try {
    const created = await api("v1/sessions", { method: "POST", body });
    await openSession(created.session_id);
  } catch (problem) {
    error.textContent = problem.message;
    error.hidden = false;
    $("setup").scrollIntoView({ block: "nearest" });
  } finally {
    button.disabled = false;
    button.textContent = "Start";
  }
}

function showSetup() {
  state.sessionId = null;
  state.session = null;
  state.history = [];
  state.queue = [];
  state.compareHistory = [];
  state.compareId = "";
  state.outsideTraining = [];
  clearNotices();
  history.replaceState(null, "", location.pathname + location.search);
  $("setup").hidden = false;
  $("controls").hidden = true;
  $("welcome").hidden = false;
  $("chart").hidden = true;
  $("person-line").textContent = "Research simulator of the human body";
  renderSaved();
}

// ---- Opening a session ----

function normalise(row) {
  const start = row.time_hours - (row.request.duration_hours || 0);
  return {
    step: row.step,
    time: row.time_hours,
    result: row.result,
    events: (row.request.events || []).map((event) => ({ ...event, time: start + event.at_hours })),
  };
}

async function loadSaved() {
  try {
    state.saved = (await api("v1/sessions")).sessions;
  } catch {
    state.saved = [];
  }
  renderSaved();
}

async function openSession(sessionId) {
  const [session, past] = await Promise.all([
    api(`v1/sessions/${sessionId}?include_internal=true`).catch((problem) => {
      if (problem.status === 409) return { mismatch: problem };
      throw problem;
    }),
    api(`v1/sessions/${sessionId}/history`),
  ]);

  state.sessionId = sessionId;
  state.session = session;
  state.history = past.steps.map(normalise);
  state.queue = recall(`queue:${sessionId}`) || [];
  state.hover = null;
  state.selected = null;
  state.notices = [];
  state.outsideTraining = [];
  history.replaceState(null, "", `#${sessionId}`);

  await loadSaved();
  if (session.mismatch) {
    const original = state.saved.find((s) => s.session_id === sessionId);
    state.session = { patient: original ? original.patient : {}, conditions: original ? original.conditions : [], custom_drugs: [] };
    fail(session.mismatch);
  }

  $("setup").hidden = true;
  $("controls").hidden = false;
  $("welcome").hidden = true;
  $("chart").hidden = false;

  buildGiveForm();
  renderFilledIn();
  await chooseComparison(session.forked_from ? session.forked_from.session_id : "");
  renderSaved();
  renderCompareRow();
  renderShownPicker();
  renderNotices();
  renderWaiting();
  drawNow();
}

function buildGiveForm() {
  const custom = (state.session.custom_drugs || []).map((d) => d.name);
  const names = [...Object.keys(state.catalog.drugs), ...custom];
  $("give-drug").replaceChildren(...names.map((name) => el("option", { value: name }, drugLabel(name))));
  fillUsualDose();
  $("expose-illness").replaceChildren(
    ...Object.entries(state.catalog.conditions).map(([name, label]) => el("option", { value: name }, label)),
  );
  const line = describePerson(state.session.patient, state.session.conditions);
  $("person-line").textContent = line;
}

// The usual dose is half the warning level, which suits each listed drug without naming any.
function fillUsualDose() {
  const known = state.catalog.drugs[$("give-drug").value];
  const limit = known && known.max_single_dose_mg;
  $("give-dose").value = limit ? limit / 2 : 500;
}

function renderFilledIn() {
  const box = $("filled-in");
  box.replaceChildren();
  const filled = state.session.imputed_parameters;
  if (!filled || !Object.keys(filled).length) return;
  const rows = Object.entries(filled).filter(([name]) => state.catalog.patient[name]);
  box.append(
    el("p", { class: "hint" }, "Filled in from similar people, as an average across the virtual people:"),
    el("ul", { class: "saved" }, ...rows.map(([name, value]) =>
      el("li", {}, `${state.catalog.patient[name].note}: ${value.mean}`))),
  );
}

// ---- Giving medicine ----

function saveQueue() {
  if (state.sessionId) remember(`queue:${state.sessionId}`, state.queue);
}

function giveDose(event) {
  event.preventDefault();
  const drug = $("give-drug").value;
  const dose = Number($("give-dose").value);
  if (!(dose > 0)) return;
  const every = Number($("give-every").value);
  const count = every ? Math.max(Number($("give-count").value) || 1, 1) : 1;
  const route = $("give-route").value;
  const start = now();
  for (let k = 0; k < count; k += 1) {
    state.queue.push({ type: "dose", drug, dose_mg: dose, route, at: start + k * every });
  }
  saveQueue();
  renderWaiting();
  drawNow();
}

function expose() {
  state.queue.push({ type: "exposure", pathogen: $("expose-illness").value, at: now() });
  saveQueue();
  renderWaiting();
  drawNow();
}

function renderWaiting() {
  const box = $("waiting");
  if (!state.queue.length) {
    box.hidden = true;
    return;
  }
  const next = Math.min(...state.queue.map((q) => q.at));
  const doses = state.queue.filter((q) => q.type === "dose");
  const count = doses.length;
  const summary = count
    ? `${plural(count, "dose")} waiting. They take effect as time moves forward.`
    : "An exposure is waiting. It takes effect as time moves forward.";
  const when = next <= now() + 1e-6 ? "The next one happens straight away." : `The next one is at ${elapsed(next)}.`;
  box.hidden = false;
  box.replaceChildren(
    el("div", {}, `${summary} ${when}`),
    el("button", { type: "button", onclick: () => { state.queue = []; saveQueue(); renderWaiting(); drawNow(); } }, "Cancel what is waiting"),
  );
}

// ---- Moving time forward ----

// Long moves are split into small steps, which is what draws a real curve. A dose always
// starts a step, and another step falls an hour later so each peak shows on the chart.
function planSteps(start, end) {
  const length = end - start;
  const size = length <= 2 ? 0.25 : length <= 24 ? 0.5 : length <= 72 ? 1 : length <= 336 ? 2 : Math.max(4, length / MAX_STEPS_PER_MOVE);
  const cuts = new Set([start]);
  for (let t = start; t < end - 1e-6; t += size) cuts.add(+t.toFixed(4));
  for (const item of state.queue) {
    if (item.at >= start - 1e-6 && item.at < end - 1e-6) {
      cuts.add(+item.at.toFixed(4));
      if (item.type === "dose" && item.at + 1 < end) cuts.add(+(item.at + 1).toFixed(4));
    }
  }
  const sorted = [...cuts].sort((a, b) => a - b);
  return sorted.map((from, i) => [from, i + 1 < sorted.length ? sorted[i + 1] : end]);
}

function setRunning(running) {
  state.running = running;
  $("stop-button").hidden = !running;
  $("time-buttons").hidden = running;
  for (const id of ["give-button", "expose-button", "restart-button", "new-person-button", "redraw-button", "moment-fork", "moment-back"]) {
    $(id).disabled = running;
  }
}

async function moveTime(hours) {
  if (state.running || !state.sessionId) return;
  const start = now();
  const plan = planSteps(start, start + hours);
  setRunning(true);
  state.stopping = false;
  state.selected = null;
  state.hover = null;
  $("moment").hidden = true;
  state.notices = state.notices.filter((n) => n.action);
  state.outsideTraining = [];
  renderNotices();
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  // On a small screen the controls sit above the chart, so bring the chart into view.
  if (matchMedia("(max-width: 860px)").matches) {
    $("chart").scrollIntoView({ block: "start", behavior: calm ? "auto" : "smooth" });
  }
  try {
    for (const [from, to] of plan) {
      if (state.stopping) break;
      const due = state.queue.filter((q) => q.at >= from - 1e-6 && q.at < to - 1e-6);
      const events = due.map(({ at, type, drug, dose_mg, route, pathogen }) => ({
        type,
        at_hours: Math.max(at - from, 0),
        ...(type === "dose" ? { drug, dose_mg, route } : { pathogen }),
      }));
      const result = await api(`v1/sessions/${state.sessionId}/step`, {
        method: "POST",
        body: { duration: to - from, events, include_internal: true },
      });
      state.history.push({
        step: result.step,
        time: result.time_hours,
        result,
        events: result.applied_events.map((e) => ({ ...e, time: from + e.at_hours })),
      });
      state.queue = state.queue.filter((q) => !due.includes(q));
      for (const warning of result.warnings) showNotice(warning);
      if (result.failed_virtual_people) {
        showNotice(`${plural(result.failed_virtual_people, "virtual person")} could not be simulated at ${elapsed(result.time_hours)} and ${result.failed_virtual_people === 1 ? "was" : "were"} left out.`);
      }
      if (state.model.status === "trained" && !result.support.within_training_support) {
        for (const reason of result.support.reasons) {
          const plain = plainReason(reason);
          if (!state.outsideTraining.includes(plain)) state.outsideTraining.push(plain);
        }
      }
      if (!calm) scheduleDraw();
    }
  } catch (problem) {
    fail(problem);
  } finally {
    setRunning(false);
    saveQueue();
    renderWaiting();
    renderNotices();
    drawNow();
    announce();
    loadSaved();
  }
}

function announce() {
  const parts = currentStrips().slice(0, 4).map((strip) => {
    const value = strip.read(state.history[state.history.length - 1].result);
    return value ? `${strip.label} ${value.p50}` : "";
  });
  $("announcer").textContent = `At ${elapsed(now())}. ${parts.filter(Boolean).join(", ")}.`;
}

// ---- Going back and trying other paths ----

async function goBackTo(step) {
  try {
    await api(`v1/sessions/${state.sessionId}/reset`, { method: "POST", body: { to_step: step } });
    state.queue = [];
    saveQueue();
    await openSession(state.sessionId);
  } catch (problem) {
    fail(problem);
  }
}

async function redrawPeople() {
  try {
    await api(`v1/sessions/${state.sessionId}/reset`, { method: "POST", body: { new_people: true } });
    state.queue = [];
    saveQueue();
    await openSession(state.sessionId);
  } catch (problem) {
    fail(problem);
  }
}

async function tryAnotherPath(step) {
  try {
    const parent = state.sessionId;
    const fork = await api(`v1/sessions/${parent}/fork`, { method: "POST", body: { from_step: step } });
    await openSession(fork.session_id);
    showNotice("You are on a new path from that moment. The dashed line shows the path you came from.");
  } catch (problem) {
    fail(problem);
  }
}

function family() {
  const parents = new Map(state.saved.map((s) => [s.session_id, s.forked_from && s.forked_from.session_id]));
  const rootOf = (id) => {
    let current = id;
    while (parents.get(current) && parents.has(parents.get(current))) current = parents.get(current);
    return current;
  };
  const root = rootOf(state.sessionId);
  return state.saved.filter((s) => s.session_id !== state.sessionId && rootOf(s.session_id) === root);
}

async function chooseComparison(sessionId) {
  state.compareId = sessionId || "";
  state.compareHistory = [];
  if (!sessionId) return;
  try {
    const past = await api(`v1/sessions/${sessionId}/history`);
    state.compareHistory = past.steps.map(normalise);
  } catch {
    state.compareId = "";
  }
}

function renderCompareRow() {
  const relatives = family();
  $("compare-row").hidden = !relatives.length;
  $("legend-compare").hidden = !state.compareId;
  if (!relatives.length) return;
  $("compare-select").replaceChildren(
    el("option", { value: "" }, "Nothing"),
    ...relatives.map((s) => el("option", { value: s.session_id, selected: s.session_id === state.compareId },
      `A path up to ${elapsed(s.time_hours)}`)),
  );
}

// ---- Saved sessions ----

function renderSaved() {
  const list = $("saved-list");
  $("saved-section").hidden = !state.saved.length;
  list.replaceChildren(...state.saved.map((saved) => {
    const remove = el("button", { type: "button" }, "Delete");
    askFirst(remove, "Delete for good?", async () => {
      await api(`v1/sessions/${saved.session_id}`, { method: "DELETE" });
      if (saved.session_id === state.sessionId) showSetup();
      await loadSaved();
    });
    const isOpen = saved.session_id === state.sessionId;
    const text = `${describePerson(saved.patient, saved.conditions)}, up to ${elapsed(saved.time_hours)}${saved.forked_from ? ", another path" : ""}`;
    return el("li", { class: saved.forked_from ? "path" : "" },
      text,
      el("div", { class: "saved-actions" },
        el("button", { type: "button", disabled: isOpen, onclick: () => openSession(saved.session_id).catch(fail) }, isOpen ? "Open now" : "Open"),
        remove));
  }));
}

// ---- What the chart shows ----

function defaultShown() {
  // Each organ lists its main measurement first, so a new organ shows up on its own.
  return state.catalog.organs.filter((o) => o.measurements.length).map((o) => `m:${o.measurements[0].name}`);
}

function shareFor(measurementName) {
  for (const organ of state.catalog.organs) {
    for (const share of organ.shares) {
      if (share.measurement === measurementName) return share;
    }
  }
  return null;
}

function drugsInPlay() {
  const names = new Set();
  for (const row of state.history) {
    for (const event of row.events) if (event.type === "dose") names.add(event.drug);
  }
  for (const item of state.queue) if (item.type === "dose") names.add(item.drug);
  return [...names];
}

function currentStrips() {
  const strips = [];
  for (const drug of drugsInPlay()) {
    strips.push({
      id: `d:${drug}`,
      label: `${drugLabel(drug)} level`,
      unit: "mg per litre",
      noise: 0.25,
      decimals: 2,
      read: (result) => (result.drug_levels_mg_l && result.drug_levels_mg_l[drug]) || NOTHING,
    });
  }
  for (const organ of state.catalog.organs) {
    for (const measurement of organ.measurements) {
      if (!state.shown.has(`m:${measurement.name}`)) continue;
      const share = shareFor(measurement.name);
      strips.push({
        id: `m:${measurement.name}`,
        label: measurement.label,
        unit: measurement.unit,
        noise: measurement.noise,
        decimals: decimalsFor(measurement.noise),
        share,
        threshold: share ? share.threshold : undefined,
        read: (result) => result.observations && result.observations[measurement.name],
      });
    }
  }
  for (const entry of internalEntries()) {
    if (state.shown.has(entry.id)) strips.push(...entry.strips());
  }
  return strips;
}

// Things inside the body. A state held per drug or per illness gets one strip for each
// one that actually moved, which hides the many that never change.
function internalEntries() {
  const entries = [];
  for (const organ of state.catalog.organs) {
    for (const st of organ.states) {
      entries.push({
        id: `s:${st.name}`,
        organ: organ.name,
        label: st.label,
        strips: () => {
          if (!st.per) {
            return [{
              id: `s:${st.name}`, label: st.label, unit: "", noise: 0.02, decimals: 2,
              read: (result) => result.internal && result.internal[st.name],
            }];
          }
          const seen = new Map();
          for (const row of state.history) {
            const group = row.result.internal && row.result.internal[st.name];
            for (const [key, value] of Object.entries(group || {})) {
              const range = seen.get(key) || [Infinity, -Infinity];
              seen.set(key, [Math.min(range[0], value.p10), Math.max(range[1], value.p90)]);
            }
          }
          return [...seen.entries()]
            .filter(([, range]) => range[1] - range[0] > 0.1)
            .map(([key]) => ({
              id: `s:${st.name}:${key}`,
              label: st.label,
              unit: st.per === "pathogen" ? illnessLabel(key) : drugLabel(key),
              noise: 0.02,
              decimals: 2,
              read: (result) => result.internal && result.internal[st.name] && result.internal[st.name][key],
            }));
        },
      });
    }
  }
  entries.push({
    id: "s:memory_neuron_activity",
    organ: "",
    label: "Memory neuron activity",
    strips: () => [{
      id: "s:memory_neuron_activity", label: "Memory neuron activity", unit: "", noise: 0.02, decimals: 2,
      read: (result) => result.internal && result.internal.memory_neuron_activity,
    }],
  });
  return entries;
}

function renderShownPicker() {
  const box = $("shown-list");
  const checkbox = (id, label) => el("label", { class: "check" },
    el("input", {
      type: "checkbox",
      checked: state.shown.has(id),
      onchange: (event) => {
        if (event.target.checked) state.shown.add(id);
        else state.shown.delete(id);
        remember("shown", [...state.shown]);
        drawNow();
      },
    }),
    label);
  const groups = [];
  const internals = internalEntries();
  for (const organ of state.catalog.organs) {
    const measured = organ.measurements.map((m) => checkbox(`m:${m.name}`, m.label));
    const inside = internals.filter((i) => i.organ === organ.name).map((i) => checkbox(i.id, `${i.label}, inside the body`));
    if (!measured.length && !inside.length) continue;
    groups.push(el("div", { class: "shown-group" },
      el("h3", {}, capitalise(organ.name)),
      el("p", {}, organ.description),
      ...measured,
      ...inside));
  }
  groups.push(el("div", { class: "shown-group" },
    el("h3", {}, "Neural network"),
    el("p", {}, "How active the trained memory neurons are."),
    checkbox("s:memory_neuron_activity", "Memory neuron activity")));
  box.replaceChildren(...groups);
}

// ---- Drawing ----

function scheduleDraw() {
  if (state.frame) return;
  state.frame = requestAnimationFrame(() => {
    state.frame = null;
    drawNow();
  });
}

function chartEnd() {
  const times = [now(), MINIMUM_CHART_HOURS, ...state.queue.map((q) => q.at)];
  if (state.compareHistory.length) times.push(state.compareHistory[state.compareHistory.length - 1].time);
  return Math.max(...times) * 1.02;
}

function seriesOf(rows, read) {
  const points = [];
  for (const row of rows) {
    const value = read(row.result);
    if (value) points.push({ t: row.time, p10: value.p10, p50: value.p50, p90: value.p90 });
  }
  return points;
}

function drawDoseRows(tMax, ticks, width) {
  const rows = new Map();
  const row = (key, label) => {
    if (!rows.has(key)) rows.set(key, { label, marks: [], ghostMarks: [], given: 0 });
    return rows.get(key);
  };
  for (const entry of state.history) {
    for (const event of entry.events) {
      if (event.type === "dose") {
        const target = row(`d:${event.drug}`, drugLabel(event.drug));
        target.marks.push({ t: event.time });
        target.given += 1;
      } else {
        row(`e:${event.pathogen}`, "Exposed").marks.push({ t: event.time, exposure: true });
      }
    }
  }
  for (const item of state.queue) {
    if (item.type === "dose") row(`d:${item.drug}`, drugLabel(item.drug)).marks.push({ t: item.at, waiting: true });
    else row(`e:${item.pathogen}`, "Exposed").marks.push({ t: item.at, waiting: true });
  }
  for (const entry of state.compareHistory) {
    for (const event of entry.events) {
      if (event.type === "dose") row(`d:${event.drug}`, drugLabel(event.drug)).ghostMarks.push(event.time);
    }
  }
  const nodes = [...rows.values()].map((data) => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("aria-hidden", "true");
    drawDoseRow(svg, { width, height: 24, tMax, ticks, marks: data.marks, ghostMarks: data.ghostMarks });
    return el("div", { class: "strip dose-row" },
      el("div", { class: "strip-label" }, data.label),
      el("div", { class: "strip-plot" }, svg),
      el("div", { class: "strip-value" }, data.given ? `${plural(data.given, "dose")} given` : ""));
  });
  $("dose-rows").replaceChildren(...nodes);
}

function drawNow() {
  if (!state.sessionId || $("chart").hidden) return;
  const width = $("axis-plot").clientWidth;
  if (!width) return;
  const tMax = chartEnd();
  const ticks = timeTicks(tMax);
  const doseTimes = state.history.flatMap((row) => row.events.filter((e) => e.type === "dose").map((e) => e.time));

  drawDoseRows(tMax, ticks, width);
  state.valueNodes = new Map();
  const nodes = currentStrips().map((strip) => {
    const points = seriesOf(state.history, strip.read);
    const ghost = seriesOf(state.compareHistory, strip.read);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("role", "img");
    const last = points[points.length - 1];
    svg.setAttribute("aria-label", `${strip.label} over time${last ? `, now ${last.p50.toFixed(strip.decimals)}` : ""}`);
    drawStrip(svg, {
      width,
      height: 72,
      tMax,
      ticks,
      points,
      ghost,
      doseTimes,
      domain: domainFor([...points, ...ghost], strip),
      threshold: strip.threshold,
      thresholdLabel: strip.share ? strip.share.label : "",
    });
    const main = el("div", { class: "value-main" });
    const range = el("div", { class: "value-range" });
    const share = el("div", { class: "value-share" });
    state.valueNodes.set(strip.id, { strip, main, range, share });
    return el("div", { class: "strip" },
      el("div", { class: "strip-label" }, strip.label, el("span", { class: "strip-unit" }, SHORT_UNITS[strip.unit] || strip.unit)),
      el("div", { class: "strip-plot" }, svg),
      el("div", { class: "strip-value" }, main, range, share));
  });
  $("strips").replaceChildren(...nodes);
  drawAxis($("axis-svg"), { width, height: 22, tMax, ticks });
  updateReadings();
}

// Readings follow the cursor, and otherwise show the present moment.
function updateReadings() {
  if (!state.history.length) return;
  const index = state.hover ?? state.selected ?? state.history.length - 1;
  const row = state.history[index];
  const looking = index !== state.history.length - 1;
  const clock = $("clock");
  clock.textContent = looking ? `Looking at ${elapsed(row.time)}` : now() ? `After ${elapsed(row.time)}` : "At the start";
  clock.classList.toggle("looking", looking);

  for (const { strip, main, range, share } of state.valueNodes.values()) {
    const value = strip.read(row.result);
    if (!value) {
      main.textContent = "";
      range.textContent = "";
      share.textContent = "";
      continue;
    }
    main.textContent = value.p50.toFixed(strip.decimals);
    range.textContent = `Most: ${value.p10.toFixed(strip.decimals)} to ${value.p90.toFixed(strip.decimals)}`;
    const portion = strip.share && row.result.share_of_people ? row.result.share_of_people[strip.share.key] : undefined;
    share.textContent = portion === undefined ? "" : `${strip.share.label}: ${Math.round(portion * 100)}% of people`;
  }

  renderBody(row);

  const cursor = $("cursor");
  const plot = $("axis-plot");
  const showing = state.hover !== null || state.selected !== null;
  cursor.hidden = !showing;
  if (showing) {
    const x = timeToX(row.time, chartEnd(), plot.clientWidth);
    cursor.style.left = `${plot.offsetLeft + x}px`;
  }
}

// The diagram shows the same moment as the readings, so moving along the chart replays it.
function renderBody(row) {
  const parts = new Map();
  const part = (name) => {
    if (!parts.has(name)) parts.set(name, { heat: 0, lines: [] });
    return parts.get(name);
  };
  const clamp = (x) => Math.min(Math.max(x, 0), 1);
  let germs = 0;
  for (const organ of state.catalog.organs) {
    for (const m of organ.measurements) {
      const value = row.result.observations && row.result.observations[m.name];
      if (!m.body_part || !value) continue;
      const heat = clamp((value.p50 - m.healthy) / (m.severe - m.healthy));
      if (m.germs) germs = Math.max(germs, heat);
      else part(m.body_part).heat = Math.max(part(m.body_part).heat, heat);
      if (heat > 0.04) {
        part(m.body_part).lines.push(`${m.label} ${value.p50.toFixed(decimalsFor(m.noise))} ${SHORT_UNITS[m.unit] || m.unit}`);
      }
    }
  }

  let drug = 0;
  for (const [name, level] of Object.entries(row.result.drug_levels_mg_l || {})) {
    const peak = Math.max(...state.history.map((h) => (h.result.drug_levels_mg_l && h.result.drug_levels_mg_l[name] ? h.result.drug_levels_mg_l[name].p50 : 0)));
    drug = Math.max(drug, peak ? level.p50 / peak : 0);
    if (level.p50 >= 0.01) part("blood").lines.push(`${drugLabel(name)} ${level.p50.toFixed(2)} mg/L`);
  }

  const svg = $("body-svg");
  drawBody(svg, { parts, germs, drug });

  const showing = [...parts.entries()]
    .filter(([, data]) => data.lines.length)
    .sort(([a], [b]) => partOrder(a) - partOrder(b));
  $("body-list").replaceChildren(...showing.map(([name, data]) => {
    const fill = el("i");
    fill.style.opacity = data.heat;
    return el("li", {},
      el("div", { class: "body-swatch" }, fill),
      el("div", {}, el("strong", {}, partLabel(name)), ...data.lines.map((line) => el("span", {}, line))));
  }));

  const signs = [];
  for (const organ of state.catalog.organs) {
    for (const share of organ.shares) {
      const portion = row.result.share_of_people && row.result.share_of_people[share.key];
      if (portion >= 0.5) signs.push(share.label.toLowerCase());
    }
  }
  const status = signs.length ? `Most of these people have: ${signs.join(", ")}.` : "Most of these people show no signs of illness.";
  $("body-status").textContent = status;
  svg.setAttribute("aria-label", `Body diagram. ${status}`);
}

// Reasons from the server name parameters by their code names, so use their plain names here.
function plainReason(reason) {
  let text = reason;
  for (const [name, entry] of Object.entries(state.catalog.patient)) {
    text = text.replace(name, entry.note.split(",")[0]);
  }
  return text;
}

function renderNotices() {
  const box = $("notices");
  const items = state.notices.map((notice) => el("p", { class: "notice" },
    el("span", {}, notice.text),
    notice.action ? el("button", { type: "button", onclick: notice.action.run }, notice.action.label) : ""));
  if (state.outsideTraining.length) {
    items.push(el("p", { class: "notice" },
      el("span", {}, `This goes beyond what the model was trained on, so treat it with extra care. ${state.outsideTraining.join(" ")}`)));
  }
  box.replaceChildren(...items);
}

function renderModelLine() {
  const line = $("model-line");
  if (state.model.status === "trained") {
    const support = state.model.details && state.model.details.support;
    line.textContent = support ? `Trained on ${plural(support.subjects, "subject")}` : "Trained";
    line.title = "Trained on the studies listed at /v1/model. Still a research simulation.";
  } else {
    line.textContent = "Untrained, equations only";
    line.title = "Results come from textbook equations and assumptions, not from fitted data.";
  }
}

// ---- Choosing a moment on the chart ----

function indexAt(clientX) {
  const plot = $("axis-plot").getBoundingClientRect();
  if (clientX < plot.left || clientX > plot.right || !state.history.length) return null;
  const tMax = chartEnd();
  const t = ((clientX - plot.left) / Math.max(plot.width - 8, 1)) * tMax;
  let best = 0;
  let distance = Infinity;
  state.history.forEach((row, i) => {
    const gap = Math.abs(row.time - t);
    if (gap < distance) {
      distance = gap;
      best = i;
    }
  });
  return best;
}

function selectMoment(index) {
  state.selected = index;
  const row = state.history[index];
  const atEnd = index === state.history.length - 1;
  $("moment").hidden = false;
  $("moment-time").textContent = `At ${elapsed(row.time)}`;
  $("moment-back").hidden = atEnd;
  updateReadings();
}

function closeMoment() {
  state.selected = null;
  state.hover = null;
  $("moment").hidden = true;
  updateReadings();
}

function wireChart() {
  const sheet = $("flowsheet");
  sheet.addEventListener("pointermove", (event) => {
    const index = indexAt(event.clientX);
    if (index === state.hover) return;
    state.hover = index;
    updateReadings();
  });
  sheet.addEventListener("pointerleave", () => {
    state.hover = null;
    updateReadings();
  });
  sheet.addEventListener("click", (event) => {
    const index = indexAt(event.clientX);
    if (index !== null && !state.running) selectMoment(index);
  });
  sheet.addEventListener("keydown", (event) => {
    if (!state.history.length || state.running) return;
    const last = state.history.length - 1;
    const from = state.selected ?? last;
    if (event.key === "ArrowLeft") selectMoment(Math.max(from - 1, 0));
    else if (event.key === "ArrowRight") selectMoment(Math.min(from + 1, last));
    else if (event.key === "Home") selectMoment(0);
    else if (event.key === "End") selectMoment(last);
    else if (event.key === "Escape") closeMoment();
    else return;
    event.preventDefault();
  });
  $("moment-close").addEventListener("click", closeMoment);
  $("moment-fork").addEventListener("click", () => tryAnotherPath(state.history[state.selected].step));
  askFirst($("moment-back"), "Forget what came after?", () => goBackTo(state.history[state.selected].step));
}

// ---- Starting up ----

function wireControls() {
  $("setup").addEventListener("submit", (event) => {
    event.preventDefault();
    startSession(readSetup());
  });
  $("example-button").addEventListener("click", () => {
    const first = Object.keys(state.catalog.conditions)[0];
    startSession({
      patient: { age_years: 30, sex: "male", weight_kg: 75 },
      conditions: first ? [{ name: first, days_since_symptom_onset: 1 }] : [],
      people: 64,
      include_internal: true,
    });
  });
  $("give").addEventListener("submit", giveDose);
  $("give-drug").addEventListener("change", fillUsualDose);
  $("give-every").addEventListener("change", () => {
    const repeating = Number($("give-every").value) > 0;
    $("give-count-field").hidden = !repeating;
    $("give-button").textContent = repeating ? "Start this course" : "Give dose";
  });
  $("expose-button").addEventListener("click", expose);
  for (const button of $("time-buttons").querySelectorAll("button")) {
    button.addEventListener("click", () => moveTime(Number(button.dataset.hours)));
  }
  $("custom-time").addEventListener("submit", (event) => {
    event.preventDefault();
    const hours = Number($("custom-hours").value);
    if (hours > 0) moveTime(hours);
  });
  $("stop-button").addEventListener("click", () => { state.stopping = true; });
  askFirst($("restart-button"), "Forget everything?", () => goBackTo(0));
  $("new-person-button").addEventListener("click", showSetup);
  askFirst($("redraw-button"), "Forget everything so far?", redrawPeople);
  $("compare-select").addEventListener("change", async (event) => {
    await chooseComparison(event.target.value);
    $("legend-compare").hidden = !state.compareId;
    drawNow();
  });
  new ResizeObserver(scheduleDraw).observe($("axis-plot"));
}

async function boot() {
  wireControls();
  wireChart();
  try {
    [state.catalog, state.model] = await Promise.all([api("v1/catalog"), api("v1/model")]);
  } catch (problem) {
    $("welcome").replaceChildren(el("p", { class: "error" }, problem.message));
    return;
  }
  buildSetupForm();
  renderModelLine();
  const stored = recall("shown");
  state.shown = new Set(Array.isArray(stored) ? stored : defaultShown());
  await loadSaved();
  const wanted = location.hash.slice(1);
  if (wanted && /^s_[0-9a-f]+$/.test(wanted)) {
    try {
      await openSession(wanted);
      return;
    } catch {
      // The session is gone, so start fresh.
    }
  }
  showSetup();
}

boot();
