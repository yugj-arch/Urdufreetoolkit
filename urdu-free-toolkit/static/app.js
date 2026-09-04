/* Urdu Toolkit front-end. No dependencies. Talks to:
   GET  /api/config        -> runtime flags (settings read-only on Vercel)
   GET  /api/providers     -> engine lists per capability
   POST /api/ocr           -> SSE stream, one row per engine (single image)
   POST /api/transliterate -> {results:[...]}  (also used for one-word lookups)
   GET/POST /api/settings   -> API-key status / save

   The tool is two steps — read an image, then transliterate — and the result
   is shown as a Rekhta-style ghazal: couplet sets, a اردو / हिंदी / Roman
   switch that repaints in place, a text-size control, per-couplet copy /
   "download as card", and tap-a-word to see that word in every script.
*/
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/* The engines each step pre-ticks on load: the recommended ones that are
   available, else the first working engine of that kind. You can tick more than
   one per step to compare them side by side. */
const RECOMMENDED = { ocr: ["gcv"], translit: ["rule"] };

const ZOOM = { min: 0.7, max: 1.9, step: 0.15 };

const state = {
  providers: { ocr: [], translit: [] },
  engines: { ocr: [], translit: [] },   // ticked engine ids per step
  config: { settings_readonly: false },
  script: "ur",          // ur | hi | ro   — which script the reader shows
  diacritics: false,     // Roman spelling: false = plain, true = Rekhta marks
  scale: 1,              // reader text size multiplier
  results: {},           // engine id -> { ur:[], hi:[], ro:[], rod:[], ms } (step 2)
  activeEngine: null,    // which engine's ghazal the reader is showing
  result: null,          // results[activeEngine] — the lines the reader renders
  wordCache: {},         // urdu token -> { ur, hi, ro, rod }
  ready: null,           // promise: provider discovery finished
  file: null,            // the image queued in step 1
};

/* ---------- boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  wireDropzone();
  wireSettings();
  wireReaderControls();
  wireWordPop();
  wireQuickChips();
  wireReaderEngines();
  $("#run-ocr").addEventListener("click", runOcr);
  $("#run-translit").addEventListener("click", runTranslit);

  try { state.diacritics = localStorage.getItem("urdu.diacritics") === "on"; } catch (e) {}
  try { state.scale = clampZoom(parseFloat(localStorage.getItem("urdu.readerScale")) || 1); } catch (e) {}
  try { state.script = ["ur", "hi", "ro"].includes(localStorage.getItem("urdu.script"))
        ? localStorage.getItem("urdu.script") : "ur"; } catch (e) {}

  loadConfig();
  state.ready = loadProviders();   // first call warms provider discovery — can take a second
});

/* ---------- runtime config ---------- */
async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (res.ok) Object.assign(state.config, await res.json());
  } catch (e) { /* keep local defaults */ }
}

/* ---------- providers ---------- */
const ENGINE_BOX = { ocr: "#ocr-engines", translit: "#translit-engines" };

async function loadProviders() {
  try {
    state.providers = await (await fetch("/api/providers")).json();
  } catch (e) {
    $("#ocr-engines").innerHTML = '<p class="err">Could not load engines.</p>';
    return;
  }
  renderEngineList("ocr");
  renderEngineList("translit");
}

/* The tickable engine list for a step. Unavailable engines (no API key) show
   greyed out with the reason, so you can see what a key would unlock. After
   drawing, settle on a sensible set of ticks. */
function renderEngineList(cap) {
  const box = $(ENGINE_BOX[cap]);
  if (!box) return;
  const rows = state.providers[cap] || [];
  if (!rows.length) { box.innerHTML = '<p class="hint">No engines found for this step.</p>'; return; }

  box.innerHTML = rows.map((r) => {
    const off = r.available ? "" : " off";
    const dis = r.available ? "" : " disabled";
    const reason = r.available ? "" : ` title="${esc(r.reason || "unavailable")}"`;
    return `<label class="eng${off}"${reason}>
      <input type="checkbox" value="${esc(r.id)}" data-cap="${cap}"${dis}>
      <span>
        <span class="lbl">${esc(r.label)}</span><span class="badge-tag">${esc(r.badge || "")}</span>
        ${r.price ? `<span class="price-tag">${esc(r.price)}</span>` : ""}
        ${r.note ? `<span class="note">${esc(r.note)}</span>` : ""}
        ${r.available ? "" : `<span class="note">${esc(r.reason || "needs an API key")}</span>`}
      </span>
    </label>`;
  }).join("");

  box.querySelectorAll('input[type="checkbox"]').forEach((b) =>
    b.addEventListener("change", () => saveEngines(cap)));
  restoreEngines(cap);
}

/* Tick the set remembered from last time (dropping any that can't run now); if
   that leaves nothing, tick the recommended engine, else the first that works. */
function restoreEngines(cap) {
  const rows = state.providers[cap] || [];
  const avail = new Set(rows.filter((r) => r.available).map((r) => r.id));
  let want = savedEngines(cap).filter((id) => avail.has(id));
  if (!want.length) {
    const rec = (RECOMMENDED[cap] || []).filter((id) => avail.has(id));
    want = rec.length ? rec : (avail.size ? [[...avail][0]] : []);
  }
  setChecked(cap, want);
  state.engines[cap] = checkedIds(cap);
}

function checkedIds(cap) {
  return $$(`input[data-cap="${cap}"]:checked`).map((c) => c.value);
}
function setChecked(cap, ids) {
  const on = new Set(ids);
  $$(`input[data-cap="${cap}"]`).forEach((c) => { c.checked = !c.disabled && on.has(c.value); });
}
function saveEngines(cap) {
  state.engines[cap] = checkedIds(cap);
  try { localStorage.setItem("urdu.engines." + cap, state.engines[cap].join(",")); } catch (e) {}
}
function savedEngines(cap) {
  try { return (localStorage.getItem("urdu.engines." + cap) || "").split(",").filter(Boolean); }
  catch (e) { return []; }
}

/* The "Recommended" / "Clear" chips under each list. */
function wireQuickChips() {
  $$("[data-pick]").forEach((b) => b.addEventListener("click", () => {
    const [cap, what] = b.dataset.pick.split(":");
    const rows = state.providers[cap] || [];
    const avail = new Set(rows.filter((r) => r.available).map((r) => r.id));
    if (what === "none") setChecked(cap, []);
    else if (what === "recommended") setChecked(cap, (RECOMMENDED[cap] || []).filter((id) => avail.has(id)));
    saveEngines(cap);
  }));
}

function labelFor(cap, id) {
  const row = (state.providers[cap] || []).find((r) => r.id === id);
  return row ? row.label : id;
}

/* ---------- step 1: read one image ---------- */
function wireDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");
  const preview = $("#preview");
  const label = $("#drop-label");

  const show = (file) => {
    state.file = file;
    $("#run-ocr").disabled = !file;
    if (!file) { preview.hidden = true; preview.src = ""; return; }
    const rd = new FileReader();
    rd.onload = () => { preview.src = rd.result; preview.hidden = false; };
    rd.readAsDataURL(file);
    label.innerHTML = "<strong>" + esc(file.name) + "</strong> — click to replace";
  };

  input.addEventListener("change", () => { if (input.files[0]) show(input.files[0]); input.value = ""; });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
    if (f) show(f);
  });
}

async function runOcr() {
  if (!state.file) return;
  if (!state.engines.ocr.length) await state.ready;
  const ids = state.engines.ocr.length ? state.engines.ocr : checkedIds("ocr");
  if (!ids.length) return fail("#ocr-err", "Tick at least one OCR engine.");

  hide("#ocr-err");
  const btn = $("#run-ocr");
  btn.disabled = true; btn.textContent = "Reading…";

  const out = $("#ocr-out");
  out.hidden = false;
  out.innerHTML = "";
  ids.forEach((id) => out.appendChild(ocrCardShell(id)));

  try {
    const fd = new FormData();
    fd.append("image", state.file);
    fd.append("providers", ids.join(","));
    const res = await fetch("/api/ocr", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await errText(res));
    await readSSE(res, (payload) => { if (!payload.done) fillOcrCard(payload); });
  } catch (e) {
    fail("#ocr-err", "Couldn't read the image: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Read image";
  }
}

function ocrCardShell(id) {
  const el = document.createElement("div");
  el.className = "ocr-card";
  el.dataset.pid = id;
  el.innerHTML =
    `<div class="card-h"><span class="lbl">${esc(labelFor("ocr", id))}</span><span class="ms">reading…</span></div>` +
    `<div class="running hint">running…</div>`;
  return el;
}

function fillOcrCard(row) {
  const el = $(`#ocr-out .ocr-card[data-pid="${cssEsc(row.provider_id)}"]`);
  if (!el) return;
  const head = `<div class="card-h"><span class="lbl">${esc(labelFor("ocr", row.provider_id))}</span>` +
    `<span class="ms">${row.ms != null ? row.ms + " ms" : ""}</span></div>`;
  if (!row.ok) {
    el.innerHTML = head + `<p class="err">${esc(row.error || "the engine failed")}</p>`;
    return;
  }
  el.innerHTML = head;
  const ta = document.createElement("textarea");
  ta.dir = "rtl";
  ta.value = row.text || "";
  el.appendChild(ta);
  if (!(row.text || "").trim())
    el.insertAdjacentHTML("beforeend", '<p class="err">read no text from this image</p>');

  const use = document.createElement("button");
  use.className = "btn-primary use-btn";
  use.type = "button";
  use.innerHTML = "Use this text&nbsp;↓";
  use.addEventListener("click", () => {
    $("#urdu-input").value = ta.value;
    $("#step-translit").scrollIntoView({ behavior: "smooth", block: "start" });
    runTranslit();
  });
  el.appendChild(use);
}

/* ---------- step 2: transliterate ---------- */
async function runTranslit() {
  const text = $("#urdu-input").value.replace(/\s+$/g, "");
  if (!text.trim()) return fail("#tr-err", "Nothing to transliterate yet.");
  if (!state.engines.translit.length) await state.ready;
  const ids = state.engines.translit.length ? state.engines.translit : checkedIds("translit");
  if (!ids.length) return fail("#tr-err", "Tick at least one transliteration engine.");

  hide("#tr-err");
  const btn = $("#run-translit");
  btn.disabled = true; btn.textContent = "Working…";

  try {
    const res = await fetch("/api/transliterate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, providers: ids }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    const rows = (data.results || []).filter((r) => r && r.ok);
    if (!rows.length) {
      const first = (data.results || [])[0];
      throw new Error((first && first.error) || "the engine failed");
    }

    const urLines = splitLines(text);
    state.results = {};
    rows.forEach((row) => {
      state.results[row.provider_id] = {
        ur: urLines,
        hi: splitLines(row.devanagari),
        ro: splitLines(row.roman),
        rod: splitLines(row.roman_diacritic || row.roman),
        ms: row.ms,
      };
    });
    state.wordCache = {};
    renderReaderEngines(rows.map((r) => r.provider_id));
    showEngine(rows[0].provider_id);
    $("#reader").hidden = false;
    applyScript(); applyZoom(); applyDiacritics();
    $("#reader").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    fail("#tr-err", "Transliteration failed: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Transliterate";
  }
}

/* The engine tab strip above the reader — one button per engine that returned a
   result. Hidden when only one engine ran. */
function renderReaderEngines(ids) {
  const bar = $("#reader-engines");
  if (!bar) return;
  bar.hidden = ids.length < 2;
  bar.innerHTML = ids.map((id) =>
    `<button type="button" data-eng="${esc(id)}">${esc(labelFor("translit", id))}</button>`).join("");
}
function wireReaderEngines() {
  const bar = $("#reader-engines");
  if (!bar) return;
  bar.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-eng]");
    if (b) showEngine(b.dataset.eng);
  });
}

/* Point the reader at one engine's result and repaint. Every other reader
   control (script, size, diacritics, word popover, card) works off state.result
   and needs no changes. */
function showEngine(id) {
  if (!state.results[id]) return;
  state.activeEngine = id;
  state.result = state.results[id];
  state.wordCache = {};
  closeWordPop();
  $$("#reader-engines button").forEach((b) => {
    const on = b.dataset.eng === id;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
  const r = state.results[id];
  $("#reader-foot").textContent =
    "via " + labelFor("translit", id) + (r.ms != null ? "  ·  " + r.ms + " ms" : "");
  renderGhazal();
}

/* ---------- reader controls ---------- */
function wireReaderControls() {
  $$("#script-switch button").forEach((b) =>
    b.addEventListener("click", () => { state.script = b.dataset.s; persist(); applyScript(); repaintReader(); }));
  $$("#size-switch button").forEach((b) =>
    b.addEventListener("click", () => {
      state.scale = clampZoom(state.scale + (b.dataset.z === "+" ? ZOOM.step : -ZOOM.step));
      persist(); applyZoom();
    }));
  $$("#dia-switch button").forEach((b) =>
    b.addEventListener("click", () => {
      state.diacritics = b.dataset.d === "on";
      // Always live — never disabled, never forces a script change. Sets the
      // Roman spelling; visible now if the reader shows Roman, otherwise the
      // moment the reader is switched to Roman.
      persist(); applyDiacritics(); repaintReader();
    }));
}

const clampZoom = (v) => Math.min(ZOOM.max, Math.max(ZOOM.min, Math.round(v * 100) / 100));
function persist() {
  try {
    localStorage.setItem("urdu.script", state.script);
    localStorage.setItem("urdu.readerScale", String(state.scale));
    localStorage.setItem("urdu.diacritics", state.diacritics ? "on" : "off");
  } catch (e) {}
}

function applyScript() {
  $("#ghazal").dataset.script = state.script;
  $$("#script-switch button").forEach((b) => {
    const on = b.dataset.s === state.script;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
}
function applyZoom() { $("#ghazal").style.setProperty("--rs", state.scale); }
function applyDiacritics() {
  $$("#dia-switch button").forEach((b) => {
    const on = (b.dataset.d === "on") === state.diacritics;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

/* The reader repaints from stored lines only — the Plain/Diacritics switch and
   the script switch never re-hit the server. */
function repaintReader() { if (state.result) renderGhazal(); }

/* ---------- ghazal rendering ---------- */
const curKey = () => (state.script === "ro" ? (state.diacritics ? "rod" : "ro") : state.script);
function linesFor() {
  const k = curKey();
  const n = maxLines();
  const src = (state.result && state.result[k]) || [];
  return Array.from({ length: n }, (_, i) => src[i] || "");
}
function maxLines() {
  if (!state.result) return 0;
  return Math.max(...["ur", "hi", "ro", "rod"].map((k) => (state.result[k] || []).length), 0);
}

function renderGhazal() {
  const g = $("#ghazal");
  const lines = linesFor();
  const rtl = state.script === "ur";
  g.innerHTML = "";
  closeWordPop();

  for (let i = 0; i < lines.length; i += 2) {
    const sher = document.createElement("div");
    sher.className = "sher";
    const body = document.createElement("div");
    body.className = "sher-body";
    for (let j = i; j < Math.min(i + 2, lines.length); j++) {
      body.appendChild(misraEl(lines[j], j, rtl));
    }
    sher.appendChild(body);
    sher.appendChild(sherTools(i));
    g.appendChild(sher);
  }
  if (!lines.length) g.innerHTML = '<p class="hint">Nothing to show.</p>';
}

function misraEl(text, lineIdx, rtl) {
  const p = document.createElement("p");
  p.className = "misra";
  if (rtl) p.dir = "rtl";
  const parts = String(text).split(/(\s+)/);
  parts.forEach((chunk, k) => {
    if (!chunk) return;
    if (k % 2 === 1 || /^\s+$/.test(chunk)) { p.appendChild(document.createTextNode(chunk)); return; }
    const w = document.createElement("span");
    w.className = "w";
    w.textContent = chunk;
    w.dataset.line = lineIdx;
    w.dataset.wi = (k / 2) | 0;
    w.addEventListener("click", (e) => { e.stopPropagation(); openWordPop(w); });
    p.appendChild(w);
  });
  return p;
}

function sherTools(startLine) {
  const box = document.createElement("div");
  box.className = "sher-tools";
  const copy = mkBtn("Copy", () => {
    const txt = linesFor().slice(startLine, startLine + 2).filter(Boolean).join("\n");
    copyText(txt, copy);
  });
  const card = mkBtn("Card", () => downloadCard(startLine));
  box.append(copy, card);
  return box;
}
function mkBtn(label, fn) {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = label;
  b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
  return b;
}

/* ---------- word popover ---------- */
function wireWordPop() {
  $("#wp-close").addEventListener("click", closeWordPop);
  document.addEventListener("click", (e) => {
    if (!$("#word-pop").hidden && !$("#word-pop").contains(e.target)) closeWordPop();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeWordPop(); });
  window.addEventListener("resize", closeWordPop);
  window.addEventListener("scroll", () => { if (!$("#word-pop").hidden) closeWordPop(); }, { passive: true });
}

function closeWordPop() {
  $("#word-pop").hidden = true;
  $$(".w.is-open").forEach((w) => w.classList.remove("is-open"));
}

async function openWordPop(wordEl) {
  closeWordPop();
  wordEl.classList.add("is-open");
  const pop = $("#word-pop");
  const lineIdx = +wordEl.dataset.line, wi = +wordEl.dataset.wi;

  let forms = alignedForms(lineIdx, wi);
  const fill = (f) => ["ur", "hi", "ro", "rod"].forEach((k) =>
    ($(`#word-pop [data-k="${k}"]`).textContent = (f && f[k]) || "—"));

  if (forms) {
    pop.classList.remove("is-loading");
    fill(forms);
  } else if (state.script === "ur") {
    fill({ ur: wordEl.textContent });
    pop.classList.add("is-loading");
    forms = await lookupWord(wordEl.textContent.trim());
    pop.classList.remove("is-loading");
    if (forms) fill(forms);
  } else {
    // couldn't line the scripts up and we don't have the Urdu token
    pop.classList.remove("is-loading");
    fill({ [curKey()]: wordEl.textContent });
  }

  placePop(pop, wordEl);
}

/* Line the four scripts up token-for-token. Works whenever every script split
   the misra into the same number of words — which the rule engine almost
   always does. */
function alignedForms(lineIdx, wi) {
  if (!state.result) return null;
  const toks = {};
  for (const k of ["ur", "hi", "ro", "rod"]) {
    toks[k] = String((state.result[k] || [])[lineIdx] || "").split(/\s+/).filter(Boolean);
  }
  const n = toks[curKey()].length;
  if (!n || !["ur", "hi", "ro", "rod"].every((k) => toks[k].length === n)) return null;
  return { ur: toks.ur[wi], hi: toks.hi[wi], ro: toks.ro[wi], rod: toks.rod[wi] };
}

async function lookupWord(tok) {
  if (!tok) return null;
  if (state.wordCache[tok]) return state.wordCache[tok];
  try {
    const res = await fetch("/api/transliterate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: tok, providers: [state.activeEngine] }),
    });
    const row = ((await res.json()).results || [])[0];
    if (!row || !row.ok) return null;
    const f = { ur: tok, hi: row.devanagari || "—",
                ro: row.roman || "—", rod: row.roman_diacritic || row.roman || "—" };
    state.wordCache[tok] = f;
    return f;
  } catch (e) { return null; }
}

function placePop(pop, anchor) {
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  let left = r.left + window.scrollX + r.width / 2 - pw / 2;
  left = Math.max(10 + window.scrollX, Math.min(left, window.scrollX + document.documentElement.clientWidth - pw - 10));
  // prefer above the word — the next misra usually sits right below it
  const roomAbove = r.top - 70 > ph;
  const top = roomAbove ? r.top + window.scrollY - ph - 8 : r.bottom + window.scrollY + 8;
  pop.style.left = left + "px";
  pop.style.top = top + "px";
}

/* ---------- couplet -> PNG card (native canvas, no libraries) ---------- */
function downloadCard(startLine) {
  const lines = linesFor().slice(startLine, startLine + 2).filter(Boolean);
  if (!lines.length) return;
  const W = 1200, H = 630;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const c = cv.getContext("2d");
  c.fillStyle = "#fdfaf3"; c.fillRect(0, 0, W, H);
  c.fillStyle = "#eb0046"; c.fillRect(W / 2 - 34, 96, 68, 3);

  const ur = state.script === "ur";
  const fam = ur ? '"Noto Nastaliq Urdu","Jameel Noori Nastaleeq",serif'
    : state.script === "hi" ? '"Nirmala UI","Noto Serif Devanagari",serif'
      : 'Georgia,"Times New Roman",serif';
  const size = ur ? 50 : 44;
  c.fillStyle = "#171717";
  c.textAlign = "center"; c.textBaseline = "middle";
  c.direction = ur ? "rtl" : "ltr";
  c.font = size + 'px ' + fam;
  const gap = size * 1.9;
  const startY = H / 2 - (lines.length - 1) * gap / 2;
  lines.forEach((ln, i) => c.fillText(ln, W / 2, startY + i * gap, W - 180));

  c.font = '20px Georgia, serif'; c.fillStyle = "#8a8a8a"; c.direction = "ltr";
  c.fillText("اردو  ·  Urdu Toolkit", W / 2, H - 52);

  const a = document.createElement("a");
  a.href = cv.toDataURL("image/png");
  a.download = "sher.png";
  a.click();
}

/* ---------- settings ---------- */
/* The Settings panel is hidden from the site. Keys are read from the host's
   environment; /api/settings and the settings_readonly flag still back it. */
function wireSettings() {
  const btn = $("#settings-btn");
  if (!btn) return;
  const modal = $("#settings-modal");
  btn.addEventListener("click", async () => {
    const status = await (await fetch("/api/settings")).json();
    const ro = state.config.settings_readonly;
    $("#settings-fields").innerHTML = Object.keys(status).map((k) =>
      `<label>${esc(k)} <span class="hint">${status[k] ? "(set)" : "(not set)"}</span></label>
       <input type="${k.endsWith("KEY") ? "password" : "text"}" data-key="${esc(k)}"
              ${ro ? "disabled" : ""}
              placeholder="${ro ? "" : (status[k] ? "leave blank to keep" : "")}">`).join("");
    const note = $("#settings-readonly-note");
    if (note) note.hidden = !ro;
    $("#save-settings").disabled = ro;
    $("#settings-status").textContent = "";
    if (!state.providers.ocr.length) await loadProviders();
    renderSettingsEngines();
    modal.hidden = false;
  });
  $("#close-settings").addEventListener("click", () => (modal.hidden = true));
  $("#save-settings").addEventListener("click", async () => {
    if (state.config.settings_readonly) return;
    const body = {};
    $$("#settings-fields input").forEach((i) => { if (i.value.trim()) body[i.dataset.key] = i.value.trim(); });
    if (!Object.keys(body).length) { modal.hidden = true; return; }
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    $("#settings-status").textContent = "Saved: " + (data.saved || []).join(", ");
    await loadProviders();
    renderSettingsEngines();
    setTimeout(() => (modal.hidden = true), 700);
  });
}

function renderSettingsEngines() {
  const box = $("#settings-engines");
  if (!box) return;
  box.innerHTML = [["ocr", "OCR"], ["translit", "Transliterate"]].map(([cap, title]) => {
    const rows = state.providers[cap] || [];
    if (!rows.length) return "";
    return `<div class="eng-price-group"><strong>${esc(title)}</strong>` +
      rows.map((r) => `<div class="eng-price-row"><span>${esc(r.label)}` +
        `${r.available ? "" : ' <span class="hint">— ' + esc(r.reason || "needs a key") + "</span>"}</span>` +
        `<span class="price-tag">${esc(r.price || "—")}</span></div>`).join("") +
      `</div>`;
  }).join("");
}

/* ---------- tiny helpers ---------- */
function splitLines(s) {
  return String(s == null ? "" : s).replace(/\r/g, "").split("\n").map((l) => l.trim()).filter(Boolean);
}
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function cssEsc(s) { return String(s).replace(/["\\]/g, "\\$&"); }
function hide(sel) { const e = $(sel); if (e) e.hidden = true; }
function fail(sel, msg) { const e = $(sel); if (e) { e.textContent = msg; e.hidden = false; } }
async function errText(res) {
  try { return (await res.json()).error || res.statusText; } catch (e) { return res.statusText; }
}
async function readSSE(res, onPayload) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith("data:")) onPayload(JSON.parse(line.slice(5).trim()));
    }
  }
}
function copyText(text, btn) {
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    if (!btn) return;
    const old = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => (btn.textContent = old), 1200);
  });
}
