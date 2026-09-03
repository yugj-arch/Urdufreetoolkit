/* Urdu Toolkit front-end. No dependencies. Talks to:
   GET  /api/config           -> runtime flags (batch cap, settings read-only)
   GET  /api/providers        -> engine lists per capability
   POST /api/ocr              -> SSE stream, one row per engine (single image)
   POST /api/transliterate    -> {results:[...]}
   POST /api/batch            -> SSE stream, one row per (file x ocr engine) for many images
   GET/POST /api/settings     -> API-key status / save
*/
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/* Engine ids the "Recommended" chip ticks per step — also what loadProviders()
   pre-selects on load. If none of them are available (e.g. an API key isn't
   set), loadProviders() falls back to the first engine that actually works. */
const RECOMMENDED = {
  ocr: ["gcv"],
  translit: ["rule"],
};

const state = {
  images: [],     // File[] added to the reader — one image or many
  providers: { ocr: [], translit: [] },
  batchRows: [],  // accumulated result rows, for the CSV download (multi-image runs)
  config: { on_vercel: false, batch_max_files: 30, settings_readonly: false },
  diacritics: false,  // Roman spelling: false = plain ASCII, true = Rekhta-style marks
};

/* ---------- boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  wireDropzone();
  wireSettings();
  wireScriptToggle();
  wireDiacriticsToggle();
  $("#run-ocr").addEventListener("click", runOcr);
  $("#run-translit").addEventListener("click", runTranslit);
  $("#batch-csv").addEventListener("click", downloadBatchCsv);
  $$("[data-pick]").forEach((b) => b.addEventListener("click", () => pick(b.dataset.pick)));
  loadConfig();
  loadProviders();
});

/* ---------- script toggle (Roman / Devanagari / Both) ---------- */
/* Mirrors Rekhta's ENG/HIN/URD switch: picks which script the result cards
   show. Pure CSS does the hiding via body[data-script]; choice is remembered. */
function wireScriptToggle() {
  const box = $("#script-toggle");
  if (!box) return;
  let saved = "both";
  try { saved = localStorage.getItem("urdu.script") || "both"; } catch (e) {}
  setScript(saved);
  $$("#script-toggle button").forEach((b) =>
    b.addEventListener("click", () => setScript(b.dataset.script)));
}

function setScript(v) {
  if (!["roman", "deva", "both"].includes(v)) v = "both";
  document.body.dataset.script = v;
  try { localStorage.setItem("urdu.script", v); } catch (e) {}
  $$("#script-toggle button").forEach((b) => {
    const on = b.dataset.script === v;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

/* ---------- diacritics toggle (Plain / Diacritics) ---------- */
/* Picks the Roman spelling shown: plain ASCII ("khraab") or Rekhta-style marks
   ("ḳharāb"). Every result already carries BOTH spellings, so flipping this
   just repaints the Roman blocks on screen — no re-run. Choice is remembered.
   Devanagari is unaffected. */
function wireDiacriticsToggle() {
  const box = $("#diacritics-toggle");
  if (!box) return;
  let saved = "off";
  try { saved = localStorage.getItem("urdu.diacritics") || "off"; } catch (e) {}
  setDiacritics(saved === "on");
  $$("#diacritics-toggle button").forEach((b) =>
    b.addEventListener("click", () => setDiacritics(b.dataset.dia === "on")));
}

function setDiacritics(on) {
  state.diacritics = !!on;
  try { localStorage.setItem("urdu.diacritics", on ? "on" : "off"); } catch (e) {}
  $$("#diacritics-toggle button").forEach((b) => {
    const sel = (b.dataset.dia === "on") === state.diacritics;
    b.classList.toggle("on", sel);
    b.setAttribute("aria-pressed", sel ? "true" : "false");
  });
  refreshRomanFields();
}

/* ---------- runtime config ---------- */
async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (res.ok) Object.assign(state.config, await res.json());
  } catch (e) { /* keep the local defaults */ }
  applyConfig();
}

/* Reflect the server flags in the UI: the image queue shows the per-run cap that
   comes back from /api/config (30 locally, fewer on Vercel). */
function applyConfig() {
  renderQueue();
}

/* ---------- providers ---------- */
async function loadProviders() {
  try {
    const res = await fetch("/api/providers");
    state.providers = await res.json();
  } catch (e) {
    $("#ocr-engines").innerHTML = '<p class="err">Could not load engines: ' + esc(e.message) + "</p>";
    return;
  }
  renderEngineList("ocr", "#ocr-engines");
  renderEngineList("translit", "#translit-engines");
  // Start with a working default so the page is usable without touching engines:
  // the recommended engine, else the first one that's actually available.
  ["ocr", "translit"].forEach((c) => {
    if (!checkedIds(c).length) pick(c + ":recommended");
    if (!checkedIds(c).length) checkFirstAvailable(c);
  });
}

/* Tick the first engine in the list that isn't disabled — the fallback when
   no recommended engine can run (no API keys, model not downloaded, …). */
function checkFirstAvailable(cap) {
  const box = $$(`input[data-cap="${cap}"]`).find((b) => !b.disabled);
  if (box) box.checked = true;
}

/* `capAttr` overrides the value written to each checkbox's data-cap; kept as a
   hook for reusing the same engine list under a second picker. */
function renderEngineList(cap, sel, capAttr = cap) {
  const box = $(sel);
  const rows = state.providers[cap] || [];
  if (!rows.length) { box.innerHTML = '<p class="dim">No engines found for this step.</p>'; return; }
  box.innerHTML = rows.map((r) => {
    const dis = r.available ? "" : "disabled";
    const off = r.available ? "" : "off";
    const reason = r.available ? "" : ` title="${esc(r.reason || "unavailable")}"`;
    return `<label class="eng ${off}"${reason}>
      <input type="checkbox" value="${esc(r.id)}" data-cap="${capAttr}" ${dis}>
      <span>
        <span class="lbl">${esc(r.label)}</span><span class="badge-tag">${esc(r.badge)}</span>
        ${r.price ? `<span class="price-tag">${esc(r.price)}</span>` : ""}
        ${r.note ? `<span class="note">${esc(r.note)}</span>` : ""}
        ${r.available ? "" : `<span class="note">${esc(r.reason || "")}</span>`}
      </span>
    </label>`;
  }).join("");
}

/* Read-only "what each engine costs" table inside the Settings modal. */
function renderSettingsEngines() {
  const box = $("#settings-engines");
  if (!box) return;
  const groups = [["ocr", "OCR"], ["translit", "Transliterate"]];
  box.innerHTML = groups.map(([cap, title]) => {
    const rows = state.providers[cap] || [];
    if (!rows.length) return "";
    return `<div class="eng-price-group"><strong>${esc(title)}</strong>` +
      rows.map((r) => `<div class="eng-price-row"><span>${esc(r.label)}</span>` +
        `<span class="price-tag">${esc(r.price || "—")}</span></div>`).join("") +
      `</div>`;
  }).join("");
}

function checkedIds(cap) {
  return $$(`input[data-cap="${cap}"]:checked`).map((c) => c.value);
}

function pick(spec) {
  const [cap, what] = spec.split(":");
  $$(`input[data-cap="${cap}"]`).forEach((b) => {
    if (b.disabled) { b.checked = false; return; }
    if (what === "none") b.checked = false;
    else if (what === "offline") {
      const row = (state.providers[cap] || []).find((r) => r.id === b.value);
      b.checked = !!row && row.badge === "offline";
    } else if (what === "recommended") {
      b.checked = (RECOMMENDED[cap] || []).includes(b.value);
    }
  });
}

/* ---------- image reader (one image or many) ---------- */
function wireDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");
  input.addEventListener("change", () => { addImages(input.files); input.value = ""; });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    addImages(e.dataTransfer.files);
  });
}

/* Per-run cap from GET /api/config — 30 locally, fewer on Vercel where the
   request body is capped at ~4.5 MB. */
const batchMax = () => state.config.batch_max_files || 30;

function addImages(fileList) {
  for (const f of fileList) {
    if (!f.type.startsWith("image/")) continue;
    if (state.images.length >= batchMax()) break;
    if (state.images.some((x) => x.name === f.name && x.size === f.size)) continue;
    state.images.push(f);
  }
  renderQueue();
}

/* Draw the queue under the dropzone. One image: show its preview and keep the
   familiar single-image feel. Many: hide the preview, list the files with a
   remove button each. The Run button follows whether anything is queued. */
function renderQueue() {
  const box = $("#image-queue");
  const label = $("#drop-label");
  const preview = $("#preview");
  const n = state.images.length;
  $("#run-ocr").disabled = n === 0;

  if (!n) {
    box.innerHTML = '<p class="dim">No images added.</p>';
    if (preview) { preview.src = ""; preview.style.display = "none"; }
    if (label) label.innerHTML =
      `<strong>Click or drop</strong> image(s) (JPG / PNG) — up to ${batchMax()}`;
    return;
  }

  if (n === 1 && preview) {
    const rd = new FileReader();
    rd.onload = () => { preview.src = rd.result; preview.style.display = "block"; };
    rd.readAsDataURL(state.images[0]);
    if (label) label.innerHTML =
      "<strong>" + esc(state.images[0].name) + "</strong> — click to add more";
  } else {
    if (preview) { preview.src = ""; preview.style.display = "none"; }
    if (label) label.innerHTML =
      `<strong>Click or drop</strong> to add more (JPG / PNG) — up to ${batchMax()}`;
  }

  box.innerHTML =
    `<p class="dim">${n} image${n === 1 ? "" : "s"} ready` +
    (n >= batchMax() ? ` (max ${batchMax()})` : "") + `</p>` +
    state.images.map((f, i) =>
      `<div class="batch-file"><span>${esc(f.name)}</span>` +
      `<button type="button" data-rm="${i}">&times;</button></div>`).join("");
  $$("#image-queue [data-rm]").forEach((b) => b.addEventListener("click", () => {
    state.images.splice(Number(b.dataset.rm), 1);
    renderQueue();
  }));
}

/* ---------- read image(s) ---------- */
/* One image -> POST /api/ocr, rich side-by-side cards with an editable Urdu box
   and a "use this text" button that hands it to step 2. Two or more ->
   POST /api/batch, one table row per (file x engine) you can export as CSV. */
async function runOcr() {
  const ids = checkedIds("ocr");
  if (!state.images.length) return alert("Add an image first.");
  if (!ids.length) return alert("Tick at least one OCR engine.");
  if (state.images.length === 1) return runSingle(state.images[0], ids);
  return runMany(state.images, ids);
}

async function runSingle(file, ids) {
  $("#batch-results").hidden = true;
  $("#ocr-progress").hidden = true;
  const cols = $("#ocr-columns");
  cols.hidden = false;
  cols.innerHTML = "";
  ids.forEach((id) => cols.appendChild(ocrColShell(id)));
  $("#run-ocr").disabled = true;

  try {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("providers", ids.join(","));
    const res = await fetch("/api/ocr", { method: "POST", body: fd });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).error || msg; } catch (e) {}
      throw new Error(msg);
    }
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
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.done) continue;
        fillOcrCol(payload);
      }
    }
    revealTranslit();
  } catch (e) {
    alert("OCR failed: " + e.message);
  } finally {
    $("#run-ocr").disabled = false;
  }
}

async function runMany(files, ocrIds) {
  const cols = $("#ocr-columns");
  cols.hidden = true;
  cols.innerHTML = "";
  const tbody = $("#batch-results tbody");
  tbody.innerHTML = "";
  state.batchRows = [];
  $("#batch-results").hidden = false;
  $("#batch-csv").hidden = true;
  $("#run-ocr").disabled = true;
  const prog = $("#ocr-progress");
  prog.hidden = false;
  prog.textContent = `Processing 0 / ${files.length}…`;

  try {
    const fd = new FormData();
    files.forEach((f) => fd.append("images", f, f.name));
    fd.append("ocr_providers", ocrIds.join(","));
    fd.append("translit_providers", "");   // step 1 is OCR only — step 2 does transliteration
    const res = await fetch("/api/batch", { method: "POST", body: fd });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).error || msg; } catch (e) {}
      throw new Error(msg);
    }
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
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.done) continue;
        if (payload.progress) {
          prog.textContent = `Processing ${payload.progress} / ${payload.total}…`;
          continue;
        }
        addBatchRow(payload);
      }
    }
    prog.textContent = `Done — ${state.batchRows.length} row${state.batchRows.length === 1 ? "" : "s"}.`;
    $("#batch-csv").hidden = state.batchRows.length === 0;
  } catch (e) {
    prog.textContent = "Run failed: " + e.message;
  } finally {
    $("#run-ocr").disabled = false;
  }
}

function labelFor(cap, id) {
  const row = (state.providers[cap] || []).find((r) => r.id === id);
  return row ? row.label : id;
}

function ocrColShell(id) {
  const el = document.createElement("div");
  el.className = "col";
  el.dataset.pid = id;
  el.innerHTML = `<h3><span>${esc(labelFor("ocr", id))}</span><span class="ms">…</span></h3>
    <div class="running">running…</div>`;
  return el;
}

function fillOcrCol(row) {
  const el = $(`#ocr-columns .col[data-pid="${cssEsc(row.provider_id)}"]`);
  if (!el) return;
  if (!row.ok) {
    el.classList.add("is-error");
    el.innerHTML = `<h3><span>${esc(labelFor("ocr", row.provider_id))}</span></h3>` +
      `<div class="err">${esc(row.error || "failed")}</div>`;
    return;
  }
  el.innerHTML = "";

  const ta = document.createElement("textarea");
  ta.dir = "rtl";
  ta.value = row.text || "";
  const field = elWith("div", "field field--urdu");
  const head = elWith("div", "field-head");
  head.appendChild(elWith("span", "lab", "Urdu text"));
  head.appendChild(copyBtn(row.text || ""));
  field.appendChild(head);
  field.appendChild(ta);
  el.appendChild(field);

  if (row.notes) el.appendChild(elWith("div", "foot", "note: " + row.notes));
  if (row.roman || row.roman_diacritic)
    el.appendChild(romanResultField(row.roman, row.roman_diacritic));
  if (row.devanagari) el.appendChild(resultField("Devanagari", row.devanagari, false));

  const use = document.createElement("button");
  use.className = "use-btn";
  use.textContent = "Use this text ↓";
  use.addEventListener("click", () => {
    $("#urdu-input").value = ta.value;
    checkedIds("translit").length || pick("translit:recommended");
    runTranslit();
    revealTranslit(true);
  });
  el.appendChild(use);
  el.appendChild(engineFoot("ocr", row.provider_id, row.ms));
}

/* ---------- transliteration ---------- */
function revealTranslit(scroll) {
  $("#translit-card").hidden = false;
  if (scroll) $("#translit-card").scrollIntoView({ behavior: "smooth" });
}

async function runTranslit() {
  const text = $("#urdu-input").value.trim();
  const ids = checkedIds("translit");
  if (!text) return alert("Nothing to transliterate.");
  if (!ids.length) return alert("Tick at least one transliteration engine.");

  const cols = $("#translit-columns");
  cols.innerHTML = ids.map((id) =>
    `<div class="col" data-pid="${esc(id)}"><h3><span>${esc(labelFor("translit", id))}</span>` +
    `<span class="ms">…</span></h3><div class="running">running…</div></div>`).join("");
  $("#run-translit").disabled = true;

  try {
    const res = await fetch("/api/transliterate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, providers: ids }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    for (const row of data.results || []) fillTranslitCol(row);
  } catch (e) {
    alert("Transliteration failed: " + e.message);
  } finally {
    $("#run-translit").disabled = false;
  }
}

function fillTranslitCol(row) {
  const el = $(`#translit-columns .col[data-pid="${cssEsc(row.provider_id)}"]`);
  if (!el) return;
  if (!row.ok) {
    el.classList.add("is-error");
    el.innerHTML = `<h3><span>${esc(labelFor("translit", row.provider_id))}</span></h3>` +
      `<div class="err">${esc(row.error || "failed")}</div>`;
    return;
  }
  el.innerHTML = "";
  el.appendChild(romanResultField(row.roman, row.roman_diacritic));
  el.appendChild(resultField("Devanagari", row.devanagari || "—", false));
  el.appendChild(engineFoot("translit", row.provider_id, row.ms));
}

/* A labelled result block: label + copy button on one row, the text below.
   `roman` picks the serif Roman styling over the Devanagari font. */
function resultField(label, text, roman) {
  const wrap = elWith("div", "field " + (roman ? "field--roman" : "field--deva"));
  const head = elWith("div", "field-head");
  head.appendChild(elWith("span", "lab", label));
  head.appendChild(copyBtn(text === "—" ? "" : text));
  const out = document.createElement("div");
  out.className = "out" + (roman ? " roman" : "");
  out.textContent = text;
  wrap.appendChild(head);
  wrap.appendChild(out);
  return wrap;
}

/* Roman block carrying BOTH spellings (plain + diacritic) on the .out element.
   The Plain/Diacritics toggle repaints these via refreshRomanFields() — no
   re-fetch. */
function romanResultField(plain, dia) {
  const wrap = elWith("div", "field field--roman");
  const head = elWith("div", "field-head");
  head.appendChild(elWith("span", "lab", "Roman"));
  head.appendChild(copyBtn(""));
  const out = document.createElement("div");
  out.className = "out roman";
  out.dataset.plain = plain || "";
  out.dataset.dia = dia || "";
  wrap.appendChild(head);
  wrap.appendChild(out);
  paintRoman(out);
  return wrap;
}

function paintRoman(out) {
  const txt = (state.diacritics ? out.dataset.dia : out.dataset.plain)
    || out.dataset.plain || out.dataset.dia || "—";
  out.textContent = txt;
  const copy = out.parentElement && out.parentElement.querySelector(".copy-btn");
  if (copy) setCopyPayload(copy, txt === "—" ? "" : txt);
}

/* Repaint every Roman block on the page for the current toggle state. */
function refreshRomanFields() { $$(".out.roman").forEach(paintRoman); }

function engineFoot(cap, id, ms) {
  const d = elWith("div", "foot");
  d.textContent = "via " + labelFor(cap, id) + (ms != null ? "  ·  " + ms + " ms" : "");
  return d;
}

/* ---------- multi-image results table + CSV ---------- */
const BATCH_COLS = ["file", "ocr_engine", "urdu", "ms", "error"];

function addBatchRow(row) {
  state.batchRows.push(row);
  const tr = document.createElement("tr");
  tr.innerHTML = BATCH_COLS.map((c) => {
    const v = row[c] == null ? "" : String(row[c]);
    const rtl = c === "urdu" ? ' dir="rtl"' : "";
    const cls = c === "error" && v ? ' class="err"' : "";
    return `<td${rtl}${cls}>${esc(v)}</td>`;
  }).join("");
  $("#batch-results tbody").appendChild(tr);
}

function downloadBatchCsv() {
  const cell = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const lines = [BATCH_COLS.join(",")];
  for (const r of state.batchRows) lines.push(BATCH_COLS.map((c) => cell(r[c])).join(","));
  // BOM so Excel reads the Urdu column as UTF-8
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "urdu-ocr.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ---------- settings ---------- */
function wireSettings() {
  const modal = $("#settings-modal");
  $("#settings-btn").addEventListener("click", async () => {
    const status = await (await fetch("/api/settings")).json();
    const ro = state.config.settings_readonly;
    $("#settings-fields").innerHTML = Object.keys(status).map((k) =>
      `<label>${esc(k)} <span class="dim">${status[k] ? "(set)" : "(not set)"}</span></label>
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
    $$("#settings-fields input").forEach((i) => {
      if (i.value.trim()) body[i.dataset.key] = i.value.trim();
    });
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

/* ---------- tiny helpers ---------- */
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function cssEsc(s) { return String(s).replace(/["\\]/g, "\\$&"); }
function elWith(tag, cls, text) {
  const e = document.createElement(tag);
  e.className = cls;
  e.textContent = text;
  return e;
}
/* Copy button. The payload lives on the element (b._payload) and is read at
   click time, so a Roman block can update it when the diacritics toggle flips
   without rebuilding the button. */
function copyBtn(text) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "copy-btn";
  b.textContent = "Copy";
  setCopyPayload(b, text);
  b.addEventListener("click", () => {
    if (!b._payload) return;
    navigator.clipboard.writeText(b._payload).then(() => {
      b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200);
    });
  });
  return b;
}
function setCopyPayload(b, text) {
  b._payload = text || "";
  b.disabled = !b._payload;
}
