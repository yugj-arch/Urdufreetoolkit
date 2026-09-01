/* Urdu Toolkit front-end. No dependencies. Talks to:
   GET  /api/config           -> runtime flags (batch cap, settings read-only)
   GET  /api/providers        -> engine lists per capability
   POST /api/ocr              -> SSE stream, one row per engine
   POST /api/transliterate    -> {results:[...]}
   POST /api/batch            -> SSE stream, one row per (file x ocr x translit)
   GET/POST /api/settings     -> API-key status / save
*/
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const RECOMMENDED = {
  ocr: ["gpt", "gcv"],
  translit: ["gpt", "rule", "aksharamukha"],
};

const state = {
  file: null,
  providers: { ocr: [], translit: [] },
  batchFiles: [], // File[] queued in the Batch tab
  batchRows: [],  // accumulated result rows, for the CSV download
  config: { on_vercel: false, batch_max_files: 30, settings_readonly: false },
};

/* ---------- boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  wireTabs();
  wireDropzone();
  wireSettings();
  wireBatch();
  $("#run-ocr").addEventListener("click", runOcr);
  $("#run-translit").addEventListener("click", runTranslit);
  $("#use-paste").addEventListener("click", usePaste);
  $$("[data-pick]").forEach((b) => b.addEventListener("click", () => pick(b.dataset.pick)));
  loadConfig();
  loadProviders();
});

/* ---------- runtime config ---------- */
async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (res.ok) Object.assign(state.config, await res.json());
  } catch (e) { /* keep the local defaults */ }
  applyConfig();
}

/* Reflect the server flags in the UI: shrink the Batch cap hint, and when the
   host filesystem is read-only (Vercel) tell the user keys live in env vars. */
function applyConfig() {
  const hint = $("#batch-drop-hint");
  if (hint) hint.innerHTML =
    `<strong>Click or drop</strong> images (JPG / PNG) — up to ${batchMax()}`;
  renderBatchQueue();
}

function wireTabs() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    if (t.disabled) return;
    $$(".tab").forEach((x) => x.classList.remove("active"));
    $$(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#panel-" + t.dataset.tab).classList.add("active");
  }));
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
  renderEngineList("ocr", "#batch-ocr-engines", "batch-ocr");
  renderEngineList("translit", "#batch-translit-engines", "batch-translit");
}

/* `capAttr` overrides the value written to each checkbox's data-cap — the Batch
   tab passes "batch-ocr" / "batch-translit" so its picks stay separate from the
   single-image pickers that share the same `cap`. */
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
  const baseCap = cap.replace(/^batch-/, ""); // "batch-ocr" -> "ocr" for the lookups below
  const boxes = $$(`input[data-cap="${cap}"]`);
  boxes.forEach((b) => {
    if (b.disabled) { b.checked = false; return; }
    if (what === "none") b.checked = false;
    else if (what === "offline") {
      const row = (state.providers[baseCap] || []).find((r) => r.id === b.value);
      b.checked = !!row && row.badge === "offline";
    } else if (what === "recommended") {
      b.checked = (RECOMMENDED[baseCap] || []).includes(b.value);
    }
  });
}

/* ---------- dropzone ---------- */
function wireDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");
  input.addEventListener("change", (e) => e.target.files[0] && handleFile(e.target.files[0]));
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function handleFile(file) {
  state.file = file;
  const img = $("#preview");
  const rd = new FileReader();
  rd.onload = () => { img.src = rd.result; img.style.display = "block"; };
  rd.readAsDataURL(file);
  $("#drop-label").innerHTML = "<strong>" + esc(file.name) + "</strong> — click to change";
  $("#run-ocr").disabled = false;
}

/* ---------- OCR (SSE) ---------- */
async function runOcr() {
  const ids = checkedIds("ocr");
  if (!state.file) return alert("Upload an image first.");
  if (!ids.length) return alert("Tick at least one OCR engine.");

  const cols = $("#ocr-columns");
  cols.innerHTML = "";
  ids.forEach((id) => cols.appendChild(ocrColShell(id)));
  $("#run-ocr").disabled = true;

  try {
    const fd = new FormData();
    fd.append("image", state.file);
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
  el.querySelector(".ms").textContent = row.ms != null ? row.ms + " ms" : "";
  if (!row.ok) {
    el.innerHTML = el.querySelector("h3").outerHTML + `<div class="err">${esc(row.error || "failed")}</div>`;
    return;
  }
  const ta = document.createElement("textarea");
  ta.dir = "rtl";
  ta.value = row.text || "";
  const use = document.createElement("button");
  use.textContent = "Use this text";
  use.addEventListener("click", () => {
    $("#urdu-input").value = ta.value;
    revealTranslit(true);
  });
  el.innerHTML = el.querySelector("h3").outerHTML;
  el.appendChild(elWith("div", "lab", "Urdu"));
  el.appendChild(ta);
  if (row.notes) el.appendChild(elWith("div", "running", "note: " + row.notes));
  if (row.devanagari || row.roman) {
    el.appendChild(elWith("div", "lab", "translit (from this engine)"));
    if (row.devanagari) el.appendChild(outBlock(row.devanagari, false));
    if (row.roman) el.appendChild(outBlock(row.roman, true));
  }
  el.appendChild(use);
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
      body: JSON.stringify({ text, providers: ids, roman_style: $("#roman-style").value }),
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
  const head = el.querySelector("h3").outerHTML.replace(">…<", ">" + (row.ms ?? "") + " ms<");
  if (!row.ok) { el.innerHTML = head + `<div class="err">${esc(row.error || "failed")}</div>`; return; }
  el.innerHTML = head;
  el.appendChild(elWith("div", "lab", "Devanagari"));
  el.appendChild(outBlock(row.devanagari || "—", false));
  el.appendChild(copyBtn(row.devanagari || ""));
  el.appendChild(elWith("div", "lab", "Roman"));
  el.appendChild(outBlock(row.roman || "—", true));
  el.appendChild(copyBtn(row.roman || ""));
}

/* ---------- paste text ---------- */
function usePaste() {
  const t = $("#paste-box").value.trim();
  if (!t) return;
  $("#urdu-input").value = t;
  state.file = null;
  revealTranslit(true);
}

/* ---------- batch (SSE) ---------- */
function wireBatch() {
  const input = $("#batch-file-input");
  input.addEventListener("change", () => { addBatchFiles(input.files); input.value = ""; });
  const dz = $("#batch-dropzone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    addBatchFiles(e.dataTransfer.files);
  });
  $("#run-batch").addEventListener("click", runBatch);
  $("#batch-csv").addEventListener("click", downloadBatchCsv);
}

/* Batch cap comes from GET /api/config — 30 locally, fewer on Vercel where the
   request body is capped at ~4.5 MB. */
const batchMax = () => state.config.batch_max_files || 30;

function addBatchFiles(fileList) {
  for (const f of fileList) {
    if (!f.type.startsWith("image/")) continue;
    if (state.batchFiles.length >= batchMax()) break;
    if (state.batchFiles.some((x) => x.name === f.name && x.size === f.size)) continue;
    state.batchFiles.push(f);
  }
  renderBatchQueue();
}

function renderBatchQueue() {
  const box = $("#batch-queue");
  const n = state.batchFiles.length;
  $("#run-batch").disabled = n === 0;
  if (!n) { box.innerHTML = '<p class="dim">No images queued.</p>'; return; }
  box.innerHTML =
    `<p class="dim">${n} image${n === 1 ? "" : "s"} queued` +
    (n >= batchMax() ? ` (max ${batchMax()})` : "") + `</p>` +
    state.batchFiles.map((f, i) =>
      `<div class="batch-file"><span>${esc(f.name)}</span>` +
      `<button type="button" data-rm="${i}">&times;</button></div>`).join("");
  $$("#batch-queue [data-rm]").forEach((b) => b.addEventListener("click", () => {
    state.batchFiles.splice(Number(b.dataset.rm), 1);
    renderBatchQueue();
  }));
}

async function runBatch() {
  const ocrIds = checkedIds("batch-ocr");
  const trIds = checkedIds("batch-translit");
  if (!state.batchFiles.length) return alert("Queue at least one image.");
  if (!ocrIds.length) return alert("Tick at least one OCR engine.");

  const tbody = $("#batch-results tbody");
  tbody.innerHTML = "";
  state.batchRows = [];
  $("#batch-results").hidden = false;
  $("#batch-csv").hidden = true;
  $("#run-batch").disabled = true;
  const prog = $("#batch-progress");
  prog.hidden = false;
  prog.textContent = `Processing 0 / ${state.batchFiles.length}…`;

  try {
    const fd = new FormData();
    state.batchFiles.forEach((f) => fd.append("images", f, f.name));
    fd.append("ocr_providers", ocrIds.join(","));
    fd.append("translit_providers", trIds.join(","));
    fd.append("roman_style", $("#batch-roman-style").value);
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
    prog.textContent = "Batch failed: " + e.message;
  } finally {
    $("#run-batch").disabled = false;
  }
}

const BATCH_COLS = ["file", "ocr_engine", "urdu", "translit_engine", "devanagari", "roman", "ms", "error"];

function addBatchRow(row) {
  state.batchRows.push(row);
  const tr = document.createElement("tr");
  tr.innerHTML = BATCH_COLS.map((c) => {
    const v = row[c] == null ? "" : String(row[c]);
    const rtl = (c === "urdu" || c === "devanagari") ? ' dir="rtl"' : "";
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
  // BOM so Excel reads the Urdu/Devanagari columns as UTF-8
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "urdu-batch.csv";
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
function outBlock(text, roman) {
  const e = document.createElement("div");
  e.className = "out" + (roman ? " roman" : "");
  e.textContent = text;
  return e;
}
function copyBtn(text) {
  const b = document.createElement("button");
  b.textContent = "Copy";
  b.addEventListener("click", () => {
    navigator.clipboard.writeText(text).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200); });
  });
  return b;
}
