/* Urdu Toolkit front-end. No dependencies. Talks to:
   GET  /api/providers        -> engine lists per capability
   POST /api/ocr              -> SSE stream, one row per engine
   POST /api/transliterate    -> {results:[...]}
   POST /api/render           -> image/png
   GET/POST /api/settings     -> API-key status / save
*/
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const RECOMMENDED = {
  ocr: ["surya", "easyocr", "rapidocr"],
  translit: ["aksharamukha", "uroman", "rule"],
};

const state = {
  file: null,
  providers: { ocr: [], translit: [], translate: [], render: [] },
  lastTranslit: [], // rows from the most recent /api/transliterate run
};

/* ---------- boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  wireTabs();
  wireDropzone();
  wireSettings();
  $("#run-ocr").addEventListener("click", runOcr);
  $("#run-translit").addEventListener("click", runTranslit);
  $("#run-render").addEventListener("click", runRender);
  $("#use-paste").addEventListener("click", usePaste);
  $$("[data-pick]").forEach((b) => b.addEventListener("click", () => pick(b.dataset.pick)));
  loadProviders();
});

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
  renderRenderSelect();
}

function renderEngineList(cap, sel) {
  const box = $(sel);
  const rows = state.providers[cap] || [];
  if (!rows.length) { box.innerHTML = '<p class="dim">No engines found for this step.</p>'; return; }
  box.innerHTML = rows.map((r) => {
    const dis = r.available ? "" : "disabled";
    const off = r.available ? "" : "off";
    const reason = r.available ? "" : ` title="${esc(r.reason || "unavailable")}"`;
    return `<label class="eng ${off}"${reason}>
      <input type="checkbox" value="${esc(r.id)}" data-cap="${cap}" ${dis}>
      <span>
        <span class="lbl">${esc(r.label)}</span><span class="badge-tag">${esc(r.badge)}</span>
        ${r.note ? `<span class="note">${esc(r.note)}</span>` : ""}
        ${r.available ? "" : `<span class="note">${esc(r.reason || "")}</span>`}
      </span>
    </label>`;
  }).join("");
}

function renderRenderSelect() {
  const s = $("#render-provider");
  const rows = (state.providers.render || []).filter((r) => r.available);
  if (!rows.length) {
    s.innerHTML = '<option value="">none available</option>';
    return;
  }
  s.innerHTML = rows.map((r) => `<option value="${esc(r.id)}">${esc(r.label)}</option>`).join("");
}

function checkedIds(cap) {
  return $$(`input[data-cap="${cap}"]:checked`).map((c) => c.value);
}

function pick(spec) {
  const [cap, what] = spec.split(":");
  const boxes = $$(`input[data-cap="${cap}"]`);
  boxes.forEach((b) => {
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
  if (state.file) $("#render-card").hidden = false;
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
    state.lastTranslit = data.results || [];
    for (const row of state.lastTranslit) fillTranslitCol(row);
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

/* ---------- render ---------- */
async function runRender() {
  if (!state.file) return alert("Redraw needs the image you ran OCR on.");
  const pid = $("#render-provider").value;
  if (!pid) return alert("No redraw engine is available.");
  const script = $('input[name="render-script"]:checked').value;
  const ok = state.lastTranslit.find((r) => r.ok && (script === "roman" ? r.roman : r.devanagari));
  if (!ok) return alert("Run a transliteration first — the redraw uses that text.");
  const lines = script === "roman" ? ok.roman : ok.devanagari;

  const status = $("#render-status");
  status.hidden = false; status.textContent = "redrawing… this can take a while for AI engines.";
  $("#run-render").disabled = true;
  $("#rendered").hidden = true;
  $("#download-render").hidden = true;

  try {
    const fd = new FormData();
    fd.append("image", state.file);
    fd.append("provider", pid);
    fd.append("lines", lines);
    const res = await fetch("/api/render", { method: "POST", body: fd });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).error || msg; } catch (e) {}
      throw new Error(msg);
    }
    const url = URL.createObjectURL(await res.blob());
    const img = $("#rendered");
    img.src = url; img.hidden = false;
    const dl = $("#download-render");
    dl.href = url; dl.download = "urdu-" + script + ".png"; dl.hidden = false;
    status.hidden = true;
  } catch (e) {
    status.textContent = "Redraw failed: " + e.message;
  } finally {
    $("#run-render").disabled = false;
  }
}

/* ---------- paste text ---------- */
function usePaste() {
  const t = $("#paste-box").value.trim();
  if (!t) return;
  $("#urdu-input").value = t;
  state.file = null;
  $("#render-card").hidden = true; // no source image to redraw
  revealTranslit(true);
}

/* ---------- settings ---------- */
function wireSettings() {
  const modal = $("#settings-modal");
  $("#settings-btn").addEventListener("click", async () => {
    const status = await (await fetch("/api/settings")).json();
    $("#settings-fields").innerHTML = Object.keys(status).map((k) =>
      `<label>${esc(k)} <span class="dim">${status[k] ? "(set)" : "(not set)"}</span></label>
       <input type="${k.endsWith("KEY") ? "password" : "text"}" data-key="${esc(k)}"
              placeholder="${status[k] ? "leave blank to keep" : ""}">`).join("");
    $("#settings-status").textContent = "";
    modal.hidden = false;
  });
  $("#close-settings").addEventListener("click", () => (modal.hidden = true));
  $("#save-settings").addEventListener("click", async () => {
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
