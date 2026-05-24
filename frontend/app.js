const STATE_URL = "/api/voice/state";
const ENROLL_URL = "/api/voice/enroll";
const CONTEXT_URL = "/api/voice/context";
const HISTORY_URL = "/api/voice/history";
const POLL_MS = 350;

const verdictEl = document.getElementById("verdict");
const labelEl = verdictEl.querySelector(".label");
const barWrap = document.getElementById("utt-bar-wrap");
const bar = document.getElementById("utt-bar");

const metricsEl = document.getElementById("metrics");
const mEnrolled = document.getElementById("m-enrolled");
const mTasks = document.getElementById("m-tasks");
const mMin = document.getElementById("m-min");
const mTau = document.getElementById("m-tau");

const histEmpty = document.getElementById("hist-empty");
const histList = document.getElementById("hist-list");
const histCount = document.getElementById("hist-count");
const histClear = document.getElementById("hist-clear");

const enrollBtn = document.getElementById("enroll-btn");
const enrollReset = document.getElementById("enroll-reset");
const enrollHint = document.getElementById("enroll-hint");

const tasksEl = document.getElementById("tasks");
const tasksSave = document.getElementById("tasks-save");
const tasksStatus = document.getElementById("tasks-status");

let lastHistTop = null;

function setVerdictClass(...classes) {
  verdictEl.classList.remove("speaking", "processing", "enrolling", "error", "pending");
  for (const c of classes) verdictEl.classList.add(c);
}

function renderState(s) {
  // Handle error reasons up front
  const reason = s.reason || "";
  if (reason.startsWith("sounddevice_unavailable")) {
    setVerdictClass("error");
    labelEl.textContent = "sounddevice not installed";
    return;
  }
  if (reason.startsWith("mic_open_failed")) {
    setVerdictClass("error");
    labelEl.textContent = "mic unavailable — grant permission";
    return;
  }
  if (reason.startsWith("model_load_failed")) {
    setVerdictClass("error");
    labelEl.textContent = "model load failed — check terminal";
    return;
  }
  if (reason === "loading_models") {
    setVerdictClass("pending");
    labelEl.textContent = "loading models (first run downloads ~250 MB)…";
    return;
  }
  if (s.enroll_active) {
    setVerdictClass("enrolling");
    labelEl.textContent = `enrolling… ${Math.round(s.enroll_progress * 100)}%`;
    barWrap.hidden = false;
    bar.style.width = `${s.enroll_progress * 100}%`;
    return;
  }
  if (s.speaking) {
    setVerdictClass("speaking");
    const min = s.thresholds?.min_utterance_ms || 5000;
    const cur = s.current_utterance_ms;
    const gate = cur >= min ? "✓ ≥ min" : `${Math.max(0, min - cur)} ms to min`;
    labelEl.textContent = `speaking · ${cur} ms · ${gate}`;
    barWrap.hidden = false;
    bar.style.width = `${Math.min(100, (cur / (min * 1.5)) * 100)}%`;
    return;
  }
  if (s.worker_busy || s.queue_depth > 0) {
    setVerdictClass("processing");
    const q = s.queue_depth > 0 ? ` · ${s.queue_depth} queued` : "";
    labelEl.textContent = `processing utterance…${q}`;
    barWrap.hidden = true;
    return;
  }
  setVerdictClass("pending");
  labelEl.textContent = "listening — say something";
  barWrap.hidden = true;
  bar.style.width = "0%";
}

function renderMetrics(s) {
  metricsEl.hidden = false;
  mEnrolled.textContent = s.enrolled ? "yes" : "no";
  mTasks.textContent = String(s.context_count ?? 0);
  mMin.textContent = `${s.thresholds?.min_utterance_ms ?? "—"} ms`;
  mTau.textContent = s.thresholds?.relevance?.toFixed?.(2) ?? "—";
}

function renderEnroll(s) {
  if (s.enrolled) {
    enrollBtn.textContent = "Enrolled ✓";
    enrollBtn.disabled = true;
    enrollReset.hidden = false;
    enrollHint.textContent = "Speaker verification is active.";
  } else {
    enrollBtn.textContent = "Enroll my voice";
    enrollBtn.disabled = !!s.enroll_active;
    enrollReset.hidden = true;
    enrollHint.textContent =
      "Record ~6 seconds of your voice so we can tell it apart from coworkers and background voices.";
  }
}

const VERDICT_CLASS = {
  ON_TASK: "on-task",
  OFF_TASK: "off-task",
  NOT_USER: "not-user",
  TOO_SHORT: "too-short",
  NO_CONTEXT: "no-context",
  ERROR: "off-task",
};

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

function utteranceCard(u) {
  const li = document.createElement("li");
  li.className = "hist-item";

  const verdictCls = VERDICT_CLASS[u.verdict] || "";
  const spk = u.speaker_score != null ? u.speaker_score.toFixed(3) : "—";
  const rel = u.relevance != null ? u.relevance.toFixed(3) : "—";
  const user = u.is_user ? "yes" : "no";

  li.innerHTML = `
    <div class="hist-head">
      <span class="badge ${verdictCls}">${u.verdict.replace(/_/g, " ").toLowerCase()}</span>
      <span class="small muted">${fmtTime(u.at)} · ${u.duration_ms} ms</span>
    </div>
    <p class="transcript"></p>
    <dl class="metrics tight">
      <dt>is_user</dt><dd>${user}</dd>
      <dt>speaker</dt><dd>${spk}</dd>
      <dt>relevance</dt><dd>${rel}</dd>
    </dl>
  `;
  // Set transcript via textContent to avoid HTML injection
  li.querySelector(".transcript").textContent = u.transcript || "(no transcript)";
  return li;
}

function renderHistory(history) {
  histCount.textContent = history.length ? `(${history.length})` : "";

  if (!history.length) {
    histEmpty.hidden = false;
    histList.hidden = true;
    histList.innerHTML = "";
    lastHistTop = null;
    return;
  }

  histEmpty.hidden = true;
  histList.hidden = false;

  // Only re-render if the top entry changed (new utterance arrived) or
  // count differs (cleared / trimmed).
  const topKey = history[0].at;
  if (topKey === lastHistTop && histList.children.length === history.length) return;
  lastHistTop = topKey;

  histList.innerHTML = "";
  for (const u of history) histList.appendChild(utteranceCard(u));
}

async function pollState() {
  try {
    const r = await fetch(STATE_URL, { cache: "no-store" });
    if (!r.ok) return;
    const s = await r.json();
    renderState(s);
    renderMetrics(s);
    renderEnroll(s);
    renderHistory(s.history || []);
  } catch (_) {
    /* keep polling */
  }
}

enrollBtn.addEventListener("click", async () => {
  enrollBtn.disabled = true;
  try {
    await fetch(ENROLL_URL, { method: "POST" });
  } catch (_) {
    enrollBtn.disabled = false;
  }
});

enrollReset.addEventListener("click", async () => {
  await fetch(ENROLL_URL, { method: "DELETE" });
});

histClear.addEventListener("click", async () => {
  await fetch(HISTORY_URL, { method: "DELETE" });
});

tasksSave.addEventListener("click", async () => {
  tasksStatus.textContent = "saving…";
  const tasks = tasksEl.value.split("\n").map((t) => t.trim()).filter(Boolean);
  const r = await fetch(CONTEXT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tasks }),
  });
  if (r.ok) {
    const data = await r.json();
    tasksStatus.textContent = `saved · ${data.tasks.length} task(s)`;
    setTimeout(() => (tasksStatus.textContent = ""), 2200);
  } else {
    tasksStatus.textContent = "save failed";
  }
});

// Prefill textarea with whatever the server already has
(async () => {
  try {
    const r = await fetch(CONTEXT_URL);
    if (r.ok) {
      const data = await r.json();
      if (data.tasks?.length) tasksEl.value = data.tasks.join("\n");
    }
  } catch (_) {}
})();

setInterval(pollState, POLL_MS);
pollState();
