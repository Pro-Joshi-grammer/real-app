// ── Types ──
type State = "SETTINGS" | "CAPTURING" | "PROCESSING" | "ANSWER" | "ERROR";

interface ProviderConfig {
  type: "openai" | "gemini" | "bedrock";
  label: string;
  api_key?: string;
  base_url?: string;
  model: string;
  aws_access_key?: string;
  aws_secret_key?: string;
  region?: string;
  max_tokens?: number;
  timeout?: number;
}

interface TestResult {
  label: string;
  type: string;
  ok: boolean;
  latency_ms: number;
  error?: string;
}

// ── Provider presets ──
const PRESETS: Record<string, string> = {
  custom: "",
  fireworks: "https://api.fireworks.ai/inference/v1",
  nvapi: "https://integrate.api.nvidia.com/v1",
};

// ── State ──
let state: State = "SETTINGS";
let savedProviders: ProviderConfig[] = [];

// ── DOM refs ──
const $ = (id: string) => document.getElementById(id)!;

const settingsEl = $("settings");
const cameraView = $("camera-view");
const processingView = $("processing-view");
const answerView = $("answer-view");
const errorView = $("error-view");
const video = $("video") as HTMLVideoElement;
const canvas = $("canvas") as HTMLCanvasElement;
const answerText = $("answer-text");
const errorText = $("error-text");
const latencyDebug = $("latency-debug");

// settings elements
const providerList = $("provider-list");
const addBtn = $("add-provider-btn") as HTMLButtonElement;
const testBtn = $("test-btn") as HTMLButtonElement;
const saveBtn = $("save-btn") as HTMLButtonElement;
const testStatus = $("test-status");
const results = $("results");
const rankingList = $("ranking-list");
const settingsBtn = $("settings-btn");

// ── Camera ──
let stream: MediaStream | null = null;
let captureTimer: number | null = null;

const CAPTURE_INTERVAL = 2000; // 2s — down from 400ms
let captureInFlight = false;

// ── Dwell detection ──
let prevChecksum = 0;
const DWELL_THRESHOLD = 0.03; // 3% change needed to re-trigger

// ── Retry backoff ──
let consecutiveErrors = 0;
const MAX_CONSECUTIVE_ERRORS = 5;
const BACKOFF_MS = 30_000;

// ── Camera startup ──
async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();
  // Reset dwell & error state on fresh start
  prevChecksum = 0;
  consecutiveErrors = 0;
}

function stopCamera() {
  if (captureTimer !== null) {
    clearInterval(captureTimer);
    captureTimer = null;
  }

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  captureInFlight = false;
}

// ── Capture ──
const MAX_WIDTH = 1600;

function captureFrame(): string | null {
  if (!video.videoWidth || !video.videoHeight) return null;

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  let w = video.videoWidth;
  let h = video.videoHeight;

  if (w > MAX_WIDTH) {
    h = Math.round((h / w) * MAX_WIDTH);
    w = MAX_WIDTH;
  }

  canvas.width = w;
  canvas.height = h;

  ctx.drawImage(video, 0, 0, w, h);

  return canvas.toDataURL("image/jpeg", 0.88);
}

// ── Scene-change dwell detection ──
// Quick checksum over an 8x8 downsampled grid.
function frameChecksum(): number {
  const ctx = canvas.getContext("2d");
  if (!ctx) return 0;
  const w = canvas.width,
    h = canvas.height;
  const stepX = Math.max(1, Math.floor(w / 8));
  const stepY = Math.max(1, Math.floor(h / 8));
  let sum = 0;
  const data = ctx.getImageData(0, 0, w, h).data;
  for (let y = 0; y < h; y += stepY) {
    for (let x = 0; x < w; x += stepX) {
      const idx = (y * w + x) * 4;
      sum += data[idx] + data[idx + 1] + data[idx + 2];
    }
  }
  return sum;
}

function sceneChanged(): boolean {
  const cur = frameChecksum();
  if (prevChecksum === 0) {
    prevChecksum = cur;
    return true; // first frame always triggers
  }
  const diff = Math.abs(cur - prevChecksum) / Math.max(prevChecksum, 1);
  if (diff > DWELL_THRESHOLD) {
    prevChecksum = cur;
    return true;
  }
  return false;
}

// ── State transitions ──
function setState(s: State) {
  state = s;

  settingsEl.classList.toggle("hidden", s !== "SETTINGS");
  cameraView.classList.toggle("hidden", s !== "CAPTURING");
  processingView.classList.toggle("hidden", s !== "PROCESSING");
  answerView.classList.toggle("hidden", s !== "ANSWER");
  errorView.classList.toggle("hidden", s !== "ERROR");

  if (s === "CAPTURING") {
    startCaptureLoop();
  } else {
    stopCaptureLoop();
  }
}

function stopCaptureLoop() {
  if (captureTimer !== null) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
}

function startCaptureLoop() {
  if (captureTimer !== null) return;
  captureTimer = window.setInterval(tryCapture, CAPTURE_INTERVAL);
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Provider entry renderer ──

function renderProviderEntry(index: number, data?: Partial<ProviderConfig>): HTMLDivElement {
  const card = document.createElement("div");
  card.className = "provider-card";
  card.dataset.index = String(index);

  const type = data?.type || "openai";
  const label = data?.label || "";
  const baseUrl = data?.base_url || "";
  const apiKey = data?.api_key || "";
  const model = data?.model || "";
  const awsAccess = (data as any)?.aws_access_key || "";
  const awsSecret = (data as any)?.aws_secret_key || "";
  const region = (data as any)?.region || "";

  // Preset detection for OpenAI
  let preset = "custom";
  if (type === "openai") {
    if (baseUrl.includes("fireworks")) preset = "fireworks";
    else if (baseUrl.includes("nvidia")) preset = "nvapi";
  }

  const isFirst = index === 0;
  const isOpenai = type === "openai";
  const isCustom = preset === "custom";

  card.innerHTML = `
    <div class="card-header">
      <label>Type</label>
      <select class="prov-type">
        <option value="openai" ${isOpenai ? "selected" : ""}>OpenAI Compatible</option>
        <option value="gemini" ${type === "gemini" ? "selected" : ""}>Google Gemini</option>
        <option value="bedrock" ${type === "bedrock" ? "selected" : ""}>AWS Bedrock</option>
      </select>
      <button class="remove-btn" ${isFirst ? "disabled style='opacity:0.3'" : ""}>×</button>
    </div>
    <div class="card-fields">
      <div class="field-row">
        <label>Label</label>
        <input type="text" class="prov-label" placeholder="My Provider" value="${label}" />
      </div>
      <div class="field-row openai-only preset-row" style="${isOpenai ? "" : "display:none"}">
        <label>Preset</label>
        <select class="prov-preset">
          <option value="custom" ${preset === "custom" ? "selected" : ""}>Custom</option>
          <option value="fireworks" ${preset === "fireworks" ? "selected" : ""}>Fireworks AI</option>
          <option value="nvapi" ${preset === "nvapi" ? "selected" : ""}>NVIDIA API</option>
        </select>
      </div>
      <div class="field-row openai-only url-row" style="${isOpenai && isCustom ? "" : "display:none"}">
        <label>Base URL</label>
        <input type="url" class="prov-url" placeholder="https://api.openai.com/v1" value="${isOpenai && isCustom ? baseUrl : ""}" />
      </div>
      <div class="field-row openai-only key-row" style="${isOpenai ? "" : "display:none"}">
        <label>API Key</label>
        <input type="password" class="prov-key" placeholder="sk-..." value="${isOpenai ? apiKey : ""}" />
      </div>
      <div class="field-row gemini-only" style="${type === "gemini" ? "" : "display:none"}">
        <label>API Key</label>
        <input type="password" class="gemini-key" placeholder="AIza..." value="${type === "gemini" ? apiKey : ""}" />
      </div>
      <div class="field-row bedrock-only" style="${type === "bedrock" ? "" : "display:none"}">
        <label>Access Key</label>
        <input type="text" class="bedrock-access" placeholder="AKIA..." value="${type === "bedrock" ? awsAccess : ""}" />
      </div>
      <div class="field-row bedrock-only" style="${type === "bedrock" ? "" : "display:none"}">
        <label>Secret Key</label>
        <input type="password" class="bedrock-secret" placeholder="..." value="${type === "bedrock" ? awsSecret : ""}" />
      </div>
      <div class="field-row bedrock-only" style="${type === "bedrock" ? "" : "display:none"}">
        <label>Region</label>
        <input type="text" class="bedrock-region" placeholder="us-east-1" value="${type === "bedrock" ? region : ""}" />
      </div>
      <div class="field-row model-row">
        <label>Model</label>
        <input type="text" class="prov-model" placeholder="gpt-4o" value="${model}" />
      </div>
    </div>
  `;

  // ── Wire card events ──

  // Type change → show/hide type-specific fields
  const typeSelect = card.querySelector(".prov-type") as HTMLSelectElement;
  typeSelect.addEventListener("change", () => {
    const t = typeSelect.value;
    card.querySelectorAll(".openai-only").forEach(el => (el as HTMLElement).style.display = t === "openai" ? "" : "none");
    card.querySelectorAll(".gemini-only").forEach(el => (el as HTMLElement).style.display = t === "gemini" ? "" : "none");
    card.querySelectorAll(".bedrock-only").forEach(el => (el as HTMLElement).style.display = t === "bedrock" ? "" : "none");
    // Reset preset to custom when switching types
    if (t !== "openai") {
      const presetSelect = card.querySelector(".prov-preset") as HTMLSelectElement;
      if (presetSelect) presetSelect.value = "custom";
    }
  });

  // Preset change → show/hide URL row
  const presetSelect = card.querySelector(".prov-preset") as HTMLSelectElement;
  if (presetSelect) {
    presetSelect.addEventListener("change", () => {
      const p = presetSelect.value;
      const urlRow = card.querySelector(".url-row") as HTMLElement;
      if (urlRow) urlRow.style.display = p === "custom" ? "" : "none";
    });
  }

  // Remove button
  const removeBtn = card.querySelector(".remove-btn") as HTMLButtonElement;
  removeBtn.addEventListener("click", () => {
    if (!isFirst) card.remove();
  });

  return card;
}

// ── Collect provider configs from cards ──
function collectProviders(): ProviderConfig[] {
  const providers: ProviderConfig[] = [];
  const cards = providerList.querySelectorAll(".provider-card");

  cards.forEach((card) => {
    const type = (card.querySelector(".prov-type") as HTMLSelectElement).value as "openai" | "gemini" | "bedrock";
    const label = (card.querySelector(".prov-label") as HTMLInputElement).value.trim() || type;

    if (type === "openai") {
      const preset = (card.querySelector(".prov-preset") as HTMLSelectElement)?.value || "custom";
      const base_url = preset !== "custom"
        ? PRESETS[preset]
        : (card.querySelector(".prov-url") as HTMLInputElement).value.trim();
      const api_key = (card.querySelector(".prov-key") as HTMLInputElement).value.trim();
      const model = (card.querySelector(".prov-model") as HTMLInputElement).value.trim();
      if (base_url && api_key && model) {
        providers.push({ type, label, base_url, api_key, model });
      }
    } else if (type === "gemini") {
      const api_key = (card.querySelector(".gemini-key") as HTMLInputElement).value.trim();
      const model = (card.querySelector(".prov-model") as HTMLInputElement).value.trim();
      if (api_key && model) {
        providers.push({ type, label, api_key, model });
      }
    } else if (type === "bedrock") {
      const aws_access_key = (card.querySelector(".bedrock-access") as HTMLInputElement).value.trim();
      const aws_secret_key = (card.querySelector(".bedrock-secret") as HTMLInputElement).value.trim();
      const region = (card.querySelector(".bedrock-region") as HTMLInputElement).value.trim();
      const model = (card.querySelector(".prov-model") as HTMLInputElement).value.trim();
      if (aws_access_key && aws_secret_key && region && model) {
        providers.push({ type, label, aws_access_key, aws_secret_key, region, model });
      }
    }
  });

  return providers;
}

// ── Test & rank ──
async function handleTest() {
  const providers = collectProviders();

  if (providers.length === 0) {
    testStatus.className = "error";
    testStatus.textContent = "Fill in at least one provider with all required fields.";
    testStatus.classList.remove("hidden");
    return;
  }

  testBtn.disabled = true;
  testBtn.textContent = "Testing...";
  testStatus.className = "hidden";
  results.classList.add("hidden");

  try {
    const res = await fetch("/api/providers/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ providers }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText);
    }

    const data = await res.json();
    const testResults: TestResult[] = data.results;

    // Display ranking
    rankingList.innerHTML = "";
    testResults.forEach((r) => {
      const li = document.createElement("li");
      li.className = r.ok ? "" : "errored";
      li.innerHTML = `
        <span class="rank-provider">${r.label}</span>
        <span class="rank-latency">
          ${r.ok ? r.latency_ms + " ms" : "✗ " + (r.error || "failed")}
        </span>
      `;
      rankingList.appendChild(li);
    });

    results.classList.remove("hidden");

    // Save ordered providers, fastest first.
    const okProviders = testResults
      .filter((r) => r.ok)
      .sort((a, b) => a.latency_ms - b.latency_ms);

    if (okProviders.length === 0) {
      saveBtn.disabled = true;
      testStatus.className = "error";
      testStatus.textContent = "All providers failed. Check your credentials.";
      testStatus.classList.remove("hidden");
    } else {
      saveBtn.disabled = false;

      savedProviders = okProviders
        .map((r) => providers.find((p) => p.label === r.label)!)
        .filter((p) => p !== undefined);

      testStatus.className = "ok";
      testStatus.textContent =
        `${okProviders.length} provider(s) OK. ` +
        `Fastest: ${okProviders[0].label} at ${okProviders[0].latency_ms} ms.`;
      testStatus.classList.remove("hidden");
    }
  } catch (e: any) {
    testStatus.className = "error";
    testStatus.textContent = "Test failed: " + (e.message || "Unknown error");
    testStatus.classList.remove("hidden");
  } finally {
    testBtn.disabled = false;
    testBtn.textContent = "Test & Rank";
  }
}

// ── Save and start camera ──
async function handleSaveAndStart() {
  if (savedProviders.length === 0) return;

  localStorage.setItem("sa_providers", JSON.stringify(savedProviders));
  stopCamera();

  try {
    await startCamera();
    setState("CAPTURING");
  } catch (e: any) {
    errorText.textContent = "Camera access denied";
    setState("ERROR");
  }
}

// ── Load saved providers into the form ──
function loadSavedProviders(): boolean {
  const raw = localStorage.getItem("sa_providers");
  if (!raw) return false;
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length > 0) {
      savedProviders = parsed;
      return true;
    }
  } catch { /* ignore */ }
  return false;
}

// ── Render the provider list from saved data or default ──
function renderProviderList() {
  providerList.innerHTML = "";
  const hasSaved = loadSavedProviders();
  if (hasSaved) {
    savedProviders.forEach((p, i) => {
      providerList.appendChild(renderProviderEntry(i, p));
    });
  } else {
    providerList.appendChild(renderProviderEntry(0));
  }
}

// ── Answer capture loop ──
async function tryCapture() {
  if (captureInFlight || state !== "CAPTURING") return;

  captureInFlight = true;

  const t0 = performance.now();
  const base64 = captureFrame();
  const captureMs = performance.now() - t0;

  if (!base64) {
    captureInFlight = false;
    return;
  }

  // Dwell detection: skip if scene hasn't changed enough.
  if (!sceneChanged()) {
    captureInFlight = false;
    return;
  }

  setState("PROCESSING");

  try {
    const t1 = performance.now();
    const res = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: base64,
        providers: savedProviders,
      }),
    });

    const totalMs = Math.round(performance.now() - t0);
    const modelMs = Math.round(performance.now() - t1);

    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(errBody || `API error ${res.status}`);
    }

    const data = await res.json();
    const answer = String(data.answer || "").trim();

    if (!answer) {
      throw new Error("No answer returned. Trying again...");
    }

    // Success → reset error counter
    consecutiveErrors = 0;

    // Debug overlay
    if (window.location.search.includes("debug")) {
      latencyDebug.classList.remove("hidden");
      latencyDebug.textContent =
        `capture:${Math.round(captureMs)}ms ` +
        `model:${modelMs}ms ` +
        `total:${totalMs}ms`;
    }

    // Show answer
    setState("ANSWER");
    answerText.textContent = answer;

    // Keep answer visible for 7s, then back to capture
    await sleep(7000);
    setState("CAPTURING");
  } catch (e: any) {
    consecutiveErrors++;
    const msg = e?.message || "Request failed";

    setState("ERROR");
    errorText.textContent = msg.length > 100 ? msg.slice(0, 100) + "..." : msg;

    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
      errorText.textContent = `Too many errors. Pausing ${BACKOFF_MS / 1000}s...`;
      await sleep(BACKOFF_MS);
      consecutiveErrors = 0;
    } else {
      await sleep(2500);
    }

    setState("CAPTURING");
  } finally {
    captureInFlight = false;
  }
}

// ── Settings UI wiring ──
function setupSettingsUI() {
  addBtn.addEventListener("click", () => {
    const idx = providerList.querySelectorAll(".provider-card").length;
    providerList.appendChild(renderProviderEntry(idx));
  });

  testBtn.addEventListener("click", handleTest);
  saveBtn.addEventListener("click", handleSaveAndStart);

  settingsBtn.addEventListener("click", () => {
    stopCamera();
    savedProviders = [];
    setState("SETTINGS");
  });
}

// ── Boot ──
renderProviderList();
setupSettingsUI();
setState("SETTINGS"); // always show settings on boot
