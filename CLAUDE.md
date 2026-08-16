# Screen Answer — Codebase Guide

**Read this first.** New sessions should start here to get the full picture without re-reading the code. Append changes to the **Change Log** at the bottom. Keep every section accurate when you edit code.

---

## What it is

An MVP mobile web app: **phone camera → capture a question on screen → AI answers it → show answer full-screen (green) for 7s → camera resumes → repeat.**

- Latency is the #1 priority: fastest practical end-to-end answer.
- Single-purpose, no accounts, no DB, no image storage, no chat history.
- Full design doc: `AI_Screen_Answer_MVP_DeepSeek_Handoff.md` (authoritative spec).

## Stack

| Part | Tech |
|---|---|
| Frontend | Vanilla TypeScript + Vite (`frontend/`), no framework |
| Backend | Python + FastAPI + uvicorn (`backend/`), async |
| AI clients | OpenAI-compatible (primary), Google Gemini, AWS Bedrock |
| OCR | Server-side `easyocr`, lazy-loaded, runs in a thread |

---

## Project structure

```
screen-reader-updated/
├── CLAUDE.md                     ← you are here
├── AI_Screen_Answer_MVP_DeepSeek_Handoff.md   ← full spec
├── README.md                     ← quick start
├── backend/
│   ├── .env / .env.example       ← config (see below)
│   ├── requirements.txt
│   ├── test_extract.py           ← runnable self-check for answer extraction
│   └── app/
│       ├── main.py               ← FastAPI app, routes, prompts, extractor
│       ├── config.py             ← env → AsyncOpenAI client + model names
│       └── providers.py          ← per-provider API call logic (openai/gemini/bedrock)
└── frontend/
    ├── index.html                ← all views (settings/camera/processing/answer/error)
    ├── style.css
    ├── vite.config.ts            ← /api proxy → localhost:8000
    ├── tsconfig.json
    └── src/main.ts               ← entire frontend logic (state machine, camera, settings UI)
```

---

## Backend

### Run

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Config (`backend/app/config.py`)
`load_config()` reads `.env`, returns `{client: AsyncOpenAI, text_model, image_model, max_tokens, timeout}`.
- `text_model`/`image_model` can be empty — env is only a **fallback** when the frontend sends no providers.
- **Security note:** `config.py` has a hardcoded default API key fallback (`sk-awhDH...`). Real key comes from `.env` (gitignored); don't commit real keys.

### Routes (`backend/app/main.py`)
- **`POST /api/answer`** — main flow. Body: `{ image_base64, providers?: [...] }`. Pipeline:
  1. **Vision (VLM) path is primary**: decode base64 image, call `call_provider` + `IMAGE_PROMPT` for every image. (OCR → text-model routing is intentionally unused for the MVP; OCR code `get_ocr` kept but not called.)
  2. `extract_answer` requires exact `<answer>…</answer>` tags — missing/malformed → `""` (never raw reasoning), treated as a provider failure.
  3. Iterate providers as a **fallback chain**: try each, first success wins; empty answers, malformed tags, and bare option letters (`A`/`A)`/`A.`) are treated as failures and move on.
  4. Returns `{ "answer": "<extracted answer>" }`; 502 if all providers fail.
  - If `providers` is absent/empty, builds one provider from env config.
- **`POST /api/providers/test`** — body `{ providers: [...] }`. Latency-tests all providers concurrently via `asyncio.gather`, returns `{ results: [{label, type, ok, latency_ms, error?}] }` sorted fastest-successful-first.
- **`GET /api/health`** — `{ status: "ok" }`.

### Providers (`backend/app/providers.py`)
Every provider config dict carries its own `type`, `api_key`, `base_url`, `model`, and optional credentials, so the backend supports **multiple user-entered providers with auto-fallback**:
- **`openai`** (also covers Fireworks/NVIDIA — any OpenAI-compatible base URL). `_call_openai_text` / `_call_openai_image` (image sent as `image_url` data URL).
- **`gemini`** — uses `google-genai` SDK, sync calls run via `asyncio.to_thread`. Optional import (`google.genai`) guarded.
- **`bedrock`** — boto3, Claude message format, `asyncio.to_thread`. Optional import (`boto3`) guarded.
- `test_provider(config)` = wall-clock latency of one tiny request ("Reply with exactly one word: OK", 5 tokens).
- `call_provider(config, image_base64, prompt)` = image/vision path.
- `call_provider_text(config, text, system_prompt)` = text-only path (post-OCR).

### Prompts & answer extraction
- `TEXT_QA_SYSTEM` / `IMAGE_PROMPT`: instruct the model to wrap ONLY the final answer in `<answer>…</answer>` tags; everything outside is discarded (allows arbitrary content — formulas, code — with no schema).
- `extract_answer(raw)` → regex-pulls `<answer>…</answer>`, else falls back to full response; both pass through `normalize_answer`.
- `normalize_answer(raw)` → strips markdown fences, leading intro phrases ("the correct answer is", "answer:"), collapses whitespace.
- **Test:** `python backend/test_extract.py` (assert-based, no framework).

---

## Frontend

### Run

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

`vite.config.ts` proxies `/api` → `http://localhost:8000`. Open on laptop or phone (same network). Append `?debug` for latency overlay.

### State machine (`frontend/src/main.ts`)
`SETTINGS → CAPTURING → PROCESSING → ANSWER (7s) → CAPTURING`, with `PROCESSING → ERROR → CAPTURING`. One boolean gate (`captureInFlight`) ensures only one AI request at a time.

### Settings flow (boot state)
- Boot always shows **SETTINGS**. User adds provider cards (OpenAI-compatible / Gemini / Bedrock) with presets (custom, Fireworks `https://api.fireworks.ai/inference/v1`, NVIDIA `https://integrate.api.nvidia.com/v1`).
- **Test & Rank** → `POST /api/providers/test` → ranked list; **Save & Start** persists ordered (fastest-first) providers to `localStorage["sa_providers"]` and starts camera.
- Valid providers (all required fields filled) only. First card is non-removable.

### Capture loop
- `startCamera()` → `getUserMedia` (rear `facingMode: "environment"`).
- `tryCapture()` every **2s** (`CAPTURE_INTERVAL`, down from 400ms): capture frame → JPEG 0.88, max width 1600 → **dwell/scene-change check** (8×8 downsampled checksum; only proceed if ≥3% change) → `POST /api/answer` → show answer 7s → resume.
- **Backoff:** after 5 consecutive errors, pause 30s. Otherwise 2.5s between retries.

### Views
`index.html` holds 5 views toggled by `.hidden`: settings, camera (video + capture hint + settings gear btn + debug overlay), processing (spinner), answer (green bg, `#00c853`, white centered clamp text), error (red bg).

---

## API contract (frontend ⇄ backend)

```http
POST /api/answer
{ "image_base64": "<data:image/jpeg;base64,...>", "providers": [ {type,label,base_url,api_key,model,...} ] }
→ 200 { "answer": "C) Large Language Model" } | 502 on all-provider failure

POST /api/providers/test
{ "providers": [ ... ] }   # same provider shape
→ 200 { "results": [{label,type,ok,latency_ms,error?}] }   # sorted fastest-first
```

---

## Status / known state
- Clean git repo, branch `main`. Recent commits: `e079010 updated backend`, `2e9a764 v2`.
- `backend/.env` exists (with live key); `.env`, `.venv`, `node_modules`, `dist` are gitignored.
- Server has ~700 MB free disk: avoid heavy deps / image archives / extra build artifacts.

---

## Change Log
*Append new entries here (newest on top) after making changes; update any affected section above.*

- VQA pipeline fix (backend):
  - `/api/answer` now always calls the **vision model** (`call_provider` + `IMAGE_PROMPT`) for every image. OCR → text-model routing removed from the primary path; OCR code (`get_ocr`) kept but unused.
  - `IMAGE_PROMPT` tightened: response must be EXACTLY `<answer>FINAL ANSWER</answer>`, nothing else; MCQs require `C) Full option text`; explicitly covers long/multi-option MCQs, True/False, fill-blank, math/numerical, code/output, tables, charts, diagrams, multi-column layouts.
  - `extract_answer` returns `""` when `<answer>…</answer>` is missing or malformed (never returns raw reasoning) → triggers provider fallback.
  - Fallback validation: empty answers, malformed tags, and bare letters (`A`/`A)`/`A.`) are treated as provider failures and move to the next provider.
  - `normalize_answer` preserves line breaks (only collapses runs of spaces/tabs within a line) so code/formulas/structured answers keep their formatting.
  - `test_extract.py` updated to match (untagged/malformed → `""`; multiline preserved).
  - Frontend already validated non-empty answer before `setState("ANSWER")` — no change needed.
  - Verified: `python test_extract.py` passes, backend py_compile clean, `npx tsc --noEmit` clean.

- (start of project) — repo snapshot committed at `e079010`; full codebase mapped and documented here.
