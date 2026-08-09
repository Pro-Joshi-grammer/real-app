# MVP Agent Handoff — Ultra-Fast Camera Answer Web App

## Goal

Build a minimal mobile-first web app that uses the phone camera to repeatedly capture questions, send them to a configurable AI backend, and display only the answer.

Example:

Camera sees:

"What is the full form of LLM?
A) Large Learning Model
B) Long Language Model
C) Large Language Model
D) Linear Language Model"

App displays:

`C) Large Language Model`

on a full-screen green screen for 7 seconds, then automatically resumes camera capture.

This is an MVP. Prioritize latency and reliability over architecture/features.

---

## Core Flow

```text
PHONE CAMERA
    ↓
capture frame
    ↓
extract text if possible
    ↓
┌──────────────────────────┐
│ Good text?               │
│                          │
│ YES → text model         │
│ NO  → image model        │
└──────────────────────────┘
    ↓
short answer
    ↓
GREEN FULL-SCREEN RESULT
    ↓
7 seconds
    ↓
CAMERA AGAIN
```

Only CAMERA mode.

Do NOT implement:
- screen sharing
- screen capture
- Android app
- native APIs
- background recording
- audio
- accounts
- database
- chat history
- persistent image storage
- unnecessary UI

---

# Performance Requirements

Latency is the primary requirement.

Target the fastest practical end-to-end response.

Optimize for:

1. Minimal frame processing.
2. No unnecessary image uploads.
3. Local/client-side OCR if it is fast enough.
4. Text model when OCR succeeds.
5. Image model only when OCR fails/is insufficient.
6. Small compressed images.
7. Persistent HTTP connections.
8. Async/non-blocking backend.
9. Minimal JSON serialization.
10. Minimal model output.
11. No model reasoning/explanation.
12. No repeated API calls for the same question.

Do not introduce heavy ML frameworks unless benchmarking proves they improve the result.

Do not run local LLM/VLM inference.

Do not process full-resolution camera frames continuously.

---

# Frontend

Use the lightest practical web stack.

Preferred:

- TypeScript
- Vite
- React only if useful

Vanilla TypeScript is acceptable if it produces a simpler/faster implementation.

Camera:

```js
navigator.mediaDevices.getUserMedia({
  video: {
    facingMode: "environment"
  },
  audio: false
})
```

Prefer the rear camera.

Show a live camera preview.

Capture individual frames from the video rather than uploading a video stream.

Use a canvas/offscreen canvas for frame extraction.

---

# Capture Strategy

Do NOT continuously send frames to the server.

Use a lightweight polling loop, approximately 2–5 FPS while searching for a question.

Pipeline:

```text
camera frame
   ↓
resize
   ↓
OCR / text extraction
   ↓
question detection
   ↓
stable question?
   ↓
YES
   ↓
API request
```

When an API request is in progress:

- do not send another request
- discard stale frames
- optionally retain only the latest frame

After the answer screen finishes, restart capture processing.

---

# OCR

Use client-side OCR only if it is lightweight and fast on the target phone/browser.

The OCR layer must be replaceable.

Interface concept:

```ts
type OCRResult = {
  text: string
  confidence?: number
}
```

If OCR gives sufficiently usable question text:

```text
OCR → text API → text model
```

If OCR is empty, poor, or clearly incomplete:

```text
frame → compressed image → image API → image model
```

Do not add heavy server-side OCR for the MVP.

Do not send both text and image unless necessary.

---

# Question Detection

Avoid API calls for irrelevant frames.

Basic detection is enough initially.

Look for signals such as:

- `?`
- A/B/C/D options
- numbered questions
- sufficient text length
- multiple text blocks

Do not build an AI classifier for this.

---

# Stable Frame Detection

The camera will produce many nearly identical frames.

Normalize OCR text:

- lowercase
- trim
- collapse whitespace
- remove irrelevant punctuation

Compare consecutive OCR results.

Example:

```text
frame 1 → question X
frame 2 → question X
frame 3 → question X
```

After 2 stable matches, process the question.

Maintain:

```text
lastQuestionHash
```

If the same question is seen again during the current cycle, do not call the model again.

---

# API Contract

Frontend should communicate with ONE backend endpoint.

Example:

```http
POST /api/answer
Content-Type: application/json
```

Text request:

```json
{
  "mode": "text",
  "text": "question and options",
  "model": "auto"
}
```

Image request:

```json
{
  "mode": "image",
  "image_base64": "...",
  "model": "auto"
}
```

Response:

```json
{
  "answer": "C) Large Language Model"
}
```

Optional development metadata:

```json
{
  "answer": "C) Large Language Model",
  "latency_ms": 642,
  "model": "..."
}
```

Do not expose provider/API implementation details to the frontend.

---

# Model Configuration

The backend must NOT hardcode model selection.

Use environment variables.

Example:

```env
TEXT_MODEL=...
IMAGE_MODEL=...
```

Optional:

```env
FALLBACK_TEXT_MODEL=...
FALLBACK_IMAGE_MODEL=...
MAX_OUTPUT_TOKENS=...
REQUEST_TIMEOUT_SECONDS=...
```

The implementation should simply use the configured models through the existing AI/API integration.

Do not assume a particular provider in the application architecture.

The model names are configuration, not source-code constants.

---

# Prompt Requirements

The model must return the answer only.

For MCQs:

```text
You are a fast answer extraction engine.

Identify the correct option.

Return ONLY:
<letter>) <answer>

Example:
C) Large Language Model

No explanation.
No reasoning.
No markdown.
No introductory text.
```

For non-MCQ questions:

```text
Answer the question with the shortest correct answer.

Return ONLY the answer.
No explanation.
No reasoning.
No markdown.
```

Use the smallest practical output-token limit.

---

# Answer Normalization

Models may return:

```text
The correct answer is C) Large Language Model.
```

Normalize it to:

```text
C) Large Language Model
```

Implement a small deterministic normalizer.

Remove:

- markdown
- introductory phrases
- explanations
- unnecessary whitespace

For MCQs, preserve the option letter and answer.

---

# Result UI

When answer arrives:

```text
state = ANSWER
```

Render:

```text
full viewport
green background
white centered H1
```

Example:

```text
┌─────────────────────────┐
│                         │
│                         │
│ C) Large Language Model │
│                         │
│                         │
└─────────────────────────┘
```

Requirements:

- full screen
- no scrolling
- centered horizontally and vertically
- large readable text
- responsive font size
- wrap long answers
- green background
- white text

Display for:

```text
7 seconds
```

Then:

```text
ANSWER
  ↓
clear answer
  ↓
resume camera
```

Do not require a button press.

---

# State Machine

Keep the implementation explicit:

```text
IDLE
 ↓
CAPTURING
 ↓
PROCESSING
 ↓
ANSWER
 ↓ 7 sec
CAPTURING
```

Error:

```text
PROCESSING
 ↓
ERROR
 ↓
CAPTURING
```

Do not allow multiple concurrent answer requests.

---

# Backend Performance

Use an async lightweight backend.

Preferred:

Python + FastAPI + Uvicorn

Rust is acceptable only if it materially improves the implementation; do not choose Rust merely for theoretical performance.

The dominant latency will be network + model inference, not Python execution.

Backend requirements:

- async request handling
- persistent client connections
- request timeout
- small request validation layer
- no unnecessary middleware
- no database
- no image persistence
- no synchronous blocking calls in request handlers
- no verbose production logging

If the existing environment already has a suitable Python stack, reuse it.

---

# Image Processing

Before sending image:

1. Resize.
2. Compress.
3. Keep text readable.
4. Send only the required frame.
5. Do not store it.

Start with approximately 640–900px width and benchmark.

Do not upload 1080p/4K images by default.

Use JPEG/WebP depending on which gives better size/readability.

---

# Latency Instrumentation

Measure:

```text
capture_ms
ocr_ms
encode_ms
upload_ms
model_ms
total_ms
```

In development/debug mode show:

```text
OCR: xxx ms
AI: xxx ms
Total: xxx ms
```

Do not display this in normal mode.

The goal is to identify the actual bottleneck rather than prematurely optimizing.

---

# Error Handling

If OCR fails:

→ use image model.

If text model fails:

→ optional configured fallback text model.

If image model fails:

→ optional configured fallback image model.

If all fail:

show a minimal error state briefly, then automatically return to camera.

Never enter an infinite retry loop.

Set strict request timeouts.

---

# Avoid

Do NOT implement:

- screen capture
- screen sharing
- native Android
- React Native
- Flutter
- WebSockets unless benchmarking proves they are needed
- Redis
- PostgreSQL
- MongoDB
- authentication
- user accounts
- analytics
- chat history
- image storage
- local LLMs
- local VLMs
- server-side heavy OCR
- background jobs
- queues
- microservices
- Docker unless required by the existing deployment
- unnecessary abstractions
- complex state-management libraries
- large UI frameworks
- continuous video uploads
- repeated calls for identical questions

Keep it small.

---

# Security / Resource Constraints

The server has limited free disk space (~700 MB).

Therefore:

- no model weights
- no large dependencies
- no image archives
- no uploaded-file persistence
- no unnecessary build artifacts
- no development dependencies in production
- no `node_modules` required on production if frontend is prebuilt
- clean logs/caches

API keys/credentials must never be sent to the browser.

The browser only communicates with the backend.

---

# Development Requirements

The project must be easy to run locally.

Frontend:

```bash
npm install
npm run dev
```

Backend:

```bash
python -m venv .venv
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test on laptop browser first.

Then test the same development server from a real phone over the local network.

Example:

```text
Laptop:
192.168.1.10

Frontend:
http://192.168.1.10:5173

Backend:
http://192.168.1.10:8000
```

If camera access requires a secure context, configure local HTTPS rather than deploying every change.

Do NOT require deployment for normal development/testing.

---

# Project Structure

Keep it simple:

```text
project/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── camera/
│   │   ├── ocr/
│   │   ├── api/
│   │   └── components/
│   ├── package.json
│   └── .env.example
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api.py
│   │   ├── ai.py
│   │   └── normalizer.py
│   ├── requirements.txt
│   └── .env.example
│
└── README.md
```

Avoid creating dozens of files for a small MVP.

---

# Implementation Order

## Phase 1
Build camera UI.

## Phase 2
Build fake answer flow:

```text
camera
→ fake request
→ green answer
→ 7 sec
→ camera
```

## Phase 3
Connect backend.

## Phase 4
Implement text model path.

## Phase 5
Implement image model path.

## Phase 6
Add OCR.

## Phase 7
Add stable-frame + duplicate detection.

## Phase 8
Measure and optimize latency.

Do not build advanced features before the complete basic loop works.

---

# MVP Acceptance Criteria

The MVP is DONE when:

- Camera opens on a phone.
- Rear camera is preferred.
- A question can be captured.
- OCR text is used when sufficiently reliable.
- Image fallback works.
- Configured text model is used for text.
- Configured image model is used for images.
- No model names are hardcoded.
- Answer is displayed alone.
- Answer is centered on a green full-screen UI.
- Answer stays for 7 seconds.
- Camera automatically resumes.
- Duplicate frames/questions do not trigger duplicate requests.
- Only one AI request is active at a time.
- Images are compressed before upload.
- Images are not persisted.
- Backend is lightweight.
- Local development works without deployment.
- Real-phone testing works from the laptop development server.
- No unnecessary infrastructure is introduced.

# Priority

P0:
Camera → AI → answer → 7 sec → repeat.

P1:
OCR → text model optimization.

P1:
Image fallback.

P1:
Duplicate/stable-frame detection.

P2:
Latency metrics and optimization.

Everything else is out of scope for this MVP.
