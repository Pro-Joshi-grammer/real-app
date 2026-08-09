# Screen Answer — MVP

Phone camera → AI answer → green screen → repeat.

## Quick start

### Backend

```bash
cd backend
cp .env.example .env   # edit .env with your AI provider details
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 on your laptop or phone (same network).

Add `?debug` to the URL to see latency metrics: http://localhost:5173?debug

## .env

| Variable | Required | Description |
|---|---|---|
| `AI_BASE_URL` | yes | OpenAI-compatible API base URL |
| `AI_API_KEY` | yes | API key |
| `IMAGE_MODEL` | yes | Model ID for image questions |
| `TEXT_MODEL` | no | Model ID for text questions |
| `MAX_OUTPUT_TOKENS` | no | Default 50 |
| `REQUEST_TIMEOUT_SECONDS` | no | Default 15 |
