import asyncio
import base64
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import load_config
from .providers import test_provider, call_provider, call_provider_text


# ── Config (env fallback) ──

cfg = None
_ocr = None  # lazy init


def get_ocr():
    global _ocr
    if _ocr is None:
        import easyocr
        _ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global cfg
    cfg = load_config()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Text-only QA prompt (used after OCR) ──

TEXT_QA_SYSTEM = """
You are a question-answering engine. The text below was extracted from an image
of a question displayed on a screen. Answer the question directly.

Return ONLY the final answer. If the question has multiple-choice options visible,
return the option letter followed by the complete answer text — for example:

C) Large Language Model

If no options are visible, return the computed or factual answer with enough
detail to be useful by itself.

Rules:
- Do not add explanations, reasoning, or introductory phrases.
- Do not say "the correct answer is", "the answer is", or similar.
- Do not use markdown.
- Return ONLY the final answer.
"""

# ── Visual question-answering prompt ──

IMAGE_PROMPT = """
You are a visual question-answering engine.

Read the ENTIRE question and all visible information in the image, then provide
the FINAL ANSWER.

Handle all question types, including:

- Multiple choice questions
- True/False questions
- Fill in the blank
- Short factual questions
- Definition questions
- Numerical/calculation questions
- Math problems
- Programming/code questions
- Code output prediction questions
- Conceptual/theory questions
- Questions containing tables
- Questions containing diagrams
- Questions containing charts
- Questions containing formulas

For MULTIPLE CHOICE questions:

Return the option LETTER followed by the COMPLETE option text.

Example:
C) Large Language Model

NEVER return only:
A
B
C
D

For other question types:

Return the direct final answer with enough detail to be useful.

Do not provide chain-of-thought or hidden reasoning.

Rules:

- Analyze the complete image before answering.
- Read all visible answer choices before selecting an answer.
- Do not describe the image.
- Do not describe people, faces, objects, colors, or layout.
- Do not say "the user wants".
- Do not say "the image shows".
- Do not provide an analysis section.
- Do not provide reasoning steps unless they are required as part of the final answer.
- Do not use markdown.
- Do not add introductory text.
- Do not say "the correct answer is".
- Return ONLY the final answer.
"""


# ── Normalizer ──

def normalize_answer(raw: str) -> str:
    if not raw:
        return ""

    s = raw.strip()

    # Remove markdown code fences.
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Remove common introductory phrases.
    s = re.sub(
        r"^(the correct answer is|answer:?|the answer is:?)\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    # Normalize whitespace.
    s = re.sub(r"\s+", " ", s)

    return s


# ── Helpers ──

def _decode_b64(data: str) -> bytes:
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    return base64.b64decode(data)


# ── Routes ──

@app.post("/api/providers/test")
async def providers_test(body: dict):
    """Test latency for each provider."""

    providers = body.get("providers", [])

    if not providers:
        raise HTTPException(
            status_code=400,
            detail="providers list is required",
        )

    results = await asyncio.gather(
        *(test_provider(p) for p in providers)
    )

    # Fastest successful providers first.
    # Failed providers go to the end.
    results.sort(
        key=lambda r: (
            not r["ok"],
            r["latency_ms"],
        )
    )

    return {"results": results}


@app.post("/api/answer")
async def answer(body: dict):
    """
    Answer an image using the supplied provider list.

    Pipeline:
      1. Run OCR on the image to extract text.
      2. If OCR yields meaningful text → send to text-only LLM (cheaper, more models).
      3. If OCR fails → fall back to vision-model path (existing behavior).
    """

    image_base64 = body.get("image_base64", "")

    if not image_base64:
        raise HTTPException(
            status_code=400,
            detail="image_base64 is required",
        )

    providers = body.get("providers")

    # If no providers were supplied by the frontend,
    # fall back to the environment-configured provider.
    if not providers:
        model = cfg.get("text_model") or cfg.get("image_model", "")

        if not model:
            raise HTTPException(
                status_code=400,
                detail="No providers configured",
            )

        providers = [
            {
                "type": "openai",
                "base_url": cfg.get(
                    "base_url",
                    cfg["client"].base_url,
                ),
                "api_key": cfg["client"].api_key,
                "model": model,
                "label": "env",
            }
        ]

    # ── OCR step ──

    ocr_text = ""

    try:
        raw_bytes = _decode_b64(image_base64)
        reader = get_ocr()
        results = await asyncio.to_thread(reader.readtext, raw_bytes)
        lines = [r[1] for r in results]
        ocr_text = " ".join(lines).strip()
    except Exception:
        ocr_text = ""  # OCR failed; fall through to VLM

    last_error = ""

    # ── Provider fallback chain ──

    for prov in providers:
        try:
            if len(ocr_text) > 10:
                # OCR gave us usable text → use any text model.
                raw = await call_provider_text(
                    prov, ocr_text, TEXT_QA_SYSTEM,
                )
            else:
                # OCR too short or empty → use vision model.
                raw = await call_provider(
                    prov, image_base64, IMAGE_PROMPT,
                )

            answer_text = normalize_answer(raw)

            if not answer_text:
                raise ValueError(
                    "Provider returned an empty answer"
                )

            # Bare option letter → try next provider.
            if re.fullmatch(r"[A-Da-d][.)]?", answer_text):
                raise ValueError(
                    f"Provider returned only an option letter: "
                    f"{answer_text}"
                )

            return {"answer": answer_text}

        except Exception as e:
            last_error = str(e)
            continue

    # Every provider failed.
    raise HTTPException(
        status_code=502,
        detail=f"All providers failed. Last error: {last_error}",
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}