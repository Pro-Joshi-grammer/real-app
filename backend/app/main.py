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

If the question has multiple-choice options visible, the final answer is the
option letter followed by the complete answer text — for example:

C) Large Language Model

If no options are visible, the final answer is the computed or factual answer
with enough detail to be useful by itself.

OUTPUT CONTRACT:
Put ONLY the final answer between the tags below. Do all thinking silently and
OUTSIDE the tags — only the tagged text is shown to the user.

<answer>FINAL_ANSWER_HERE</answer>

- Keep the final answer concise: no explanations, reasoning, or intro phrases.
- Do not use markdown unless required by the answer itself (e.g. a code snippet).
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

The final answer is the option LETTER followed by the COMPLETE option text.

Example:
C) Large Language Model

NEVER output only the letter (like just "C").

For other question types:

The final answer is the direct answer with enough detail to be useful.

OUTPUT CONTRACT:
Put ONLY the final answer between the tags below. Do all analysis and any
thinking silently and OUTSIDE the tags — only the tagged text is shown to the user.

<answer>FINAL_ANSWER_HERE</answer>

- The final answer must be a single concise answer, not an analysis.
- Do not describe the image, people, faces, objects, colors, or layout.
- Do not use markdown unless required by the answer itself (e.g. a code snippet).
"""


# ── Extractor ──

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(raw: str) -> str:
    """Pull the final answer out of a raw model response.

    Models are prompted to wrap the answer in <answer>…</answer>. Everything
    outside the tags (prefix/suffix thinking) is discarded. This lets answers
    hold arbitrary content (numbers, symbols, formulas, code) with no schema.

    Falls back to stripping the full response if no tags are present.
    """
    if not raw:
        return ""
    m = _ANSWER_TAG_RE.search(raw)
    if m:
        return normalize_answer(m.group(1))
    return normalize_answer(raw)


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

            answer_text = extract_answer(raw)

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