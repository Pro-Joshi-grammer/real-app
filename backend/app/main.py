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

Read the ENTIRE question and ALL visible information in the image — including
layout and visual context such as multi-column arrangements, headers, tables,
charts, diagrams, formulas and code — then provide the FINAL ANSWER.

Support EVERY question type, including:
- Multiple choice: SHORT and LONG / multi-option questions
- True/False
- Fill in the blank
- Short factual and definition questions
- Numerical / calculation and math problems
- Programming and code-output prediction questions
- Questions containing tables, charts, diagrams, formulas, or multi-column layouts

For MULTIPLE CHOICE questions, the answer is the option LETTER, then ")",
then the COMPLETE option text copied exactly, for example:

C) Large Language Model

NEVER return only the letter (like just "C"). Always include the full option
text, even for long or wordy options.

For all other types, return the direct answer with enough detail to be useful
on its own, keeping code, formulas and multi-line structure intact.

OUTPUT CONTRACT (STRICT — do not deviate):
Wrap ONLY the final answer in EXACTLY one pair of tags, and output NOTHING
else in the response:

<answer>FINAL ANSWER</answer>

- The response must be exactly that <answer>...</answer> pair with no
  thinking, analysis, explanation, markdown, or text before or after it.
- Do not describe the image, people, faces, objects, colors, or layout.
- If you genuinely cannot answer, still respond with ONLY <answer></answer>.
"""


# ── Extractor ──

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(raw: str) -> str:
    """Pull the final answer out of a raw model response.

    A valid answer MUST be wrapped in <answer>…</answer>. Everything outside
    the tags (thinking/reasoning) is discarded, which lets answers hold
    arbitrary content (numbers, symbols, formulas, code) with no schema.

    Missing or malformed tags return "" so the caller treats it as a provider
    failure and falls through to the next provider — untagged model reasoning
    is never shown to the user.
    """
    if not raw:
        return ""
    m = _ANSWER_TAG_RE.search(raw)
    if m:
        return normalize_answer(m.group(1))
    return ""


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

    # Collapse runs of spaces/tabs within each line but PRESERVE line breaks
    # (code, formulas, multi-line structured answers).
    s = re.sub(r"[ \t]+", " ", s)

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

    last_error = ""

    # ── Provider fallback chain ──
    # VLM inference is the PRIMARY path for every image: visual layout/context
    # matters for long/multi-option MCQs, tables, charts and diagrams. The
    # OCR → text-model routing is intentionally not used for the MVP. OCR code
    # (get_ocr) stays available but unused.
    for prov in providers:
        try:
            raw = await call_provider(prov, image_base64, IMAGE_PROMPT)

            answer_text = extract_answer(raw)

            # Missing/malformed <answer> tags → extract_answer returned "".
            if not answer_text:
                raise ValueError(
                    "Provider returned no valid <answer>…</answer>"
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