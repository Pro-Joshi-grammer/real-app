"""Per-provider API logic for latency testing and answer calls."""

import asyncio
import base64
import json
import time

# ── Import SDKs (may not be installed for all providers) ──

_HAS_BOTO3 = False
_HAS_GEMINI = False

try:
    import boto3  # noqa: F401
    _HAS_BOTO3 = True
except ImportError:
    pass

try:
    from google import genai as google_genai  # noqa: F401
    from google.genai import types as google_types
    _HAS_GEMINI = True
except ImportError:
    pass

from openai import AsyncOpenAI

# ── Test prompt (tiny, fast) ──

TEST_PROMPT = "Reply with exactly one word: OK"
TEST_MAX_TOKENS = 5


async def test_provider(config: dict) -> dict:
    """Measure wall-clock latency for a provider.

    Returns ``{label, type, ok, latency_ms, error?}``.
    """
    label = config.get("label", config["type"])
    t0 = time.monotonic()
    try:
        ptype = config["type"]
        if ptype == "openai":
            await _call_openai_text(config, TEST_PROMPT, TEST_MAX_TOKENS)
        elif ptype == "gemini":
            await _call_gemini_text(config, TEST_PROMPT, TEST_MAX_TOKENS)
        elif ptype == "bedrock":
            await _call_bedrock_text(config, TEST_PROMPT, TEST_MAX_TOKENS)
        else:
            return {"label": label, "type": config["type"], "ok": False,
                    "error": f"Unknown provider type: {ptype}", "latency_ms": 0}
        elapsed = (time.monotonic() - t0) * 1000
        return {"label": label, "type": config["type"], "ok": True,
                "latency_ms": int(elapsed)}
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return {"label": label, "type": config["type"], "ok": False,
                "error": str(e), "latency_ms": int(elapsed)}


async def call_provider(config: dict, image_base64: str, prompt: str) -> str:
    """Send an image to a provider and return the answer text.

    Raises on failure so the caller can fall through.
    """
    ptype = config["type"]
    if ptype == "openai":
        return await _call_openai_image(config, image_base64, prompt)
    elif ptype == "gemini":
        return await _call_gemini_image(config, image_base64, prompt)
    elif ptype == "bedrock":
        return await _call_bedrock_image(config, image_base64, prompt)
    else:
        raise ValueError(f"Unknown provider type: {ptype}")


async def call_provider_text(config: dict, text: str, system_prompt: str) -> str:
    """Send plain text (e.g. OCR output) to a provider and return the answer.

    Uses the text-only API path so any model works (no vision needed).
    Raises on failure so the caller can fall through.
    """
    ptype = config["type"]
    full_prompt = f"{system_prompt}\n\n{text}"
    if ptype == "openai":
        return await _call_openai_text(
            config, full_prompt,
            int(config.get("max_tokens", 512)),
        )
    elif ptype == "gemini":
        return await _call_gemini_text(
            config, full_prompt,
            int(config.get("max_tokens", 512)),
        )
    elif ptype == "bedrock":
        return await _call_bedrock_text(
            config, full_prompt,
            int(config.get("max_tokens", 512)),
        )
    else:
        raise ValueError(f"Unknown provider type: {ptype}")


# ── OpenAI-compatible (also covers Fireworks, NVIDIA) ──

def _openai_client(config: dict) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config["base_url"], api_key=config["api_key"])


async def _call_openai_text(config: dict, prompt: str, max_tokens: int) -> str:
    client = _openai_client(config)
    resp = await client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=10,
    )
    return resp.choices[0].message.content or ""


async def _call_openai_image(config: dict, image_base64: str, prompt: str) -> str:
    client = _openai_client(config)
    # strip data URL prefix
    raw = image_base64
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[-1]
    resp = await client.chat.completions.create(
        model=config["model"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{raw}"}},
            ],
        }],
        max_tokens=int(config.get("max_tokens", 512)),
        timeout=int(config.get("timeout", 30)),
    )
    return resp.choices[0].message.content or ""


# ── Gemini ──

async def _call_gemini_text(config: dict, prompt: str, max_tokens: int) -> str:
    if not _HAS_GEMINI:
        raise RuntimeError("google-genai not installed")
    client = google_genai.Client(api_key=config["api_key"])
    # run sync SDK in thread
    return await asyncio.to_thread(
        _gemini_sync_text, client, config["model"], prompt, max_tokens
    )


def _gemini_sync_text(client, model: str, prompt: str, max_tokens: int) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=google_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text


async def _call_gemini_image(config: dict, image_base64: str, prompt: str) -> str:
    if not _HAS_GEMINI:
        raise RuntimeError("google-genai not installed")
    client = google_genai.Client(api_key=config["api_key"])
    raw = image_base64
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[-1]
    image_bytes = base64.b64decode(raw)
    return await asyncio.to_thread(
        _gemini_sync_image, client, config["model"], image_bytes, prompt,
        config.get("max_tokens", 512),
    )


def _gemini_sync_image(
    client,
    model: str,
    image_bytes: bytes,
    prompt: str,
    max_tokens: int,
) -> str:
    from google.genai import types as gt

    image_part = gt.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg",
    )

    resp = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            image_part,
        ],
        config=gt.GenerateContentConfig(
            max_output_tokens=max_tokens,
        ),
    )

    return resp.text or ""


# ── Bedrock ──

async def _call_bedrock_text(config: dict, prompt: str, max_tokens: int) -> str:
    if not _HAS_BOTO3:
        raise RuntimeError("boto3 not installed")
    return await asyncio.to_thread(
        _bedrock_sync_text, config, prompt, max_tokens
    )


def _bedrock_sync_text(config: dict, prompt: str, max_tokens: int) -> str:
    client = boto3.client(
        "bedrock-runtime",
        region_name=config["region"],
        aws_access_key_id=config["aws_access_key"],
        aws_secret_access_key=config["aws_secret_key"],
    )
    # Claude message format
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(modelId=config["model"], body=body)
    resp_body = json.loads(resp["body"].read())
    return resp_body["content"][0]["text"]


async def _call_bedrock_image(config: dict, image_base64: str, prompt: str) -> str:
    if not _HAS_BOTO3:
        raise RuntimeError("boto3 not installed")
    raw = image_base64
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[-1]
    return await asyncio.to_thread(
        _bedrock_sync_image, config, raw, prompt,
        config.get("max_tokens", 512),
    )


def _bedrock_sync_image(config: dict, raw_b64: str, prompt: str,
                        max_tokens: int) -> str:
    client = boto3.client(
        "bedrock-runtime",
        region_name=config["region"],
        aws_access_key_id=config["aws_access_key"],
        aws_secret_access_key=config["aws_secret_key"],
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": raw_b64,
                    },
                },
            ],
        }],
    })
    resp = client.invoke_model(modelId=config["model"], body=body)
    resp_body = json.loads(resp["body"].read())
    return resp_body["content"][0]["text"]
