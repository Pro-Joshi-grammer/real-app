import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

def load_config():
    """Load environment config and return an AsyncOpenAI client + model names."""
    load_dotenv()
    ai_base_url = os.getenv("AI_BASE_URL", "https://api.hncsec.com/v1").rstrip("/")
    if not ai_base_url:
        raise ValueError("AI_BASE_URL is required")
    ai_api_key = os.getenv("AI_API_KEY", "sk-awhDHGsugmGAZyTFK7WfoXFbuRBUcIu1uq5Jpdnz6eeksXs7")
    if not ai_api_key:
        raise ValueError("AI_API_KEY is required")

    client = AsyncOpenAI(
        base_url=ai_base_url,
        api_key=ai_api_key,
    )
    return {
        "client": client,
        "text_model": os.getenv("TEXT_MODEL", ""),
        "image_model": os.getenv("IMAGE_MODEL", ""),
        "max_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "50")),
        "timeout": int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
    }
