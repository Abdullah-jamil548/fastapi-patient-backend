import logging
import asyncio
from groq import AsyncGroq
from app.config import settings
from app.core.exceptions import AppException

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI speech-therapy assistant helping parents whose children are learning Urdu pronunciation and speech sounds.

Provide supportive, clear, practical general information.

If the user asks in Urdu, respond in Urdu.
If the user asks in English, respond in English.

Do not claim to be a licensed clinician.
Do not diagnose medical or speech disorders.
Do not provide emergency medical advice.
When a question requires professional assessment, recommend consulting a qualified speech-language pathologist or appropriate healthcare professional."""

DEV_FALLBACK_REPLY = (
    "Hello! I am your Urdu speech learning assistant. Practice phonetics daily in a quiet room, "
    "focusing on one target letter at a time (like 'Bay' or 'Alif'). Celebrate every small win!"
)

# Groq retired these IDs for developer accounts. Map them so an old
# GROQ_MODEL env var still reaches a live model after deploy.
RETIRED_MODEL_REPLACEMENTS = {
    "llama-3.3-70b-versatile": "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b": "qwen/qwen3.8-27b",
}

PLACEHOLDER_API_KEYS = {
    "",
    "dummy_or_provide_in_env",
    "gsk_your_groq_api_key_here",
    "your_groq_api_key_here",
}


def candidate_models() -> list[str]:
    configured = [
        settings.GROQ_MODEL,
        settings.GROQ_FALLBACK_MODEL,
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
    ]
    models: list[str] = []
    for model in configured:
        if not model:
            continue
        resolved = RETIRED_MODEL_REPLACEMENTS.get(model.strip(), model.strip())
        if resolved and resolved not in models:
            models.append(resolved)
    return models


def completion_params(model: str, messages: list[dict]) -> dict:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.7,
    }
    # Instruct mode for parent Q&A. Reasoning models otherwise spend the
    # token budget on hidden thinking and return an empty answer.
    if model.startswith("qwen/"):
        params["reasoning_effort"] = "none"
    elif model.startswith("openai/gpt-oss"):
        params["reasoning_effort"] = "low"
        params["max_tokens"] = 1200
    return params


def reply_text(response) -> str | None:
    if not response.choices:
        return None
    content = getattr(response.choices[0].message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


class ChatbotService:
    def __init__(self):
        self._client = None
        api_key = (settings.GROQ_API_KEY or "").strip()
        if api_key and api_key not in PLACEHOLDER_API_KEYS:
            try:
                self._client = AsyncGroq(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")

    async def _complete(self, model: str, messages: list[dict]) -> str | None:
        params = completion_params(model, messages)
        attempts = [params]
        if "reasoning_effort" in params:
            attempts.append({key: value for key, value in params.items() if key != "reasoning_effort"})

        last_error = None
        for attempt in attempts:
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**attempt),
                    timeout=settings.CHAT_REQUEST_TIMEOUT,
                )
                text = reply_text(response)
                if text:
                    return text
                logger.error(f"Model {model} returned an empty reply")
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                logger.error(f"Error querying model {model}: {e}")
                last_error = e
        if last_error:
            raise last_error
        return None

    async def get_reply(self, message: str) -> str:
        if not self._client:
            # Fallback for dev/testing when no live API key is set
            return DEV_FALLBACK_REPLY

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]

        last_error = None
        timed_out = False

        for model in candidate_models():
            try:
                text = await self._complete(model, messages)
                if text:
                    return text
            except asyncio.TimeoutError:
                logger.error(f"Chatbot request timed out on {model}")
                timed_out = True
                last_error = TimeoutError(model)
            except Exception as e:
                logger.error(f"Error querying model {model}: {e}")
                last_error = e

        logger.error(f"All Groq models failed: {last_error}")
        if timed_out and isinstance(last_error, TimeoutError):
            raise AppException(status_code=504, detail="AI assistant is taking too long to respond. Please try again.")
        raise AppException(status_code=502, detail="AI assistant is temporarily unavailable. Please try again shortly.")

chatbot_service = ChatbotService()
