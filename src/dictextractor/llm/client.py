"""
Unified LLM client wrapping litellm.
Handles API key resolution, model routing, and provider-specific configuration.

Reasoning / thinking behaviour by model family:
  - Gemini 2.5:  old extra_body thinkingConfig approach; we disable it by default
                 (cannot be disabled on 2.5-pro, but we try on others).
  - Gemini 3+:   use litellm's `reasoning_effort` parameter which maps to thinking_level.
                 "low"  → thinking_level: low  (fast, cheap)
                 "high" → thinking_level: high (deep reasoning)
                 "none" also maps to "low" — thinking cannot be fully disabled on Gemini 3.
  - Gemini 3+ temperature: must be 1.0; litellm sets this automatically, but we honour
                 the caller's temperature for non-Gemini-3 models.
"""

import os
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

ReasoningEffort = Literal["none", "low", "medium", "high"]
T = TypeVar("T", bound=BaseModel)


def _resolve_api_key(model: str) -> Optional[str]:
    """Resolve the API key for a given model string based on provider prefix."""
    model_lower = model.lower()
    if "openrouter" in model_lower:
        return os.getenv("OPEN_ROUTER_API_KEY")
    if "gemini" in model_lower or "google" in model_lower:
        return os.getenv("GEMINI_API_KEY")
    if "claude" in model_lower or "anthropic" in model_lower:
        return os.getenv("ANTHROPIC_API_KEY")
    if "gpt" in model_lower or "openai" in model_lower:
        return os.getenv("OPENAI_API_KEY")
    return None


def _is_gemini3(model: str) -> bool:
    """Return True for Gemini 3+ model strings."""
    m = model.lower()
    return "gemini" in m and any(
        tag in m for tag in ("gemini-3", "gemini-3-flash", "gemini-3-pro", "gemini-3.1")
    )


def _is_gemini25(model: str) -> bool:
    """Return True for Gemini 2.5 model strings."""
    m = model.lower()
    return "gemini" in m and "2.5" in m


def _build_params(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[ReasoningEffort],
) -> Dict[str, Any]:
    """Assemble the litellm.completion kwargs, applying model-family-specific rules."""
    api_key = _resolve_api_key(model)
    if not api_key:
        print(f"Warning: No API key found for model '{model}'. Relying on environment variables.")

    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if api_key:
        params["api_key"] = api_key

    if _is_gemini3(model):
        effort = reasoning_effort or "low"
        params["reasoning_effort"] = effort
        print(f"  [Gemini 3] reasoning_effort={effort} (temperature fixed at 1.0 by litellm)")
    elif _is_gemini25(model):
        params["temperature"] = temperature
        if reasoning_effort in (None, "none", "low"):
            params["extra_body"] = {
                "generationConfig": {"thinking": {"thinkingConfig": {"mode": "DISABLED"}}}
            }
            print("  [Gemini 2.5] thinking disabled via extra_body")
        else:
            params["reasoning_effort"] = reasoning_effort
            print(f"  [Gemini 2.5] reasoning_effort={reasoning_effort}")
    else:
        params["temperature"] = temperature
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

    return params


def complete(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 64000,
    reasoning_effort: Optional[ReasoningEffort] = None,
) -> str:
    """
    Send a chat completion request via litellm and return the response text.

    Args:
        model: litellm-compatible model string.
        messages: Chat messages in OpenAI format.
        temperature: Sampling temperature. Ignored for Gemini 3+ (locked to 1.0).
        max_tokens: Maximum response tokens.
        reasoning_effort: Controls the thinking/reasoning budget.

    Returns:
        Raw response content string.
    """
    params = _build_params(model, messages, temperature, max_tokens, reasoning_effort)

    print(f"Calling LLM API with model: {model}...")
    response = litellm.completion(**params)

    print(f"Response received. Model: {response.model}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Usage: {response.usage}")

    content = response.choices[0].message.content
    if content is None:
        raise ValueError(
            "API returned None content. This may be due to token limits or API configuration."
        )

    print(f"Content length: {len(content)} characters")
    return content


def complete_structured(
    model: str,
    messages: List[Dict[str, Any]],
    response_schema: Type[T],
    temperature: float = 0.1,
    max_tokens: int = 64000,
    reasoning_effort: Optional[ReasoningEffort] = None,
) -> tuple[T, str, Dict[str, Any]]:
    """
    Send a chat completion request with structured output enforcement.

    Returns:
        (parsed_model, raw_json_str, usage_dict) where usage_dict contains
        token counts, image tokens, and cost_usd for this call.
    """
    params = _build_params(model, messages, temperature, max_tokens, reasoning_effort)
    params["response_format"] = response_schema

    print(f"Calling LLM API (structured) with model: {model} → {response_schema.__name__}")
    response = litellm.completion(**params)

    print(f"Response received. Model: {response.model}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Usage: {response.usage}")

    content = response.choices[0].message.content
    if content is None:
        raise ValueError(
            "API returned None content. This may be due to token limits or API configuration."
        )

    usage = _extract_usage(model, response)
    return response_schema.model_validate_json(content), content, usage


def _extract_usage(model: str, response) -> Dict[str, Any]:
    """Extract token counts and estimated cost from a litellm completion response."""
    u = response.usage
    usage: Dict[str, Any] = {
        "model": model,
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }

    # Pull out image / text token breakdown when available (Gemini reports these)
    pd = getattr(u, "prompt_tokens_details", None)
    if pd:
        usage["image_tokens"] = getattr(pd, "image_tokens", None)
        usage["text_tokens"] = getattr(pd, "text_tokens", None)

    # Attempt cost calculation via litellm's built-in pricing table
    try:
        cost = litellm.completion_cost(completion_response=response)
        usage["cost_usd"] = round(cost, 8)
    except Exception:
        usage["cost_usd"] = None

    return usage
