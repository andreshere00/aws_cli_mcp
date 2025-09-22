from __future__ import annotations

import json
from typing import Any, Dict

from pydantic import BaseModel

from src.domain.schemas import AzureOpenAIResponse


def to_dict(value: Any) -> Dict[str, Any]:
    """
    Convert SDK/Pydantic objects, JSON strings, or mappings to a plain dict.

    This is a unified coercion utility used across application/services to
    normalize provider responses before validating them into domain schemas.

    Args:
        value (Any): The input value to convert. It can be a dict, a Pydantic model,
            a JSON string, or an object with serialization methods like `model_dump`,
            `to_dict`, `dict`, or `json`.

    Returns:
        Dict[str, Any]: A plain dictionary representation of the input value. If the input
            cannot be converted, returns a dictionary with a single key `"raw"` containing
            the string representation of the input.
    """
    # Already a dict
    if isinstance(value, dict):
        return value

    # Pydantic model (v2)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    # JSON string
    if isinstance(value, str):
        return json.loads(value)

    # Pydantic-like objects
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return value.model_dump()

    # Common SDK shapes
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "json"):
        return json.loads(value.json())

    # Last resort: try to parse str(value) as JSON, else wrap
    try:
        return json.loads(str(value))
    except Exception:
        return {"raw": str(value)}


def as_text(reply: Any) -> str:
    """
    Return assistant text regardless of whether `reply` is a string, model, or SDK object.

    This function attempts to extract the textual content from various types of reply objects,
    including plain strings, objects with an `extract_text` method, or objects convertible
    to a dictionary that conforms to the AzureOpenAIResponse schema.

    Args:
        reply (Any): The reply object to extract text from. This can be:
            - A string, which is returned as is.
            - An object with an `extract_text()` method returning a string.
            - An object convertible to a dict that can be validated as an AzureOpenAIResponse.

    Returns:
        str: The extracted text from the reply. If extraction fails, returns the string
            representation of the reply.

    Examples:
        >>> as_text("hello")
        'hello'

        >>> class Reply:
        ...     def extract_text(self):
        ...         return "extracted text"
        ...
        >>> as_text(Reply())
        'extracted text'
    """
    if isinstance(reply, str):
        return reply
    if hasattr(reply, "extract_text"):
        try:
            t = reply.extract_text()
            if isinstance(t, str):
                return t
        except Exception:
            pass
    try:
        payload = to_dict(reply)
        return AzureOpenAIResponse.model_validate(payload).extract_text()
    except Exception:
        return str(reply)
