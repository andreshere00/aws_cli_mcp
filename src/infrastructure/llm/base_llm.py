from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict


class BaseLLMProviderUnitOfWork(ABC):
    """Abstract base for Responses-capable LLM clients, designed for reuse across providers."""

    def __init__(self, **kwargs) -> None:
        """Instantiate the underlying client via the provider-specific builder."""
        self.client = self._build_client(**kwargs)

    # ---- Provider hooks -------------------------------------------------

    @abstractmethod
    def _build_client(self, **kwargs) -> Any:
        """Create and return the provider client (SDK handle)."""

    # ---- Core interface (minimal required params) -----------------------

    @abstractmethod
    def create_response(
        self,
        input: str,
        model: str,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a single-turn response; subclasses may accept extra kwargs."""

    @abstractmethod
    def parse_response(
        self,
        input: str,
        model: str,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create and parse a response into a Pydantic model; subclasses may require `text_format` via kwargs."""