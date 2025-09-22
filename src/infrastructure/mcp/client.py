from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type

from pydantic import BaseModel

from src.infrastructure.llm.base_llm import BaseLLMProviderUnitOfWork


class MCPClient:
    """Application-layer façade that combines an LLM unit-of-work with MCP tool configs.

    This class is provider-agnostic: inject any `BaseLLMProviderUnitOfWork` and a list
    of MCP `tools` dicts (e.g., created by your `MCPToolConfig.model_dump()`), and use
    it from services to create or parse responses with the same toolset.
    """

    def __init__(
        self,
        *,
        llm: BaseLLMProviderUnitOfWork,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        default_model: Optional[str] = None,
        default_instructions: Optional[str] = None,
    ) -> None:
        """Initialize the client with an LLM UoW and optional MCP tool specs."""
        self.llm = llm
        self._tools: List[Dict[str, Any]] = list(tools or [])
        self.default_model = default_model
        self.default_instructions = default_instructions

    # ---- Tools registry -------------------------------------------------- #

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Return the current MCP tool specs (OpenAI Responses `tools` payload)."""
        return list(self._tools)

    def add_tool(self, tool_spec: Dict[str, Any]) -> None:
        """Add a single MCP tool spec (dict) to the registry."""
        self._tools.append(tool_spec)

    def add_tools(self, tool_specs: Iterable[Dict[str, Any]]) -> None:
        """Add multiple MCP tool specs (dicts) to the registry."""
        self._tools.extend(tool_specs)

    # ---- High-level send/parse ------------------------------------------ #

    def create_response(
        self,
        *,
        input: Any,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a single-turn response using the injected LLM and resolved tools.

        The precedence for tools is: explicit arg `tools` > kwarg `tools` > self.tools.
        This prevents "multiple values for keyword 'tools'" when callers also pass tools.
        """
        # If caller provided tools inside kwargs, extract them first
        kw_tools = kwargs.pop("tools", None)
        tools = tools if self.tools else kw_tools

        return self.llm.create_response(
            input=input,
            model=model or self.default_model,
            instructions=instructions or self.default_instructions,
            tools=tools,
            **kwargs,
        )

    def parse_response(
        self,
        *,
        text_format: Type[BaseModel],
        input: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        messages: Optional[Sequence[Mapping[str, Any]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Call provider `parse_response` with structured outputs and resolved tools.

        The precedence for tools is: explicit arg `tools` > kwarg `tools` > self.tools.
        """
        kw_tools = kwargs.pop("tools", None)
        tools = tools if self.tools else kw_tools

        return self.llm.parse_response(
            text_format=text_format,
            input=input,
            model=model or self.default_model,
            instructions=instructions or self.default_instructions,
            messages=messages,
            tools=tools,
            **kwargs,
        )
