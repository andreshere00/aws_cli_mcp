from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type

from pydantic import BaseModel

from src.infrastructure.llm.base_llm import BaseLLMProviderUnitOfWork


class MCPClient:
    """Application-layer façade that combines an LLM unit-of-work with MCP tool configs.

    This class is provider-agnostic. You inject any `BaseLLMProviderUnitOfWork` and optional
    MCP `tools` specs (dicts in the OpenAI Responses format). The client forwards requests to
    the provider while enforcing a consistent precedence for tool selection.

    Attributes:
        llm (BaseLLMProviderUnitOfWork): Provider-specific LLM client wrapped in a unit-of-work.
        _tools (list[dict[str, Any]]): List of registered MCP tool specs.
        default_model (str | None): Default model to use if not provided in method calls.
        default_instructions (str | None): Default system instructions if not provided.
    """

    def __init__(
        self,
        *,
        llm: BaseLLMProviderUnitOfWork,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        default_model: Optional[str] = None,
        default_instructions: Optional[str] = None,
    ) -> None:
        """Initialize the client with an LLM UoW and optional MCP tool specs.

        Args:
            llm: Provider-specific LLM client.
            tools: Optional iterable of MCP tool specification dicts.
            default_model: Default model name for calls.
            default_instructions: Default system instructions for calls.
        """
        self.llm = llm
        self._tools: List[Dict[str, Any]] = list(tools or [])
        self.default_model = default_model
        self.default_instructions = default_instructions

    # ---- Tools registry -------------------------------------------------- #

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Return the current MCP tool specs.

        Returns:
            list[dict[str, Any]]: A copy of the registered MCP tool specs.
        """
        return list(self._tools)

    def add_tool(self, tool_spec: Dict[str, Any]) -> None:
        """Add a single MCP tool spec to the registry.

        Args:
            tool_spec: A dictionary describing an MCP tool.
        """
        self._tools.append(tool_spec)

    def add_tools(self, tool_specs: Iterable[Dict[str, Any]]) -> None:
        """Add multiple MCP tool specs to the registry.

        Args:
            tool_specs: An iterable of MCP tool specification dicts.
        """
        self._tools.extend(tool_specs)

    # ---- High-level send/parse ------------------------------------------ #

    def create_response(
        self,
        *,
        input: Any,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a single-turn response using the injected LLM and resolved tools.

        Tool precedence is applied in this order:
        1. Explicit `tools` argument
        2. `tools` found in kwargs
        3. Registry (`self._tools`)

        Args:
            input: Input payload (string, list of messages, etc.).
            model: Override model name for this call.
            instructions: Override system instructions for this call.
            tools: Explicit tool specs to use instead of registry.
            **kwargs: Additional provider-specific options.

        Returns:
            Any: The raw provider response object.
        """
        kw_tools = kwargs.pop("tools", None)

        if tools is not None:
            resolved_tools = tools
        elif kw_tools is not None:
            resolved_tools = kw_tools
        elif self._tools:
            resolved_tools = list(self._tools)
        else:
            resolved_tools = None

        return self.llm.create_response(
            input=input,
            model=model or self.default_model,
            instructions=instructions or self.default_instructions,
            tools=resolved_tools,
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
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Call provider `parse_response` with structured outputs and resolved tools.

        Tool precedence is applied in this order:
        1. Explicit `tools` argument
        2. `tools` in kwargs
        3. Registry (`self._tools`)

        Args:
            text_format: Pydantic model describing expected output schema.
            input: User text or message content.
            model: Override model name for this call.
            instructions: Override system instructions for this call.
            messages: Optional full message list (role/content dicts).
            tools: Explicit tool specs to use instead of registry.
            **kwargs: Additional provider-specific options.

        Returns:
            Any: The provider-parsed response, often an instance of `text_format`.
        """
        kw_tools = kwargs.pop("tools", None)

        if tools is not None:
            resolved_tools = tools
        elif kw_tools is not None:
            resolved_tools = kw_tools
        elif self._tools:
            resolved_tools = list(self._tools)
        else:
            resolved_tools = None

        return self.llm.parse_response(
            text_format=text_format,
            input=input,
            model=model or self.default_model,
            instructions=instructions or self.default_instructions,
            messages=messages,
            tools=resolved_tools,
            **kwargs,
        )
