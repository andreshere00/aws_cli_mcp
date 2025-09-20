from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from src.infrastructure.mcp import MCPClient
from src.domain.schemas import ChatTurn, ToolSelection
from src.domain.utils import to_dict

from pydantic import BaseModel
from typing import Type


class ChatService:
    """High-level chat service that orchestrates LLM calls and optional MCP tools.

    This service wraps an `MCPClient`, manages an in-memory conversation history,
    resolves which tools to expose per request, and returns the assistant's text
    response (or structured output via `parse`).

    Attributes:
      client: The MCP client facade used to reach the underlying LLM provider.
      _history: In-memory list of message dictionaries forming the conversation transcript.
    """

    def __init__(
        self,
        client: MCPClient,
        *,
        system_instructions: Optional[str] = None,
        history: Optional[Sequence[ChatTurn]] = None,
    ) -> None:
        """Initialize the chat service.

        Args:
          client: MCP client used to call the LLM and pass tool specs.
          system_instructions: Optional system message to seed the conversation.
          history: Optional prior chat turns to pre-populate the transcript.

        Notes:
          The provided `history` is appended to an internal list of dicts in
          OpenAI/Responses message format (role/content).
        """
        self.client = client
        self._history: List[Mapping[str, Any]] = []
        if system_instructions:
            self.set_system(system_instructions)
        if history:
            for t in history:
                self.add_turn(t.role, t.content)

    # -----------------------------
    # Conversation management
    # -----------------------------
    def reset(self) -> None:
        """Clear the entire chat history.

        This removes all prior system, user, assistant, and tool turns.
        """
        self._history.clear()

    def set_system(self, instructions: str) -> None:
        """Append a system message to the conversation.

        Args:
          instructions: System-level guidance that shapes model behavior.
        """
        self._history.append({"role": "system", "content": instructions})

    def add_turn(self, role: Literal["system", "user", "assistant", "tool"], content: str) -> None:
        """Append a message turn to the conversation history.

        Args:
          role: The role of the message ("system", "user", "assistant", or "tool").
          content: The message content as plain text.
        """
        self._history.append({"role": role, "content": content})

    @property
    def history(self) -> List[Mapping[str, Any]]:
        """Return a copy of the current conversation history.

        Returns:
          list[dict[str, Any]]: A shallow copy of the transcript in Responses
          message format.
        """
        return list(self._history)

    # -----------------------------
    # Tool selection
    # -----------------------------
    def _resolve_tools(
    self,
    selection: ToolSelection,
    names: Optional[Iterable[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Select which tools to expose for the current request.

        Args:
          selection: Strategy for exposing tools ("all", "none", "some").
          names: When `selection.value == "some"`, a set of server labels and/or
            tool function names to allow.

        Returns:
          list[dict[str, Any]] | None: A filtered list of tool specs in Responses
          MCP format, or `None` to disable tools.
        """
        if selection.value == "none":
            return None
        all_tools = self.client.tools
        if selection.value == "all" or not names:
            return all_tools if all_tools else None

        wanted = {n.strip() for n in names if n and n.strip()}
        if not wanted:
            return None

        selected: List[Dict[str, Any]] = []
        for spec in all_tools:
            label = str(spec.get("server_label", "")).strip()
            allowed = {str(t).strip() for t in spec.get("allowed_tools", [])}
            if label in wanted or (allowed & wanted):
                selected.append(spec)
        return selected or None

    # -----------------------------
    # Chat send
    # -----------------------------
    def send(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tool_selection: "ToolSelection",
        tool_names: Optional[Iterable[str]] = None,
        response_model: type = None,
        **kwargs: Any,
    ) -> str:
        """Send a user message and return the assistant's text reply.

        This method composes the history and new user turn, resolves tool exposure,
        calls the provider via `MCPClient.create_response`, validates the raw payload
        into the configured provider schema, and extracts assistant text.

        Args:
          prompt: User message to send.
          model: Optional model/deployment to use; falls back to client default.
          instructions: Optional per-call system instructions to prepend.
          tool_selection: Tool exposure policy (all/none/some).
          tool_names: When `tool_selection` is "some", the server labels and/or tool
            names to allow.
          response_model: Provider-specific Pydantic model (subclass of `BaseLLMResponse`)
            used to validate and extract assistant text; defaults to Azure schema.
          **kwargs: Additional provider parameters forwarded to the LLM call
            (e.g., temperature, max_output_tokens).

        Returns:
          str: Extracted assistant text from the provider response.

        Notes:
          The conversation history is updated with both the user turn and the
          assistant reply when text is successfully extracted.
        """
        messages: List[Mapping[str, Any]] = self.history
        messages.append({"role": "user", "content": prompt})

        tools = self._resolve_tools(tool_selection, tool_names)

        resp_dict = self.client.create_response(
            input=messages,
            model=model,
            instructions=instructions,
            tools=tools,
            **kwargs,
        )

        payload = to_dict(resp_dict)

        if response_model is None:
            from src.domain.schemas import AzureOpenAIResponse as DefaultResponseModel
            response_model = DefaultResponseModel

        resp_obj = response_model.model_validate(payload)
        text = resp_obj.extract_text()

        if text:
            self.add_turn("user", prompt)
            self.add_turn("assistant", text)

        return text
    
    def parse(
        self,
        *,
        text_format: Type[BaseModel],
        prompt: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tool_selection: "ToolSelection",
        tool_names: Optional[Iterable[str]] = None,
        parsed_response_model: type = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Call the provider's Structured Outputs API and return the parsed model.

        This method uses Responses.parse to validate the assistant's final answer
        against a Pydantic `text_format` schema, while still supporting tools.

        Args:
          text_format: Pydantic class describing the expected structured output.
          prompt: User message to send.
          model: Optional model/deployment to use; falls back to client default.
          instructions: Optional per-call system instructions to prepend.
          tool_selection: Tool exposure policy (all/none/some).
          tool_names: When `tool_selection` is "some", the server labels and/or tool
            names to allow.
          parsed_response_model: Provider-specific Pydantic model for the parse payload;
            defaults to Azure parsed schema.
          **kwargs: Additional provider parameters forwarded to the LLM call
            (e.g., temperature, max_output_tokens).

        Returns:
          BaseModel: The validated structured output if available; otherwise the
          provider-specific parsed response object.

        Notes:
          This method also appends the user turn to history; the assistant turn is not
          added because the returned object is structured, not a plain text message.
        """
        messages: List[Mapping[str, Any]] = self.history
        messages.append({"role": "user", "content": prompt})

        tools = self._resolve_tools(tool_selection, tool_names)

        resp_dict = self.client.parse_response(
            text_format=text_format,
            input=prompt,
            model=model,
            instructions=instructions,
            messages=messages,
            tools=tools,
            **kwargs,
        )

        if parsed_response_model is None:
            from src.domain.schemas import AzureOpenAIParsedResponse as DefaultParsedModel
            parsed_response_model = DefaultParsedModel

        payload = to_dict(resp_dict)
        resp_obj = parsed_response_model.model_validate(payload)
        self.add_turn("user", prompt)

        parsed = getattr(resp_obj, "output_parsed", None) or getattr(resp_obj, "parsed", None)
        if isinstance(parsed, BaseModel):
            return parsed
        if isinstance(parsed, dict):
            return text_format.model_validate(parsed)

        return resp_obj
