import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pytest
from pydantic import BaseModel

from src.application.services.chat_service import ChatService
from src.domain.schemas import ToolSelection, ChatTurn


# -----------------------
# Test doubles / helpers
# -----------------------

class DummyMCPClient:
    """Minimal MCPClient stand-in exposing .tools and LLM methods."""

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None):
        self.tools = tools or []
        self._create_calls: List[Dict[str, Any]] = []
        self._parse_calls: List[Dict[str, Any]] = []
        self.next_create_response: Dict[str, Any] = {}
        self.next_parse_response: Dict[str, Any] = {}

    def create_response(self, **kwargs) -> Dict[str, Any]:
        self._create_calls.append(kwargs)
        return self.next_create_response

    def parse_response(self, **kwargs) -> Dict[str, Any]:
        self._parse_calls.append(kwargs)
        return self.next_parse_response


class FakeResponseModel(BaseModel):
    """Provider response wrapper used in ChatService.send tests."""
    value: str

    def extract_text(self) -> str:
        return self.value


class FakeParsedResponseModel(BaseModel):
    """Provider parsed response wrapper used in ChatService.parse tests."""
    output_parsed: Optional[Dict[str, Any]] = None
    parsed: Optional[Dict[str, Any]] = None


class StructOut(BaseModel):
    """Target schema for Structured Outputs."""
    foo: int
    bar: str


def turns(*pairs: Iterable[str]) -> List[ChatTurn]:
    """Helper to build a list of ChatTurn quickly."""
    out: List[ChatTurn] = []
    for role, content in pairs:
        out.append(ChatTurn(role=role, content=content))
    return out


# -----------------------
# Conversation management
# -----------------------

def test_history_management_set_system_add_turn_reset():
    client = DummyMCPClient()
    chat = ChatService(client, system_instructions="sys seed")

    # system message applied
    assert chat.history[0]["role"] == "system"
    assert chat.history[0]["content"] == "sys seed"

    # add a user turn
    chat.add_turn("user", "hello")
    assert chat.history[-1]["role"] == "user"
    assert chat.history[-1]["content"] == "hello"

    # history property returns a copy
    h1 = chat.history
    h1.append({"role": "assistant", "content": "should not mutate original"})
    assert chat.history[-1]["content"] == "hello"

    # reset clears all
    chat.reset()
    assert chat.history == []


# -----------------------
# Tool selection
# -----------------------

def test_resolve_tools_none_disables_all():
    tools = [
        {"server_label": "aws-s3", "allowed_tools": ["list_s3_buckets"]},
        {"server_label": "git", "allowed_tools": ["search"]},
    ]
    client = DummyMCPClient(tools=tools)
    chat = ChatService(client)

    out = chat._resolve_tools(ToolSelection(value="none"))
    assert out is None


def test_resolve_tools_all_returns_all():
    tools = [
        {"server_label": "aws-s3", "allowed_tools": ["list_s3_buckets"]},
        {"server_label": "git", "allowed_tools": ["search"]},
    ]
    client = DummyMCPClient(tools=tools)
    chat = ChatService(client)

    out = chat._resolve_tools(ToolSelection(value="all"))
    assert out == tools


def test_resolve_tools_some_by_server_label():
    tools = [
        {"server_label": "aws-s3", "allowed_tools": ["list_s3_buckets"]},
        {"server_label": "git", "allowed_tools": ["search"]},
    ]
    client = DummyMCPClient(tools=tools)
    chat = ChatService(client)

    out = chat._resolve_tools(ToolSelection(value="some"), names=["aws-s3"])
    assert out == [tools[0]]


def test_resolve_tools_some_by_allowed_function():
    tools = [
        {"server_label": "aws-s3", "allowed_tools": ["list_s3_buckets"]},
        {"server_label": "git", "allowed_tools": ["search"]},
    ]
    client = DummyMCPClient(tools=tools)
    chat = ChatService(client)

    out = chat._resolve_tools(ToolSelection(value="some"), names=["search"])
    assert out == [tools[1]]


def test_resolve_tools_some_names_empty_returns_none():
    tools = [{"server_label": "aws-s3", "allowed_tools": ["list_s3_buckets"]}]
    client = DummyMCPClient(tools=tools)
    chat = ChatService(client)

    out = chat._resolve_tools(ToolSelection(value="some"), names=["   ", ""])
    assert out is None


# -----------------------
# send()
# -----------------------

def test_send_updates_history_and_returns_text():
    client = DummyMCPClient(tools=[{"server_label": "aws-s3", "allowed_tools": ["list_s3_buckets"]}])
    chat = ChatService(client, system_instructions="sys")

    # The client will return a dict that validates against FakeResponseModel
    client.next_create_response = {"value": "assistant-text"}

    out_text = chat.send(
        prompt="hello",
        model=None,
        instructions=None,
        tool_selection=ToolSelection(value="all"),
        response_model=FakeResponseModel,
        temperature=0.2,  # arbitrary passthrough kwargs
    )

    assert out_text == "assistant-text"

    # History updated with user + assistant turns
    assert chat.history[-2]["role"] == "user"
    assert chat.history[-2]["content"] == "hello"
    assert chat.history[-1]["role"] == "assistant"
    assert chat.history[-1]["content"] == "assistant-text"

    # Ensure tools were included (since selection=all)
    assert client._create_calls, "Expected create_response to be called"
    call = client._create_calls[-1]
    assert isinstance(call.get("tools"), list)
    assert call["input"][-1]["content"] == "hello"


def test_send_no_text_does_not_append_assistant():
    client = DummyMCPClient()
    chat = ChatService(client)

    # Response validates but extract_text returns empty
    class EmptyTextModel(BaseModel):
        value: str = ""

        def extract_text(self) -> str:
            return ""

    client.next_create_response = {"value": ""}
    result = chat.send(
        prompt="hi",
        model=None,
        instructions=None,
        tool_selection=ToolSelection(value="none"),
        response_model=EmptyTextModel,
    )
    assert result == ""
    # Only user turn should be appended (assistant not added when empty text)
    assert chat.history[-1]["role"] == "user"
    assert chat.history[-1]["content"] == "hi"


# -----------------------
# parse()
# -----------------------

def test_parse_returns_structured_model_from_output_parsed():
    client = DummyMCPClient()
    chat = ChatService(client)

    # Provider parsed response object with output_parsed populated
    client.next_parse_response = {"output_parsed": {"foo": 7, "bar": "ok"}}

    parsed = chat.parse(
        text_format=StructOut,
        prompt="give me structured",
        model=None,
        instructions=None,
        tool_selection=ToolSelection(value="all"),
        parsed_response_model=FakeParsedResponseModel,
    )

    assert isinstance(parsed, StructOut)
    assert parsed.foo == 7 and parsed.bar == "ok"

    # User turn appended
    assert chat.history[-1]["role"] == "user"
    assert chat.history[-1]["content"] == "give me structured"


def test_parse_returns_structured_model_from_parsed_field():
    client = DummyMCPClient()
    chat = ChatService(client)

    # Use 'parsed' instead of 'output_parsed'
    client.next_parse_response = {"parsed": {"foo": 1, "bar": "x"}}

    parsed = chat.parse(
        text_format=StructOut,
        prompt="other structured",
        model=None,
        instructions=None,
        tool_selection=ToolSelection(value="some"),
        tool_names=["anything"],  # ensure code path is exercised
        parsed_response_model=FakeParsedResponseModel,
    )

    assert isinstance(parsed, StructOut)
    assert parsed.foo == 1 and parsed.bar == "x"


def test_parse_returns_provider_object_when_no_structured_fields():
    client = DummyMCPClient()
    chat = ChatService(client)

    # No structured fields present
    client.next_parse_response = {}

    parsed_obj = chat.parse(
        text_format=StructOut,
        prompt="no structured",
        model=None,
        instructions=None,
        tool_selection=ToolSelection(value="none"),
        parsed_response_model=FakeParsedResponseModel,
    )

    # Should be instance of FakeParsedResponseModel (validated from dict)
    assert isinstance(parsed_obj, FakeParsedResponseModel)
    assert parsed_obj.output_parsed is None and parsed_obj.parsed is None