import os
from typing import List, Optional

import pytest
from pydantic import BaseModel

# Module under test
import src.infrastructure.llm.azure_openai_uow as uow_mod
from src.infrastructure.llm.azure_openai_uow import AzureOpenAIUnitOfWork

# -----------------------
# Test doubles / stubs
# -----------------------


class CaptureCall:
    """Simple struct-like recorder."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubResponsesAPI:
    """Stub for client.responses with create/parse capturing kwargs."""

    def __init__(self):
        self.create_calls: List[CaptureCall] = []
        self.parse_calls: List[CaptureCall] = []

    def create(self, **kwargs):
        self.create_calls.append(CaptureCall(**kwargs))
        # Return something dict-like (the real SDK returns a model; tests inspect kwargs)
        return {"kind": "create", "captured": kwargs}

    def parse(self, **kwargs):
        self.parse_calls.append(CaptureCall(**kwargs))
        return {"kind": "parse", "captured": kwargs}


class StubAzureClient:
    """Stub for AzureOpenAI client; stores ctor kwargs and a responses stub."""

    def __init__(self, **kwargs):
        self.ctor_kwargs = kwargs
        self.responses = StubResponsesAPI()


class StubAzureOpenAIClass:
    """Callable class used to replace AzureOpenAI symbol in module under test."""

    last_instance: Optional[StubAzureClient] = None

    def __call__(self, **kwargs):
        client = StubAzureClient(**kwargs)
        StubAzureOpenAIClass.last_instance = client
        return client


class DummySchema(BaseModel):
    field: str = "x"


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture(autouse=True)
def patch_azure_openai(monkeypatch):
    """
    Patch the AzureOpenAI symbol used inside the module so we don't hit the network;
    also reset environment variable side effects between tests.
    """
    stub_ctor = StubAzureOpenAIClass()
    monkeypatch.setattr(uow_mod, "AzureOpenAI", stub_ctor)
    # Ensure env var is unset by default; tests set it explicitly as needed
    if "AZURE_OPENAI_DEPLOYMENT" in os.environ:
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    yield
    # cleanup is automatic; monkeypatch teardown resets attributes


# -----------------------
# __init__
# -----------------------


def test_init_forwards_kwargs_to_sdk_constructor():
    AzureOpenAIUnitOfWork(
        api_key="k", azure_endpoint="https://ex/", api_version="2025-01-01-preview"
    )
    client = uow_mod.AzureOpenAI.last_instance  # type: ignore[attr-defined]
    assert isinstance(client, StubAzureClient)
    assert client.ctor_kwargs == {
        "api_key": "k",
        "azure_endpoint": "https://ex/",
        "api_version": "2025-01-01-preview",
    }


# -----------------------
# create_response
# -----------------------


def test_create_response_uses_env_model_when_none(monkeypatch):
    # Model fallback via env
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep-from-env")
    uow = AzureOpenAIUnitOfWork()

    result = uow.create_response(
        input="Hello",
        model=None,  # fallback to env
        instructions="sys",
        max_output_tokens=321,
        metadata={"k": "v"},
        temperature=0.3,
        top_p=0.9,
        truncation="disabled",
        timeout=12.5,
        tools=[{"type": "mcp", "server_label": "x"}],
        tool_choice="auto",
    )

    # Returned stub payload
    assert isinstance(result, dict) and result["kind"] == "create"

    client = uow_mod.AzureOpenAI.last_instance  # type: ignore[attr-defined]
    call = client.responses.create_calls[-1]  # type: ignore[attr-defined]
    args = call.kwargs

    # Fallback model used
    assert args["model"] == "dep-from-env"
    # Forwarded fields
    assert args["input"] == "Hello"
    assert args["instructions"] == "sys"
    assert args["max_output_tokens"] == 321
    assert args["metadata"] == {"k": "v"}
    assert args["temperature"] == 0.3
    assert args["top_p"] == 0.9
    assert args["truncation"] == "disabled"
    assert args["timeout"] == 12.5
    assert args["tools"] == [{"type": "mcp", "server_label": "x"}]
    assert args["tool_choice"] == "auto"


def test_create_response_minimal_happy_path():
    uow = AzureOpenAIUnitOfWork()
    out = uow.create_response(input="Hi", model="my-deploy", instructions=None)
    assert out["kind"] == "create"
    client = uow_mod.AzureOpenAI.last_instance  # type: ignore[attr-defined]
    args = client.responses.create_calls[-1].kwargs  # type: ignore[attr-defined]
    assert args["model"] == "my-deploy"
    assert args["input"] == "Hi"
    assert args["instructions"] is None
    # Defaults applied
    assert args["temperature"] == 0.2
    assert args["truncation"] == "auto"
    assert args["timeout"] == 60


# -----------------------
# parse_response
# -----------------------


def test_parse_response_builds_minimal_messages_when_missing(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep-env")
    uow = AzureOpenAIUnitOfWork()

    result = uow.parse_response(
        text_format=DummySchema,
        messages=None,
        input="User says hi",
        instructions="Be terse.",
        model=None,  # fallback
        max_output_tokens=111,
        temperature=0.1,
        top_p=0.8,
        truncation="disabled",
        timeout=5,
        tools=[{"type": "mcp"}],
        tool_choice="none",
        previous_response_id="resp_prev",
    )

    assert result["kind"] == "parse"
    client = uow_mod.AzureOpenAI.last_instance  # type: ignore[attr-defined]
    args = client.responses.parse_calls[-1].kwargs  # type: ignore[attr-defined]

    # Model fallback
    assert args["model"] == "dep-env"
    # Messages constructed from instructions + input
    assert isinstance(args["input"], list) and len(args["input"]) == 2
    assert args["input"][0] == {"role": "system", "content": "Be terse."}
    assert args["input"][1] == {"role": "user", "content": "User says hi"}
    # Schema + knobs + tools
    assert args["text_format"] is DummySchema
    assert args["max_output_tokens"] == 111
    assert args["temperature"] == 0.1
    assert args["top_p"] == 0.8
    assert args["truncation"] == "disabled"
    assert args["timeout"] == 5
    assert args["tools"] == [{"type": "mcp"}]
    assert args["tool_choice"] == "none"
    assert args["previous_response_id"] == "resp_prev"


def test_parse_response_uses_provided_messages_and_ignores_input_instructions():
    uow = AzureOpenAIUnitOfWork()
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "ask"}]
    res = uow.parse_response(
        text_format=DummySchema,
        messages=msgs,
        input="ignored",
        instructions="ignored",
        model="deploy-1",
    )
    assert res["kind"] == "parse"
    client = uow_mod.AzureOpenAI.last_instance  # type: ignore[attr-defined]
    args = client.responses.parse_calls[-1].kwargs  # type: ignore[attr-defined]
    # Input is the messages list exactly
    assert args["input"] == msgs
    assert args["model"] == "deploy-1"
    assert args["text_format"] is DummySchema


def test_parse_response_raises_when_no_messages_and_no_input():
    uow = AzureOpenAIUnitOfWork()
    with pytest.raises(ValueError):
        uow.parse_response(
            text_format=DummySchema,
            messages=None,
            input=None,
            instructions=None,
            model="dep",
        )


def test_parse_response_defaults_are_applied():
    uow = AzureOpenAIUnitOfWork()
    res = uow.parse_response(
        text_format=DummySchema,
        messages=[{"role": "user", "content": "hi"}],
        model="dep",
    )
    assert res["kind"] == "parse"
    client = uow_mod.AzureOpenAI.last_instance  # type: ignore[attr-defined]
    args = client.responses.parse_calls[-1].kwargs  # type: ignore[attr-defined]
    assert args["temperature"] == 0.2
    assert args["truncation"] == "auto"
    assert args["timeout"] == 60
    # None by default if not provided
    assert args["tools"] is None
    assert args["tool_choice"] == "auto"
    assert args["max_output_tokens"] is None
    assert args["top_p"] is None
    assert args["previous_response_id"] is None
