from typing import Any, Dict, Optional

import pytest

from src.infrastructure.llm.base_llm import BaseLLMProviderUnitOfWork

# -----------------------
# Concrete test doubles
# -----------------------


class DummyClient:
    def __init__(self, **kwargs: Any) -> None:
        self.ctor_kwargs = kwargs


class ConcreteUoW(BaseLLMProviderUnitOfWork):
    """Minimal concrete subclass for testing behavior."""

    def __init__(self, **kwargs: Any) -> None:
        self._build_calls = []
        self._create_calls = []
        self._parse_calls = []
        super().__init__(**kwargs)

    def _build_client(self, **kwargs: Any) -> Any:
        # Record the kwargs used to build the client and return a dummy client
        self._build_calls.append(kwargs)
        return DummyClient(**kwargs)

    def create_response(
        self,
        input: str,
        model: str,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Record the call and echo back a dict (as a provider would)
        payload = {
            "op": "create",
            "input": input,
            "model": model,
            "instructions": instructions,
            "extra": dict(kwargs),
        }
        self._create_calls.append(payload)
        return payload

    def parse_response(
        self,
        input: str,
        model: str,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "op": "parse",
            "input": input,
            "model": model,
            "instructions": instructions,
            "extra": dict(kwargs),
        }
        self._parse_calls.append(payload)
        return payload


# -----------------------
# Tests
# -----------------------


def test_abstract_base_cannot_be_instantiated_directly():
    # ABC with abstract methods should not allow direct instantiation
    with pytest.raises(TypeError):
        BaseLLMProviderUnitOfWork()  # type: ignore[abstract]


def test_init_calls_build_client_and_stores_client():
    uow = ConcreteUoW(api_key="k", endpoint="https://x/")
    # _build_client was called once with given kwargs
    assert len(uow._build_calls) == 1
    assert uow._build_calls[0] == {"api_key": "k", "endpoint": "https://x/"}
    # client is a DummyClient created from those kwargs
    assert isinstance(uow.client, DummyClient)
    assert uow.client.ctor_kwargs == {"api_key": "k", "endpoint": "https://x/"}


def test_create_response_minimal_args():
    uow = ConcreteUoW()
    out = uow.create_response(input="hello", model="my-model")
    assert out["op"] == "create"
    assert out["input"] == "hello"
    assert out["model"] == "my-model"
    assert out["instructions"] is None
    assert out["extra"] == {}
    # call recorded
    assert uow._create_calls[-1] == out


def test_create_response_with_instructions_and_kwargs():
    uow = ConcreteUoW()
    out = uow.create_response(
        input="ping",
        model="m1",
        instructions="be nice",
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=256,
        tools=[{"type": "mcp"}],
        tool_choice="auto",
        custom="abc",
    )
    assert out["op"] == "create"
    assert out["instructions"] == "be nice"
    assert out["extra"] == {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_output_tokens": 256,
        "tools": [{"type": "mcp"}],
        "tool_choice": "auto",
        "custom": "abc",
    }


def test_parse_response_minimal_args():
    uow = ConcreteUoW()
    out = uow.parse_response(input="structure this", model="m2")
    assert out["op"] == "parse"
    assert out["input"] == "structure this"
    assert out["model"] == "m2"
    assert out["instructions"] is None
    assert out["extra"] == {}
    assert uow._parse_calls[-1] == out


def test_parse_response_with_instructions_and_kwargs():
    uow = ConcreteUoW()
    out = uow.parse_response(
        input="make json",
        model="m3",
        instructions="return JSON",
        text_format="MySchema",  # simulate provider requirement
        previous_response_id="resp_123",
        temperature=0.0,
    )
    assert out["op"] == "parse"
    assert out["instructions"] == "return JSON"
    # All extra kwargs should be preserved
    assert out["extra"] == {
        "text_format": "MySchema",
        "previous_response_id": "resp_123",
        "temperature": 0.0,
    }


def test_multiple_instances_have_independent_clients():
    u1 = ConcreteUoW(tag="one")
    u2 = ConcreteUoW(tag="two")
    assert u1.client is not u2.client
    assert u1.client.ctor_kwargs == {"tag": "one"}
    assert u2.client.ctor_kwargs == {"tag": "two"}
