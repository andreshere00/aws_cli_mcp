import json
from typing import Any, Dict

import pytest
from pydantic import BaseModel

import src.domain.tools.base_tool as base_tool
from src.domain.tools.base_tool import BaseMCPTool

# -----------------------
# Test doubles / stubs
# -----------------------


class StubFastMCP:
    """Stub of FastMCP capturing constructor arg and run() calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.run_calls = []

    def run(self, *, transport: str, host: str, port: int, path: str) -> None:
        self.run_calls.append({"transport": transport, "host": host, "port": port, "path": path})


class PModel(BaseModel):
    a: int
    b: str


class HasModelDump:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        # ignore mode, just return payload
        return self._data


# -----------------------
# BaseMCPTool.__init__
# -----------------------


def test_init_creates_fastmcp_app(monkeypatch):
    # Patch the FastMCP symbol used inside the module
    monkeypatch.setattr(base_tool, "FastMCP", StubFastMCP)

    tool = BaseMCPTool(name="My App")
    assert isinstance(tool.mcp, StubFastMCP)
    assert tool.mcp.name == "My App"


# -----------------------
# to_dict
# -----------------------


def test_to_dict_with_pydantic_model():
    model = PModel(a=1, b="x")
    out = BaseMCPTool.to_dict(model)
    assert out == {"a": 1, "b": "x"}


def test_to_dict_with_json_string():
    s = '{"k": 1, "v": "z"}'
    out = BaseMCPTool.to_dict(s)
    assert out == {"k": 1, "v": "z"}


def test_to_dict_with_dict_passthrough():
    d = {"k": 2, "v": "y"}
    out = BaseMCPTool.to_dict(d)
    assert out is d  # same object


def test_to_dict_with_has_model_dump():
    obj = HasModelDump({"x": 10})
    out = BaseMCPTool.to_dict(obj)
    assert out == {"x": 10}


def test_to_dict_with_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        BaseMCPTool.to_dict("{not-json}")


def test_to_dict_with_unsupported_type_raises():
    with pytest.raises(TypeError) as ex:
        BaseMCPTool.to_dict(123)
    assert "unsupported type" in str(ex.value)


# -----------------------
# register_tools abstract
# -----------------------


def test_register_tools_not_implemented(monkeypatch):
    # Ensure FastMCP is patched to avoid importing real fastmcp
    monkeypatch.setattr(base_tool, "FastMCP", StubFastMCP)
    tool = BaseMCPTool("X")
    with pytest.raises(NotImplementedError):
        tool.register_tools()


# -----------------------
# run() wiring
# -----------------------


def test_run_calls_register_and_mcp_run_with_defaults(monkeypatch):
    # Patch FastMCP with stub to capture run calls
    monkeypatch.setattr(base_tool, "FastMCP", StubFastMCP)

    called = {"register": 0}

    class Impl(BaseMCPTool):
        def register_tools(self) -> None:  # override abstract
            called["register"] += 1

    t = Impl("Runner")
    # Sanity
    assert isinstance(t.mcp, StubFastMCP)

    # Call run with defaults
    t.run()

    # register_tools called once
    assert called["register"] == 1

    # mcp.run called once with defaults
    assert len(t.mcp.run_calls) == 1
    rc = t.mcp.run_calls[0]
    assert rc == {
        "transport": "streamable-http",
        "host": "0.0.0.0",
        "port": 8000,
        "path": "/mcp",
    }


def test_run_calls_register_and_mcp_run_with_custom_args(monkeypatch):
    monkeypatch.setattr(base_tool, "FastMCP", StubFastMCP)

    class Impl(BaseMCPTool):
        def register_tools(self) -> None:
            """
            Dummy method to register tools
            """
            pass

    t = Impl("Runner2")
    t.run(transport="sse", host="127.0.0.1", port=9001, path="/mcp/s3")

    assert len(t.mcp.run_calls) == 1
    rc = t.mcp.run_calls[0]
    assert rc == {
        "transport": "sse",
        "host": "127.0.0.1",
        "port": 9001,
        "path": "/mcp/s3",
    }
