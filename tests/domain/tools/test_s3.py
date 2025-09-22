from typing import Any, Dict, Optional

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

# Module paths under test
import src.domain.tools.base_tool as base_tool

# -----------------------
# Test doubles / stubs
# -----------------------


class StubFastMCP:
    """Stub FastMCP capturing the name and registered tools."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: Dict[str, Any] = {}
        self.run_calls = []

    def tool(self, fn):
        # Decorator: record by function name and return original fn
        self._tools[fn.__name__] = fn
        return fn

    def run(self, *, transport: str, host: str, port: int, path: str) -> None:
        self.run_calls.append({"transport": transport, "host": host, "port": port, "path": path})


class StubUoW:
    """Stub S3UnitOfWork with programmable return/exception."""

    def __init__(self) -> None:
        self.calls = []

        # By default a harmless return
        self.next_return: Any = {"Buckets": [], "Owner": {"DisplayName": "me", "ID": "abc"}}
        self.next_raise: Optional[BaseException] = None

    def list_buckets(self, *, prefix: Optional[str], region: Optional[str], max_buckets: int):
        self.calls.append({"prefix": prefix, "region": region, "max_buckets": max_buckets})
        if self.next_raise:
            raise self.next_raise
        return self.next_return


class HasModelDump:
    """Object with model_dump(mode='json') to simulate Pydantic-like object."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        return dict(self._payload)


# -----------------------
# Import the S3MCPTool with patches
# -----------------------


@pytest.fixture(autouse=True)
def patch_fastmcp(monkeypatch):
    """Ensure BaseMCPTool uses the StubFastMCP (no real server)."""
    monkeypatch.setattr(base_tool, "FastMCP", StubFastMCP)
    yield


@pytest.fixture
def s3_module(monkeypatch):
    """Import S3MCPTool module with S3UnitOfWork patched to our Stub."""
    # Patch where S3MCPTool imports it from
    import src.infrastructure.aws as infra_aws

    monkeypatch.setattr(infra_aws, "S3UnitOfWork", StubUoW, raising=True)

    # Import the S3 tool module fresh
    import importlib

    mod = importlib.import_module("src.domain.tools.s3")
    importlib.reload(mod)
    return mod


# -----------------------
# __init__
# -----------------------


def test_init_uses_fastmcp_and_default_uow(s3_module):
    tool = s3_module.S3MCPTool()
    assert isinstance(tool.mcp, StubFastMCP)
    assert tool.mcp.name == "AWS S3"
    assert isinstance(tool.uow, StubUoW)


def test_init_can_inject_custom_uow(s3_module):
    custom = StubUoW()
    tool = s3_module.S3MCPTool(uow=custom)
    assert tool.uow is custom


# -----------------------
# _list_s3_buckets_impl: success paths
# -----------------------


def test_list_impl_returns_dict_passthrough(s3_module):
    tool = s3_module.S3MCPTool()
    # Default StubUoW returns a dict
    out = tool._list_s3_buckets_impl(prefix="pre", region="eu-west-1", max_buckets=5)
    assert "Buckets" in out and "Owner" in out
    # UoW call captured
    assert tool.uow.calls[-1] == {"prefix": "pre", "region": "eu-west-1", "max_buckets": 5}


def test_list_impl_handles_model_dump_object(s3_module):
    tool = s3_module.S3MCPTool()
    payload = {"Buckets": [{"Name": "x"}], "Owner": {"DisplayName": "me", "ID": "1"}}
    tool.uow.next_return = HasModelDump(payload)

    out = tool._list_s3_buckets_impl(prefix=None, region=None, max_buckets=10)
    assert out == payload


# -----------------------
# _list_s3_buckets_impl: AWS/generic errors
# -----------------------


def test_list_impl_handles_client_error(s3_module):
    tool = s3_module.S3MCPTool()
    tool.uow.next_raise = ClientError(
        error_response={"Error": {"Code": "ExpiredToken", "Message": "Expired"}},
        operation_name="ListBuckets",
    )

    out = tool._list_s3_buckets_impl(prefix=None, region=None, max_buckets=100)
    assert "error" in out
    assert "S3 error" in out["error"]
    assert "ExpiredToken" in out["error"] or "Expired" in out["error"]


def test_list_impl_handles_no_credentials(s3_module):
    tool = s3_module.S3MCPTool()
    tool.uow.next_raise = NoCredentialsError()

    out = tool._list_s3_buckets_impl(prefix=None, region=None, max_buckets=100)
    assert "error" in out
    assert "S3 error" in out["error"]


def test_list_impl_handles_generic_exception(s3_module):
    tool = s3_module.S3MCPTool()
    tool.uow.next_raise = RuntimeError("boom")

    out = tool._list_s3_buckets_impl(prefix="p", region="r", max_buckets=1)
    assert "error" in out
    assert out["error"].startswith("Unexpected error:")


# -----------------------
# register_tools wiring
# -----------------------


def test_register_tools_binds_list_s3_buckets(s3_module, monkeypatch):
    tool = s3_module.S3MCPTool()

    # Spy on impl to ensure it is called with forwarded args
    called = {}

    def spy_impl(**kwargs):
        called.update(kwargs)
        return {"ok": True, "args": kwargs}

    monkeypatch.setattr(tool, "_list_s3_buckets_impl", spy_impl)

    tool.register_tools()
    # The decorator stored tools on the FastMCP stub
    assert "list_s3_buckets" in tool.mcp._tools

    # Call the registered tool as FastMCP would
    result = tool.mcp._tools["list_s3_buckets"](prefix="pre", region="us-east-1", max_buckets=42)

    assert result == {
        "ok": True,
        "args": {"prefix": "pre", "region": "us-east-1", "max_buckets": 42},
    }
    assert called == {"prefix": "pre", "region": "us-east-1", "max_buckets": 42}


def test_register_tools_return_shape_docstring_example(s3_module):
    """Smoke test the default behavior without monkeypatching the impl."""
    tool = s3_module.S3MCPTool()
    tool.register_tools()
    fn = tool.mcp._tools["list_s3_buckets"]

    # Use defaults (None, None, 100) → should return dict (from StubUoW default)
    out = fn()
    assert isinstance(out, dict)
    assert "Buckets" in out and "Owner" in out
