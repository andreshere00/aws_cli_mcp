import pytest

from src.infrastructure.mcp.client import MCPClient

# --- helpers (keep using your existing ones if already defined) ----------------


def _mk_tools(n: int, prefix: str = "srv"):
    """Build n fake MCP tool entries with distinct labels."""
    return [
        {
            "type": "mcp",
            "server_url": f"http://{i}",
            "server_label": f"{prefix}-{i}",
            "allowed_tools": ["fn"],
        }
        for i in range(n)
    ]


class DummyLLM:
    """Minimal test double that echoes the args it receives."""

    def __init__(self):
        self.last = None

    def create_response(self, **kwargs):
        self.last = ("create", kwargs)
        return kwargs

    def parse_response(self, **kwargs):
        self.last = ("parse", kwargs)
        return kwargs


class DummySchema:  # used only as an identity marker in tests
    pass


# --- fixed tests ---------------------------------------------------------------


@pytest.mark.parametrize(
    "explicit, kw, registry, expected",
    [
        # explicit arg wins
        (_mk_tools(1, "exp"), _mk_tools(1, "kw"), _mk_tools(1, "reg"), "exp-0"),
        # explicit None -> kwarg wins
        (None, _mk_tools(1, "kw"), _mk_tools(1, "reg"), "kw-0"),
        # explicit None, kwarg None -> registry wins
        (None, None, _mk_tools(1, "reg"), "reg-0"),
        # only registry
        (None, None, _mk_tools(2, "r"), "r-0"),
    ],
)
def test_create_response_tool_precedence(explicit, kw, registry, expected):
    llm = DummyLLM()
    client = MCPClient(llm=llm, tools=registry)

    # Build call kwargs carefully to avoid duplicate 'tools' parameter
    call_kwargs = dict(
        input=[{"role": "user", "content": "hi"}],
        model="m",
        instructions="sys",
        temperature=0.4,
    )
    if explicit is not None:
        call_kwargs["tools"] = explicit
    elif kw is not None:
        call_kwargs["tools"] = kw  # only when explicit is None

    result = client.create_response(**call_kwargs)

    sent_tools = result.get("tools")
    assert isinstance(sent_tools, list) and sent_tools, "tools must be a non-empty list"
    assert sent_tools[0]["server_label"].startswith(expected.split("-")[0])


def test_create_response_uses_client_defaults_for_model_and_instructions_when_none():
    llm = DummyLLM()
    client = MCPClient(
        llm=llm,
        tools=_mk_tools(1),
        default_model="def-model",
        default_instructions="def-sys",
    )

    res = client.create_response(
        input=[{"role": "user", "content": "q"}],
        model=None,  # should fall back to default_model
        instructions=None,  # should fall back to default_instructions
    )

    assert res["model"] == "def-model"
    assert res["instructions"] == "def-sys"
    # tools fallback to registry when neither explicit nor kw tools are provided
    assert isinstance(res["tools"], list) and res["tools"]


@pytest.mark.parametrize(
    "explicit, kw, registry, expected",
    [
        (_mk_tools(1, "exp"), _mk_tools(1, "kw"), _mk_tools(1, "reg"), "exp-0"),
        (None, _mk_tools(1, "kw"), _mk_tools(1, "reg"), "kw-0"),
        (None, None, _mk_tools(1, "reg"), "reg-0"),
    ],
)
def test_parse_response_tool_precedence(explicit, kw, registry, expected):
    llm = DummyLLM()
    client = MCPClient(llm=llm, tools=registry, default_model="dm", default_instructions="ds")

    call_kwargs = dict(
        text_format=DummySchema,
        input="prompt",
        model=None,  # uses default_model
        instructions=None,  # uses default_instructions
        messages=[{"role": "user", "content": "prompt"}],
        top_p=0.7,
    )
    if explicit is not None:
        call_kwargs["tools"] = explicit
    elif kw is not None:
        call_kwargs["tools"] = kw  # only when explicit is None

    res = client.parse_response(**call_kwargs)

    assert res["model"] == "dm"
    assert res["instructions"] == "ds"
    assert res["text_format"] is DummySchema
    assert res["messages"] == [{"role": "user", "content": "prompt"}]

    sent_tools = res.get("tools")
    assert isinstance(sent_tools, list) and sent_tools, "tools must be a non-empty list"
    assert sent_tools[0]["server_label"].startswith(expected.split("-")[0])
