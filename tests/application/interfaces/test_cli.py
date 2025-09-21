import json
import sys
from types import SimpleNamespace
from typing import Any, Dict
import argparse

import pytest

# Import the CLI module under test
import src.application.interfaces.cli as cli


# -----------------------
# Fixtures & Test Doubles
# -----------------------

class DummyBaseModel:
    """Minimal stand-in for pydantic BaseModel with model_dump_json()."""
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def model_dump_json(self, indent: int = 2) -> str:
        return json.dumps(self._data, indent=indent)


@pytest.fixture
def tools_file_dict(tmp_path):
    p = tmp_path / "tools.json"
    p.write_text(json.dumps({
        "type": "mcp",
        "server_label": "dummy",
        "server_url": "http://localhost:9999/mcp/dummy",
        "allowed_tools": ["x"],
        "require_approval": "never",
    }), encoding="utf-8")
    return str(p)


@pytest.fixture
def tools_file_list(tmp_path):
    p = tmp_path / "tools.json"
    p.write_text(json.dumps([
        {
            "type": "mcp",
            "server_label": "dummy",
            "server_url": "http://localhost:9999/mcp/dummy",
            "allowed_tools": ["x"],
            "require_approval": "never",
        },
        {
            "type": "mcp",
            "server_label": "dummy2",
            "server_url": "http://localhost:9999/mcp/dummy2",
            "allowed_tools": ["y"],
            "require_approval": "never",
        }
    ]), encoding="utf-8")
    return str(p)


# -------------
# _load_tools_file
# -------------

def test_load_tools_file_dict(tools_file_dict):
    out = cli._load_tools_file(tools_file_dict)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["server_label"] == "dummy"


def test_load_tools_file_list(tools_file_list):
    out = cli._load_tools_file(tools_file_list)
    assert isinstance(out, list)
    assert len(out) == 2
    labels = {d["server_label"] for d in out}
    assert labels == {"dummy", "dummy2"}


def test_load_tools_file_missing(tmp_path):
    with pytest.raises(SystemExit) as ex:
        cli._load_tools_file(str(tmp_path / "missing.json"))
    assert "Tools file not found" in str(ex.value)


def test_load_tools_file_empty(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        cli._load_tools_file(str(p))
    assert "Empty tools file" in str(ex.value)


def test_load_tools_file_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        cli._load_tools_file(str(p))
    assert "Invalid JSON in tools file" in str(ex.value)


def test_load_tools_file_wrong_type(tmp_path):
    p = tmp_path / "wrong.json"
    p.write_text(json.dumps("not an object or list"), encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        cli._load_tools_file(str(p))
    assert "must contain a JSON object or an array of objects" in str(ex.value)


# -------------
# _parse_llm_params
# -------------

def test_parse_llm_params_none():
    assert cli._parse_llm_params(None) == {}


def test_parse_llm_params_valid():
    out = cli._parse_llm_params('{"temperature":0.2,"max_output_tokens":800}')
    assert out == {"temperature": 0.2, "max_output_tokens": 800}


def test_parse_llm_params_invalid_json():
    with pytest.raises(SystemExit) as ex:
        cli._parse_llm_params("{not-json")
    assert "Invalid JSON in --llm-params" in str(ex.value)


def test_parse_llm_params_not_object():
    with pytest.raises(SystemExit) as ex:
        cli._parse_llm_params('["not", "object"]')
    assert "--llm-params must be a JSON object" in str(ex.value)


# -------------
# _import_by_dotted_path
# -------------

def test_import_by_dotted_path_colon():
    # Use standard lib function for reliability
    obj = cli._import_by_dotted_path("json:loads")
    assert obj is json.loads


def test_import_by_dotted_path_dot():
    obj = cli._import_by_dotted_path("json.loads")
    assert obj is json.loads


# -------------
# _build_client
# -------------

def test_build_client(monkeypatch, tools_file_dict):
    # Stub AzureOpenAIUnitOfWork and MCPClient to avoid network/import side effects
    created = {}

    class StubUoW:
        def __init__(self):  # no args to mirror real call
            created["uow"] = True

    class StubClient:
        def __init__(self, *, llm, tools, default_model):
            created["client"] = {"llm": llm, "tools": tools, "default_model": default_model}

    monkeypatch.setitem(sys.modules, "src.infrastructure.llm", SimpleNamespace(AzureOpenAIUnitOfWork=StubUoW))
    monkeypatch.setitem(sys.modules, "src.infrastructure.mcp", SimpleNamespace(MCPClient=StubClient))

    # Re-import function bound to our patched module
    from importlib import reload
    reload(cli)

    out_client = cli._build_client(tools_file_dict)
    assert created["uow"] is True
    assert isinstance(created["client"]["tools"], list)
    assert out_client is not None  # It's a StubClient instance


# -------------
# cmd_chat (piped / interactive)
# -------------

def test_cmd_chat_piped_stdin(monkeypatch, capsys, tools_file_dict):
    # Patch _build_client to avoid real networking
    class DummyClient:
        pass

    class DummyChat:
        def __init__(self, client, system_instructions=None, history=None):
            self.calls = []
        def send(self, **kwargs):
            # Ensure LLM params reach here (temperature example)
            assert kwargs.get("temperature") == 0.9
            return "ok-text"

    monkeypatch.setattr(cli, "_build_client", lambda path: DummyClient())
    monkeypatch.setattr(cli, "ChatService", lambda *a, **k: DummyChat(*a, **k))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "user prompt")
    monkeypatch.setattr(cli, "as_text", lambda r: r)  # identity

    args = argparse.Namespace(
        tool_config=tools_file_dict,
        system=cli.DEFAULT_SYSTEM,
        llm_params='{"temperature":0.9}',
    )
    rc = cli.cmd_chat(args)
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok-text" in captured.out


def test_cmd_chat_interactive_loop_one_turn(monkeypatch, capsys, tools_file_dict):
    class DummyClient:
        pass

    sends = []

    class DummyChat:
        def __init__(self, client, system_instructions=None, history=None):
            pass
        def send(self, **kwargs):
            sends.append(kwargs["prompt"])
            return "assistant-reply"

    # Build client and ChatService stubs
    monkeypatch.setattr(cli, "_build_client", lambda path: DummyClient())
    monkeypatch.setattr(cli, "ChatService", lambda *a, **k: DummyChat(*a, **k))
    # Make interactive: stdin is a TTY
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Simulate a single user input, then KeyboardInterrupt
    inputs = iter(["hello"])
    monkeypatch.setattr(cli, "input", lambda prompt="> ": next(inputs))
    def _raise_kb():
        raise KeyboardInterrupt()
    # After first loop iteration, raise KeyboardInterrupt
    monkeypatch.setattr(DummyChat, "send", lambda self, **kw: (_raise_kb() if sends else "assistant-reply"))

    # Patch as_text
    monkeypatch.setattr(cli, "as_text", lambda r: r)

    # Run
    with pytest.raises(SystemExit):
        # Because KeyboardInterrupt occurs inside loop, the function still returns 0,
        # but we simulate SystemExit if your harness triggers it. Alternatively,
        # simply call and assert printed output; here we just run cmd_chat normally:
        pass

    # More robust: run normally and let cmd_chat handle KeyboardInterrupt.
    # Rebind send to not raise to avoid exception
    monkeypatch.setattr(DummyChat, "send", lambda self, **kw: "assistant-reply")
    # Simulate input then KeyboardInterrupt during next input()
    seq = iter(["hello"])
    monkeypatch.setattr(cli, "input", lambda prompt="> ": next(seq) if sends == [] else (_ for _ in ()).throw(KeyboardInterrupt()))
    rc = cli.cmd_chat(argparse.Namespace(
        tool_config=tools_file_dict,
        system=cli.DEFAULT_SYSTEM,
        llm_params=None,
    ))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Interactive chat" in out
    assert "assistant-reply" in out


# -------------
# cmd_parse (stdin piped & BaseModel vs fallback)
# -------------

def test_cmd_parse_with_basemodel_output(monkeypatch, capsys, tools_file_dict):
    class DummyClient:
        pass

    class DummyBM(cli.BaseModel):  # inherit real pydantic BaseModel to exercise printing
        x: int

    class DummyChat:
        def __init__(self, client, system_instructions=None, history=None):
            pass
        def parse(self, **kwargs):
            # Make sure output_format dotted path resolved to class
            assert kwargs["text_format"] is DummyBM
            # Return BaseModel instance so CLI pretty-prints JSON
            return DummyBM(x=5)

    monkeypatch.setattr(cli, "_build_client", lambda path: DummyClient())
    monkeypatch.setattr(cli, "ChatService", lambda *a, **k: DummyChat(*a, **k))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "prompt-text")

    # Import resolver returns DummyBM
    monkeypatch.setattr(cli, "_import_by_dotted_path", lambda dotted: DummyBM)

    rc = cli.cmd_parse(argparse.Namespace(
        tool_config=tools_file_dict,
        system=cli.DEFAULT_SYSTEM,
        llm_params=None,
        output_format="pkg.Mod:DummyBM",
    ))
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["x"] == 5


def test_cmd_parse_with_fallback_object(monkeypatch, capsys, tools_file_dict):
    class DummyClient:
        pass

    class DummyParsed:
        # Mimic provider parsed response object with model_dump_json
        def __init__(self, data):
            self._data = data
        def model_dump_json(self, indent=2):
            return json.dumps(self._data, indent=indent)

    class DummyChat:
        def __init__(self, client, system_instructions=None, history=None):
            pass
        def parse(self, **kwargs):
            return DummyParsed({"ok": True})

    monkeypatch.setattr(cli, "_build_client", lambda path: DummyClient())
    monkeypatch.setattr(cli, "ChatService", lambda *a, **k: DummyChat(*a, **k))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "prompt-text")
    monkeypatch.setattr(cli, "_import_by_dotted_path", lambda dotted: DummyBaseModel)

    rc = cli.cmd_parse(argparse.Namespace(
        tool_config=tools_file_dict,
        system=cli.DEFAULT_SYSTEM,
        llm_params='{"temperature":0.1}',
        output_format="pkg:Dummy",
    ))
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["ok"] is True


# -------------
# build_parser wiring
# -------------

def test_build_parser_wiring_chat_defaults():
    parser = cli.build_parser()
    ns = parser.parse_args(["chat"])
    assert ns.tool_config == cli.DEFAULT_TOOLS_PATH
    assert ns.system == cli.DEFAULT_SYSTEM
    assert ns.llm_params is None
    assert ns.func is cli.cmd_chat


def test_build_parser_wiring_parse_requires_output_format():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["parse"])
    ns = parser.parse_args(["parse", "--output-format", "json:loads"])
    assert ns.tool_config == cli.DEFAULT_TOOLS_PATH
    assert ns.system == cli.DEFAULT_SYSTEM
    assert ns.llm_params is None
    assert ns.output_format == "json:loads"
    assert ns.func is cli.cmd_parse