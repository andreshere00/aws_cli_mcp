"""
Minimal CLI for MCP/LLM chat and parse.

Examples
--------
# Chat (interactive; default tools.json and default system prompt)
python -m src.application.interfaces.cli chat

# Chat with custom tools file and system prompt; pass LLM params as JSON
python -m src.application.interfaces.cli chat \
  --tool-config tools.json \
  --system "Respond the user queries based on the provided tools" \
  --llm-params '{"temperature":0.2,"max_output_tokens":800}'

# Parse (Structured Outputs) with a Pydantic schema
echo "Return a JSON with 'buckets' field listing names" | \
python -m src.application.interfaces.cli parse \
  --tool-config tools.json \
  --system "Respond the user queries based on the provided tools" \
  --output-format src.domain.my_schemas:BucketList \
  --llm-params '{"temperature":0.0}'
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

from src.application.services.chat_service import ChatService
from src.domain.schemas import ToolSelection
from src.domain.utils import as_text
from src.infrastructure.mcp import MCPClient

DEFAULT_SYSTEM = "Respond the user queries based on the provided tools"
DEFAULT_TOOLS_PATH = "tools.json"


# -----------------------------
# Utilities
# -----------------------------


def _load_tools_file(path: str) -> List[Dict[str, Any]]:
    """Load MCP tool specs from a JSON file (list or single dict)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip() == "":
            raise SystemExit(f"Empty tools file: {path}")
        data = json.loads(content)
    except FileNotFoundError as e:
        raise SystemExit(f"Tools file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"Invalid JSON in tools file '{path}' at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]

    raise SystemExit(
        f"Tools file '{path}' must contain a JSON object or an "
        f"array of objects (got {type(data).__name__})."
    )


def _import_by_dotted_path(dotted: str) -> Any:
    """Import an object by 'pkg.module:ObjectName' or 'pkg.module.ObjectName'."""
    if ":" in dotted:
        module_name, obj_name = dotted.split(":", 1)
    else:
        module_name, obj_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, obj_name)


def _parse_llm_params(blob: Optional[str]) -> Dict[str, Any]:
    """Parse a JSON object of LLM kwargs."""
    if not blob:
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"Invalid JSON in --llm-params at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e
    if not isinstance(data, dict):
        raise SystemExit("--llm-params must be a JSON object, e.g. '{\"temperature\":0.2}'")
    return data


def _build_client(tools_path: str) -> MCPClient:
    """Wire an LLM UoW (provider-specific) and tool specs into an MCPClient (agnostic)."""
    # Lazily import to keep interface layer provider-agnostic
    from src.infrastructure.llm import AzureOpenAIUnitOfWork

    llm = AzureOpenAIUnitOfWork()  # reads env for endpoint/key/version
    tool_specs = _load_tools_file(tools_path)
    client = MCPClient(llm=llm, tools=tool_specs, default_model=None)
    return client


# -----------------------------
# Subcommand implementations
# -----------------------------


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat with optional LLM kwargs."""
    client = _build_client(args.tool_config)
    chat = ChatService(client, system_instructions=args.system)

    llm_kwargs = _parse_llm_params(args.llm_params)
    selection = ToolSelection(value="all")

    # If stdin is piped, do one-shot; otherwise interactive REPL.
    if not sys.stdin.isatty():
        prompt = sys.stdin.read()
        reply = chat.send(
            prompt=prompt,
            model=None,
            instructions=None,
            tool_selection=selection,
            tool_names=None,
            response_model=None,  # defaults internally to Azure schema
            **llm_kwargs,
        )
        print(as_text(reply))
        return 0

    print("Interactive chat. Ctrl+C to exit.")
    if args.system and args.system != DEFAULT_SYSTEM:
        print(f"[system set] {args.system}")

    try:
        while True:
            user = input("> ").strip()
            if not user:
                continue
            reply = chat.send(
                prompt=user,
                model=None,
                instructions=None,
                tool_selection=selection,
                tool_names=None,
                response_model=None,
                **llm_kwargs,
            )
            print(as_text(reply))
    except KeyboardInterrupt:
        print("\nBye!")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    """One-shot Structured Outputs (Responses.parse) with optional LLM kwargs."""
    client = _build_client(args.tool_config)
    chat = ChatService(client, system_instructions=args.system)

    text_format: Type[BaseModel] = _import_by_dotted_path(args.output_format)
    llm_kwargs = _parse_llm_params(args.llm_params)
    selection = ToolSelection(value="all")

    # One-shot: stdin or prompt line (fallback to interactive if desired later)
    if not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        prompt = input("> ").strip()

    result = chat.parse(
        text_format=text_format,
        prompt=prompt,
        model=None,
        instructions=None,
        tool_selection=selection,
        tool_names=None,
        parsed_response_model=None,  # defaults internally to Azure parsed schema
        **llm_kwargs,
    )

    if isinstance(result, BaseModel):
        print(result.model_dump_json(indent=2))
    else:
        # Fallback (provider parsed response)
        try:
            print(result.model_dump_json(indent=2))
        except Exception:
            print(json.dumps(result, default=str, indent=2))
    return 0


# -----------------------------
# CLI wiring
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-chat",
        description="Chat or parse with an LLM using MCP tools.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # chat subcommand
    p_chat = sub.add_parser("chat", help="Interactive chat (or one-shot via stdin).")
    p_chat.add_argument(
        "--tool-config",
        default=DEFAULT_TOOLS_PATH,
        help=f"Path to JSON file with MCP tool specs (default: {DEFAULT_TOOLS_PATH}).",
    )
    p_chat.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=f"System instructions to seed the session (default: '{DEFAULT_SYSTEM}').",
    )
    p_chat.add_argument(
        "--llm-params",
        help='JSON object of LLM kwargs, e.g. \'{"temperature":0.2,"max_output_tokens":800}\'.',
    )
    p_chat.set_defaults(func=cmd_chat)

    # parse subcommand
    p_parse = sub.add_parser("parse", help="Use Structured Outputs (Responses.parse).")
    p_parse.add_argument(
        "--tool-config",
        default=DEFAULT_TOOLS_PATH,
        help=f"Path to JSON file with MCP tool specs (default: {DEFAULT_TOOLS_PATH}).",
    )
    p_parse.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=f"System instructions to seed the session (default: '{DEFAULT_SYSTEM}').",
    )
    p_parse.add_argument(
        "--llm-params",
        help='JSON object of LLM kwargs, e.g. \'{"temperature":0.0,"max_output_tokens":400}\'.',
    )
    p_parse.add_argument(
        "--output-format",
        required=True,
        help="Dotted path to a Pydantic model for Responses.parse (e.g., 'pkg.module:MySchema').",
    )
    p_parse.set_defaults(func=cmd_parse)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
