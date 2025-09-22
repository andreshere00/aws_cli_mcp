from __future__ import annotations

import json
from typing import Any, Dict

from fastmcp import FastMCP
from pydantic import BaseModel


class BaseMCPTool:
    """Base class for MCP tool collections with dict conversion and server helpers.

    This class provides a foundation for creating collections of tools that can be
    registered with a FastMCP application. It includes utility methods for converting
    various data types to dictionaries and manages the lifecycle of the MCP server.

    Attributes:
        mcp (FastMCP): The FastMCP application instance initialized with the given name.
    """

    def __init__(self, name: str) -> None:
        """Initialize the MCP app with a display name.

        Args:
            name (str): The display name for the MCP application.
        """
        self.mcp = FastMCP(name)

    # ---- Utilities --------------------------------------------------------- #

    @staticmethod
    def to_dict(value: Any) -> Dict[str, Any]:
        """Convert Pydantic models, JSON strings, or mappings to a plain dict.

        Args:
            value (Any): The value to convert, which can be a Pydantic model, JSON string,
                dictionary, or any object with a `model_dump` method.

        Returns:
            Dict[str, Any]: The converted dictionary representation of the input value.

        Raises:
            TypeError: If the input value is of an unsupported type.
        """
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")  # type: ignore[attr-defined]
        raise TypeError(f"Tool returned unsupported type: {type(value)}")

    # ---- Registration & Running ------------------------------------------- #

    def register_tools(self) -> None:
        """Register tool functions with the underlying FastMCP app.

        This method should be implemented by subclasses to register their specific tools.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement register_tools().")

    def run(
        self,
        *,
        transport: str = "streamable-http",
        host: str = "0.0.0.0",
        port: int = 8000,
        path: str = "/mcp",
    ) -> None:
        """Run the MCP server with the provided transport and network options.

        This method registers the tools and starts the FastMCP server with the specified
        transport protocol and network configuration.

        Args:
            transport (str, optional): The transport protocol to use. Defaults to "streamable-http".
            host (str, optional): The network interface to bind to. Defaults to "0.0.0.0".
            port (int, optional): The port number to listen on. Defaults to 8000.
            path (str, optional): The URL path for the MCP endpoint. Defaults to "/mcp".

        Side Effects:
            Starts the MCP server which blocks the calling thread until the server is stopped.
        """
        self.register_tools()
        self.mcp.run(transport=transport, host=host, port=port, path=path)
