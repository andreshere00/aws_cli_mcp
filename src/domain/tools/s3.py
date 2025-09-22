from typing import Any, Dict, Optional

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from src.domain.tools.base_tool import BaseMCPTool
from src.infrastructure.aws import S3UnitOfWork


class S3MCPTool(BaseMCPTool):
    """MCP toolset that exposes AWS S3 operations via FastMCP.

    This tool wraps an `S3UnitOfWork` to provide list operations as MCP tools
    that can be discovered and invoked by LLMs using the OpenAI Responses tools
    format.

    Attributes:
      uow: Unit of work responsible for interacting with AWS S3.
    """

    def __init__(self, uow: Optional[S3UnitOfWork] = None) -> None:
        """Initialize the S3 MCP tool.

        Args:
          uow: Optional `S3UnitOfWork` to use; if omitted, a default instance is
            created which reads its configuration (e.g., credentials/region) from
            the environment.
        """
        super().__init__(name="AWS S3")
        self.uow = uow or S3UnitOfWork()

    # ---- Tool Implementations (instance methods) -------------------------- #

    def _list_s3_buckets_impl(
        self,
        *,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
        max_buckets: int = 100,
    ) -> Dict[str, Any]:
        """List S3 buckets and return a JSON-serializable dictionary.

        Optionally filter by bucket name prefix and/or resolved region.

        Args:
          prefix: Only include buckets whose names start with this prefix.
          region: Only include buckets whose resolved region matches this value
            (for example, "us-east-1").
          max_buckets: Maximum number of buckets to include in the response.

        Returns:
          Dict[str, Any]: A structure compatible with `ListS3BucketResponse` when
          successful, or a dictionary with an `error` key describing the failure.

        Raises:
          This method catches `BotoCoreError`, `ClientError`, and `NoCredentialsError`
          and converts them to an `{"error": ...}` payload; unexpected exceptions are
          also caught and converted.

        Example:
          >>> tool = S3MCPTool()
          >>> tool._list_s3_buckets_impl(prefix="prod-", region="eu-west-1")
          {"Buckets": [...], "Owner": {...}, ...}
        """
        try:
            resp = self.uow.list_buckets(prefix=prefix, region=region, max_buckets=max_buckets)
            return self.to_dict(resp)
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            return {"error": f"S3 error: {str(e)}"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Unexpected error: {str(e)}"}

    # ---- FastMCP Registration -------------------------------------------- #

    def register_tools(self) -> None:
        """Register S3 tools on the FastMCP server instance.

        This binds the public MCP tool functions to the internal implementation
        methods of the class so they can be invoked by an LLM through the MCP
        transport.

        Returns:
          None: Tools are registered on `self.mcp` as a side effect.
        """

        @self.mcp.tool  # FastMCP decorator registers this function as a tool
        def list_s3_buckets(
            prefix: Optional[str] = None,
            region: Optional[str] = None,
            max_buckets: int = 100,
        ) -> dict:
            """List S3 buckets (optionally filtered) and return JSON.

            Args:
              prefix: Only include buckets whose names start with this prefix.
              region: Only include buckets whose resolved region matches this value
                (for example, "us-east-1").
              max_buckets: Maximum number of buckets to include in the response.

            Returns:
              dict: A JSON-serializable dictionary matching `ListS3BucketResponse` on
              success, or `{ "error": <message> }` on failure.
            """
            return self._list_s3_buckets_impl(prefix=prefix, region=region, max_buckets=max_buckets)


# ---- Launch as an MCP server ---------------------------------------------- #

if __name__ == "__main__":
    server = S3MCPTool()
    server.run(transport="streamable-http", host="0.0.0.0", port=8000, path="/mcp/s3")
