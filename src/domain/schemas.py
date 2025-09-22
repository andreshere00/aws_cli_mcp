from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, HttpUrl

# ----------------------------- #
# ---- Chat & Tool Schemas ---- #
# ----------------------------- #


class ToolSelection(BaseModel):
    """Represents how to choose which tools are exposed to the model for a call."""

    value: Literal["all", "none", "some"]


class ChatTurn(BaseModel):
    """Represents a single message turn in the chat transcript."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


# -------------------- #
# ---- AWS Models ---- #
# -------------------- #

# ---- S3 ---- #


class Bucket(BaseModel):
    """Represents a single S3 bucket."""

    Name: str
    CreationDate: datetime
    BucketRegion: str
    BucketArn: str

    # ---- Helpers ---- #

    @staticmethod
    def normalize_region(location_constraint: Optional[str]) -> str:
        """Normalize S3 location constraint values to standard region codes."""
        if location_constraint in (None, "", "US"):
            return "us-east-1"
        if location_constraint == "EU":
            return "eu-west-1"
        return str(location_constraint)

    @staticmethod
    def arn_for(bucket_name: str) -> str:
        """Build an S3 bucket ARN."""
        return f"arn:aws:s3:::{bucket_name}"


class Owner(BaseModel):
    """Represents the owner of the S3 buckets."""

    DisplayName: str
    ID: str


class ListS3BucketResponse(BaseModel):
    """Response model for listing S3 buckets."""

    Buckets: List[Bucket]
    Owner: Owner
    ContinuationToken: Optional[str]
    Prefix: Optional[str]


# ------------------------- #
# ---- Tool definition ---- #
# ------------------------- #

# ---- MCP ---- #


class MCPToolConfig(BaseModel):
    """Configuration object for an MCP tool integration."""

    type: Literal["mcp"]
    server_label: str
    server_url: HttpUrl
    allowed_tools: List[str]
    require_approval: Literal["never", "always", "prompt"]


# ----------------------- #
# ---- LLM Providers ---- #
# ----------------------- #

# ---- Base Class ---- #


class BaseLLMResponse(BaseModel):
    """Provider-agnostic base for response payloads returned by LLMs."""

    def extract_text(self) -> str:
        """Public entry point to get assistant text, delegating to provider-specific logic."""
        text = self._extract_text()
        return text if isinstance(text, str) and text else str(self)

    # Subclasses override this with provider-specific logic.
    def _extract_text(self) -> Optional[str]:
        return None


class BaseLLMParsedResponse(BaseModel):
    """Provider-agnostic base for parsed/structured outputs."""

    def extract_text(self) -> str:
        """Public entry point to get assistant text, delegating to provider-specific logic."""
        text = self._extract_text()
        return text if isinstance(text, str) and text else str(self)

    # Subclasses override this with provider-specific logic.
    def _extract_text(self) -> Optional[str]:
        return None


# ---- AzureOpenAI ---- #

# -> Builders


class AzureContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    annotations: Optional[List[Any]] = None


class AzureOutputItem(BaseModel):
    """An item in Azure Responses `output` which can be a message OR a tool event.

    Azure may interleave message items with tool events (e.g., tool_call/tool_result).
    Only message items are guaranteed to have role/status/content.
    """

    # Common
    id: str
    type: str

    # Present for assistant "message" items
    status: Optional[str] = None
    role: Optional[str] = None
    content: Optional[List[AzureContentItem]] = None

    # Common fields seen in tool events (names vary by SDK/server)
    tool_name: Optional[str] = None
    call_id: Optional[str] = None
    error: Optional[Any] = None

    # Accept unknown extra keys without failing validation
    model_config = ConfigDict(extra="allow")


class AzureReasoning(BaseModel):
    effort: Optional[Any] = None
    summary: Optional[Any] = None


class AzureFormat(BaseModel):
    type: str


class AzureTextField(BaseModel):
    format: AzureFormat


class AzureInputTokensDetails(BaseModel):
    cached_tokens: int


class AzureOutputTokensDetails(BaseModel):
    reasoning_tokens: int


class AzureUsage(BaseModel):
    input_tokens: int
    input_tokens_details: AzureInputTokensDetails
    output_tokens: int
    output_tokens_details: AzureOutputTokensDetails
    total_tokens: int


# -> OpenAI Response API


class AzureOpenAIResponse(BaseLLMResponse):
    """Azure Responses API payload for /responses.create."""

    id: str
    object: str
    created_at: int
    status: str
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[Any] = None
    max_output_tokens: Optional[Any] = None
    model: str
    output: List[AzureOutputItem]
    parallel_tool_calls: Optional[bool] = None
    previous_response_id: Optional[Any] = None
    reasoning: AzureReasoning
    store: Optional[bool] = None
    temperature: float
    text: AzureTextField
    tool_choice: Any
    tools: List[Any]
    top_p: float
    truncation: str
    usage: AzureUsage
    user: Optional[Any] = None
    metadata: Dict[str, Any]

    def _extract_text(self) -> Optional[str]:
        """Return concatenated text from assistant message items in `output`."""
        if not isinstance(self.output, list) or not self.output:
            return None

        texts: list[str] = []
        for item in self.output:
            # We only care about assistant 'message' items
            if item.type != "message" or item.role != "assistant":
                continue
            if not item.content:
                continue

            # Collect all content items that have `text`
            for c in item.content:
                if isinstance(c.text, str) and c.text.strip():
                    texts.append(c.text)

        if texts:
            # Join multiple message/content fragments
            return "\n\n".join(texts)

        # Fallback: nothing matched strictly; try first message-like shape
        for item in self.output:
            if getattr(item, "content", None):
                c0 = item.content[0]
                if isinstance(getattr(c0, "text", None), str):
                    return c0.text
        return None


# -> OpenAI Parse API


class AzureOpenAIParsedResponse(AzureOpenAIResponse, BaseLLMParsedResponse):
    """Structured Outputs variant; includes provider-specific parsed fields when present."""

    output_parsed: Optional[Any] = None
    parsed: Optional[Any] = None

    def _extract_text(self) -> Optional[str]:
        if isinstance(self.output_parsed, str) and self.output_parsed:
            return self.output_parsed
        return super()._extract_text()
