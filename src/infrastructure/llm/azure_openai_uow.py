import os
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Type

from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types import Metadata
from pydantic import BaseModel

load_dotenv()


class AzureOpenAIUnitOfWork:
    """Unit of Work to wrap the Azure OpenAI Responses client with MCP tools support.

    This class centralizes Azure OpenAI interactions, providing convenience
    methods to:
      - Create single-turn responses with optional MCP tool invocation.
      - Parse responses into Pydantic models for structured outputs.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the underlying Azure OpenAI client.

        Args:
            **kwargs: Arbitrary keyword arguments passed to `AzureOpenAI`.
                These can include authentication parameters such as:
                  - `api_key`
                  - `azure_endpoint`
                  - `api_version`
                  - or any other valid client configuration.

        Example:
            ```python
            uow = AzureOpenAIUnitOfWork(
                api_key="my-secret",
                azure_endpoint="https://my-instance.openai.azure.com/",
                api_version="2024-01-01"
            )
            ```
        """
        params = {k: v for k, v in kwargs.items() if v is not None}
        self.client = AzureOpenAI(**params)

    def create_response(
        self,
        input: str,
        model: str,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[Metadata] = None,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        truncation: Literal["auto", "disabled"] = "auto",
        timeout: float = 60,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Literal["auto", "none"] | dict[str, Any] = "auto",
    ) -> Dict[str, Any]:
        """Create a single-turn response using Azure OpenAI.

        Args:
            input: User text prompt to process.
            model: Deployment name. Defaults to $AZURE_OPENAI_DEPLOYMENT.
            instructions: Optional system-level instructions.
            max_output_tokens: Maximum tokens in the output.
            metadata: Optional request metadata.
            temperature: Randomness control (0.0 = deterministic).
            top_p: Nucleus sampling cutoff.
            truncation: Whether to truncate long inputs. `"auto"` or `"disabled"`.
            timeout: Request timeout in seconds.
            tools: Optional list of MCP or custom tools.
            tool_choice: Strategy for tool invocation (`"auto"`, `"none"`, or explicit tool).

        Returns:
            The raw response object from the Azure OpenAI `responses.create` API.
        """
        if model is None:
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        response = self.client.responses.create(
            input=input,
            model=model,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            temperature=temperature,
            top_p=top_p,
            truncation=truncation,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
        )

        return response

    def parse_response(
        self,
        *,
        text_format: Type[BaseModel],
        messages: Optional[Sequence[Mapping[str, Any]]] = None,
        input: Optional[str] = None,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        truncation: Literal["auto", "disabled"] = "auto",
        timeout: float = 60,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Literal["auto", "none"] | dict[str, Any] = "auto",
        previous_response_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the Responses *parse* API to get a structured Pydantic-typed result.

        Args:
            text_format: A Pydantic model class describing the expected final output.
            messages: Full message list (dicts with 'role' & 'content'). Use this OR
                (instructions + input).
            input: User text content if not providing a full messages list.
            instructions: System instructions if not providing a full messages list.
            model: Azure deployment name. Defaults to $AZURE_OPENAI_DEPLOYMENT.
            max_output_tokens: Maximum tokens in the output.
            temperature: Randomness control (0.0 = deterministic).
            top_p: Nucleus sampling cutoff.
            truncation: Whether to truncate long inputs. `"auto"` or `"disabled"`.
            timeout: Request timeout in seconds.
            tools: Optional list of MCP or custom tools.
            tool_choice: Strategy for tool invocation (`"auto"`, `"none"`, or explicit tool).
            previous_response_id: Continue a prior Responses thread.

        Returns:
            An instance of `text_format` (Pydantic model) parsed from the model's final answer.

        Raises:
            ValueError: If neither `messages` nor (`instructions` and `input`) are provided.
            RuntimeError: If the API returns no parsed output (e.g., refusal or schema mismatch).

        Example:
            ```python
            class Answer(BaseModel):
                sentiment: str

            result = uow.parse_response(
                text_format=Answer,
                input="I love this product!",
                instructions="Extract the sentiment as 'positive', 'neutral', or 'negative'."
            )
            print(result.sentiment)  # -> "positive"
            ```
        """
        if model is None:
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # Build a minimal messages array if caller didn't supply one.
        if messages is None:
            if input is None:
                raise ValueError("Provide either `messages` or both `instructions` and `input`.")
            messages = []
            if instructions:
                messages.append({"role": "system", "content": instructions})
            messages.append({"role": "user", "content": input})

        # Call the structured-outputs helper; the SDK validates into `output_parsed`.
        response = self.client.responses.parse(
            model=model,
            input=list(messages),  # Responses API accepts the message list under "input"
            text_format=text_format,  # <- the schema (Pydantic BaseModel subclass)
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            truncation=truncation,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            previous_response_id=previous_response_id,
        )

        return response
