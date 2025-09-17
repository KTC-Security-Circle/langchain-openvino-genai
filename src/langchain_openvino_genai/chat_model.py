from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from pydantic import BaseModel

from langchain_openvino_genai.llm_model import OpenVINOLLM  # noqa: TC001 # 実行時にバリデーションで使用されるため
from langchain_openvino_genai.output_parsers import ToolCallParser

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""

TOOL_LLM_SYSTEM_PROMPT = """You can use the following tools to help the user.

Available tools:
{tools_description}

When you know the final answer, respond with mode "response" and provide the answer in the response field."""


def is_args_schema_model(val: object) -> TypeGuard[type[BaseModel]]:
    return isinstance(val, type) and issubclass(val, BaseModel)


class ChatOpenVINO(BaseChatModel):
    """OpenVINO LLM's as ChatModels.

    Works with `OpenVINOLLM` LLMs.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        .. code-block:: python

            from langchain_community.llms import OpenVINOLLM

            llm = OpenVINOPipeline.from_model_path(
                model_path="./openvino_model_dir",
                device="CPU",
            )

            chat = ChatOpenVINO(llm=llm, verbose=True)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user
                sentence to French."),
                ("human", "I love programming."),
            ]

            chat(...).invoke(messages)

        .. code-block:: python


    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python


    """

    llm: OpenVINOLLM
    """LLM, must be of type OpenVINOLLM"""
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None

    _additional_system_message: SystemMessage | None = None

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)

        if self.tokenizer is None:
            self.tokenizer = self.llm.tokenizer

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs)  # noqa: SLF001
        return self._to_chat_result(llm_result)

    def _stream(
        self,
        messages: list[BaseMessage],
        _stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._to_chat_prompt(messages)

        for data in self.llm.stream(request, None, stop=None, **kwargs):
            delta = data
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk

    def _to_chat_prompt(
        self,
        messages: list[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        try:
            import openvino_genai

        except ImportError:
            msg = "Could not import OpenVINO GenAI package. Please install it with `pip install openvino-genai`."
            raise ImportError(msg) from None
        if not messages:
            msg = "At least one HumanMessage must be provided!"
            raise ValueError(msg)

        # if not isinstance(messages[-1], HumanMessage):
        #     msg = "Last message must be a HumanMessage!"
        #     raise TypeError(msg)

        if self._additional_system_message is not None:
            messages = [self._additional_system_message, *messages]

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return (
            self.tokenizer.apply_chat_template(messages_dicts, add_generation_prompt=True)
            if isinstance(self.tokenizer, openvino_genai.Tokenizer)
            else self.tokenizer.apply_chat_template(messages_dicts, tokenize=False, add_generation_prompt=True)
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, ToolMessage):
            role = f"Tool {message.name} result:"
        else:
            msg = f"Unknown message type: {type(message)}"
            raise TypeError(msg)

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(message=AIMessage(content=g.text), generation_info=g.generation_info)
            chat_generations.append(chat_generation)

        return ChatResult(generations=chat_generations, llm_output=llm_result.llm_output)

    @property
    def _llm_type(self) -> str:
        return "openvino-chat-wrapper"

    def with_structured_output(
        self, schema: dict | type, **_kwargs: object
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Return a version of this LLM that produces structured output.

        Args:
            schema: A pydantic BaseModel class or a dict representing the schema.
            **kwargs: Additional keyword arguments to pass to the structured output parser.

        Returns:
            A Runnable that produces structured output conforming to the provided schema.
        """
        self.llm.set_structured_output_config(schema)

        output_parser: JsonOutputParser = JsonOutputParser()

        return self | output_parser

    def _split_tool_args_schema(self, tool: BaseTool) -> dict:
        """Split tool properties into required and optional."""
        if not is_args_schema_model(tool.args_schema):
            msg = f"Tool {tool.name} args_schema must be a pydantic BaseModel class."
            raise ValueError(msg)
        args_schema = tool.args_schema.model_json_schema()
        properties = {
            arg_name: {"type": arg_type.pop("type")} for arg_name, arg_type in args_schema.get("properties", {}).items()
        }
        required = args_schema.get("required", [])
        return {"properties": properties, "required": required}

    def _tool_signature(self, tool: BaseTool) -> str:
        """Return a signature string like tool_name(arg: Type, ...)."""
        fields = tool.args_schema.model_fields if is_args_schema_model(tool.args_schema) else {}
        parts: list[str] = []
        for name, field in fields.items():
            try:
                ann = field.annotation
            except AttributeError:
                ann = None
            if ann is None:
                type_name = "Any"
            else:
                try:
                    type_name = ann.__name__
                except AttributeError:
                    type_name = str(ann)
                type_name = type_name.replace("typing.", "")
            parts.append(f"{name}: {type_name}")
        args_part = ", ".join(parts)
        return f"{tool.name}({args_part})" if args_part else f"{tool.name}()"

    def generate_tools_system_prompt(self, tools: Sequence[BaseTool]) -> SystemMessage:
        """Generate a system prompt enumerating tools.

        Format per line:
            - tool_name(arg: Type, ...): description

        Returns a `SystemMessage` ready to prepend to conversation.
        """
        lines: list[str] = []
        for t in tools:
            sig = self._tool_signature(t)
            desc = (t.description or "").strip().replace("\n", " ")
            lines.append(f"- {sig}: {desc}")
        tools_block = "\n".join(lines) if lines else "(no tools available)"
        content = TOOL_LLM_SYSTEM_PROMPT.format(tools_description=tools_block)
        return SystemMessage(content=content)

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        _tool_choice: str | None = None,
        **_kwargs: object,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model.

        Args:
            tools: A sequence of tools to bind to the model.
            tool_choice: Optional tool choice strategy.
            **kwargs: Additional keyword arguments.

        Returns:
            A Runnable that uses the chat model with the bound tools.
        """
        schema = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["tool_call", "response"]
                },
                "tool_call": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "enum": [tool.name for tool in tools],
                        },
                        "arguments": {
                            "type": "object",
                            "oneOf": [self._split_tool_args_schema(tool) for tool in tools],
                        },
                    },
                    "description": "If the mode is 'tool_call', this field must be provided.",
                    "required": ["tool_name", "arguments"],
                    "additionalProperties": False,
                },
                "response": {
                    "type": "string",
                    "description": "If the mode is 'response', this field must be provided.",
                },
            },
            "required": ["mode"],
            "additionalProperties": False
        }
        self.llm.set_structured_output_config(schema)
        self._additional_system_message = self.generate_tools_system_prompt(tools)

        output_parser: ToolCallParser = ToolCallParser()

        return self | output_parser
