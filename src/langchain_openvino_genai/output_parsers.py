from __future__ import annotations

from typing import Literal, NotRequired, TypedDict, TypeGuard
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser


class SchemaDict(TypedDict):
    mode: Literal["tool_call", "response"]
    tool_call: NotRequired[dict]
    response: NotRequired[str]


def is_schema_dict(obj: object) -> TypeGuard[dict]:
    return isinstance(obj, dict) and "mode" in obj


class ToolCallParser(BaseOutputParser):
    def parse(self, text: str) -> AIMessage:
        """Parse the model output (AIMessage or raw string) into an AIMessage object."""
        try:
            parsed = JsonOutputParser().parse(text)
            if is_schema_dict(parsed):
                if parsed["mode"] == "tool_call":
                    return self._parse_tool_call(parsed)
                if parsed["mode"] == "response":
                    return self._parse_response(parsed)
        except Exception as e:
            msg = f"Failed to parse output: {e}"
            raise ValueError(msg) from e
        else:
            msg = "Output does not conform to the expected schema."
            raise ValueError(msg)

    def _parse_tool_call(self, parsed: dict) -> AIMessage:
        if "tool_call" in parsed:
            tool_call = parsed["tool_call"]
            try:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": str(uuid4()),
                            "name": tool_call["tool_name"],
                            "args": tool_call["arguments"],
                            "type": "tool_call",
                        }
                    ],
                )
            except KeyError as e:
                msg = f"Tool call is missing required fields: {e}"
                raise ValueError(msg) from e
        msg = "Output does not contain a valid tool call."
        raise ValueError(msg)

    def _parse_response(self, parsed: dict) -> AIMessage:
        if "response" in parsed:
            return AIMessage(content=parsed["response"])
        msg = "Output does not contain a valid response."
        raise ValueError(msg)
