from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser


class ToolCallParser(BaseOutputParser):
    def parse(self, text: str) -> AIMessage:
        """Parse the model output (AIMessage or raw string) into an AIMessage object."""
        try:
            parsed = JsonOutputParser().parse(text)
            if isinstance(parsed, dict) and "tool_name" in parsed and "arguments" in parsed:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": str(uuid4()),
                            "name": parsed["tool_name"],
                            "args": parsed["arguments"],
                            "type": "tool_call",
                        }
                    ],
                )
            raise ValueError("Parsed output does not contain required keys 'tool' and 'args'.")
        except Exception as e:
            raise ValueError(f"Failed to parse output: {e}")
