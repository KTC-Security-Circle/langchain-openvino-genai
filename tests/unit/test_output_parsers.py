import pytest
from langchain_core.messages import AIMessage

from langchain_openvino_genai.output_parsers import ToolCallParser


def test_tool_call_parser_success():
    text = '{"tool_name": "echo", "arguments": {"text": "hello"}}'
    msg = ToolCallParser().parse(text)
    assert isinstance(msg, AIMessage), f"AIMessage expected got {type(msg)}"
    assert msg.tool_calls, f"tool_calls が空: {msg}"
    tc = msg.tool_calls[0]
    assert tc["name"] == "echo", f"name mismatch: {tc}"
    assert tc["args"]["text"] == "hello", f"args mismatch: {tc}"


def test_tool_call_parser_error():
    bad_text = '{"x": 1}'
    with pytest.raises(ValueError) as e:
        ToolCallParser().parse(bad_text)
    assert "Failed to parse" in str(e.value) or "required keys" in str(e.value)
