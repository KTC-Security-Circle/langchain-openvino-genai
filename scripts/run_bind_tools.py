from langchain_core.messages import (
    # AIMessage,
    HumanMessage,
    SystemMessage,
    # ToolMessage,
)
from langchain_core.tools import tool

from langchain_openvino_genai import ChatOpenVINO, OpenVINOLLM, load_model

model_name = "OpenVINO/Qwen3-8B-int4-cw-ov"
device = "GPU"

model_path = load_model(repo_id=model_name)

ov_llm = OpenVINOLLM.from_model_path(
    model_path=model_path,
    device=device,
)


@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


chat_model = ChatOpenVINO(llm=ov_llm, verbose=True)
messages = [
    SystemMessage(content=("You are a helpful assistant that can use tools.")),
    HumanMessage(content="What is 5 - 3?"),
    # AIMessage(
    #     content="",
    #     additional_kwargs={},
    #     response_metadata={},
    #     tool_calls=[
    #         {
    #             "name": "subtract",
    #             "args": {"a": 5, "b": 3},
    #             "id": "771b44ab-e212-4fcf-9608-7303780f1774",
    #             "type": "tool_call",
    #         }
    #     ],
    # ),
    # ToolMessage(
    #     content="2",
    #     name="subtract",
    #     tool_call_id="771b44ab-e212-4fcf-9608-7303780f1774",
    # ),
]

tool_llm = chat_model.bind_tools([add, subtract])
print(tool_llm.invoke(messages))  # noqa: T201
