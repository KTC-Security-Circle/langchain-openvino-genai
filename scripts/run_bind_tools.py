from langchain_core.tools import tool

from langchain_openvino_genai import ChatOpenVINO, OpenVINOLLM, load_model

model_name = "OpenVINO/Qwen3-8B-int4-cw-ov"
device = "CPU"

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
    (
        "system",
        "You are a helpful assistant that can use tools. Use the add tool to add two numbers.",
    ),
    ("human", "What is 5 - 3?"),
]

tool_llm = chat_model.bind_tools([add, subtract])
print(tool_llm.invoke(messages))  # noqa: T201
