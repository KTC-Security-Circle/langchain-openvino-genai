from langchain_openvino_genai import ChatOpenVINO, OpenVINOLLM, load_model

model_name = "OpenVINO/Qwen3-8B-int4-cw-ov"
device = "CPU"

model_path = load_model(repo_id=model_name)

ov_llm = OpenVINOLLM.from_model_path(
    model_path=model_path,
    device=device,
)

chat_model = ChatOpenVINO(llm=ov_llm, verbose=True)
messages = [
    (
        "system",
        "You are a helpful translator. Translate the user sentence to Japanese.",
    ),
    ("human", "I love programming."),
]

# print(chat_model.invoke(messages)) # noqa: T201
chat_model.invoke(messages).pretty_print()
