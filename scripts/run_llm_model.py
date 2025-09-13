from langchain_openvino_genai import OpenVINOLLM, load_model

model_name = "OpenVINO/Qwen3-8B-int4-cw-ov"
device = "CPU"

model_path = load_model(repo_id=model_name)

ov_llm = OpenVINOLLM.from_model_path(
    model_path=model_path,
    device=device,
)

print(ov_llm.bind(max_length=30).invoke("Hello, how are you?")) # noqa: T201
