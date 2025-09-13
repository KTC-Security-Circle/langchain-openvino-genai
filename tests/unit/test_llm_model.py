import pytest

from langchain_openvino_genai.llm_model import OpenVINOLLM


def test_from_model_path_creates_components(dummy_model_path):
    try:
        llm = OpenVINOLLM.from_model_path(model_path=dummy_model_path, device="CPU")
    except RuntimeError as e:  # 実モデル未配置ケース
        pytest.skip(f"モデル未配置のため skip: {e}")
    assert llm.ov_pipe is not None, "ov_pipe が None (初期化失敗?)"
    assert llm.config is not None, "config が初期化されていません"
    assert llm.tokenizer is not None, "tokenizer が設定されていません"
    assert hasattr(llm, "streamer"), "streamer が設定されていません"


def test_llm_call_basic(ov_llm):
    out = ov_llm.invoke("Hello")
    # 期待: 文字列
    assert isinstance(out, str), f"文字列が期待されますが型={type(out)} 値={out!r}"
    # 値の妥当性(完全一致には拘らない)。空や極端に短いのは異常。
    assert len(out) >= 1, f"出力が短すぎます: {out!r}"


def test_llm_stop_strings_applied(ov_llm):
    text = ov_llm.invoke("Test", stop=["RESP"])  # モックは DUMMY_RESPONSE を返す
    # stop が働けば 'RESP' より前で切れるはず → 'DUMMY_' を含む短い文字列になる想定
    assert "RESP" not in text, f"stop 文字列が出力に含まれています: {text!r}"


def test_llm_stream_chunks(ov_llm):
    chunks = list(ov_llm.stream("Hello"))
    assert chunks, "ストリーム結果が空です"
    # それぞれ GenerationChunk を想定
    from langchain_core.outputs import GenerationChunk

    assert all(isinstance(c, GenerationChunk) for c in chunks), (
        f"異なる型が含まれています: {[type(c) for c in chunks]}"
    )


def test_structured_output_dict_schema(
    ov_llm, pet_schema, assert_has_structured_output
):
    ov_llm.set_structured_output_config(pet_schema)
    assert_has_structured_output(ov_llm, ["name", "age"])  # schema キー確認


def test_with_structured_output_returns_runnable(ov_llm, pet_schema):
    runnable = ov_llm.with_structured_output(pet_schema)
    from langchain_core.runnables import Runnable

    assert isinstance(runnable, Runnable), f"Runnable ではありません: {type(runnable)}"
