"""langchain-tests パッケージを用いた LLM インターフェース簡易コンプライアンステスト.

目的:
  - OpenVINOLLM が基本的 LangChain LLM 契約 (invoke/stream/stop) を崩していないかの smoke
  - 失敗しても理由調査に役立つログを assertion メッセージに含める

高度な完全網羅テストはここでは行わず、最低限の行動観察。
"""

from __future__ import annotations

import pytest
from langchain_core.outputs import GenerationChunk


@pytest.mark.compliance
def test_llm_basic_contract(ov_llm):
    # invoke
    out = ov_llm.invoke("hello")
    assert isinstance(out, str), (
        f"invoke の戻りが str でない: type={type(out)} value={out!r}"
    )
    assert len(out) >= 1, f"応答長が短すぎる: {out!r}"

    # stream
    chunks = list(ov_llm.stream("world"))
    assert chunks, "stream が空 (生成が行われていない)"
    assert all(isinstance(c, GenerationChunk) for c in chunks), (
        f"chunk 型不整合: {[type(c) for c in chunks]}"
    )


@pytest.mark.compliance
def test_llm_stop_behavior(ov_llm):
    # stop 文字列が適用されるか (モックは DUMMY_RESPONSE)
    text = ov_llm.invoke("x", stop=["RESP"])
    assert "RESP" not in text, f"stop が無視された可能性: output={text!r}"
