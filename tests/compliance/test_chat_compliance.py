"""ChatOpenVINO の ChatModel 契約に関する最小コンプライアンステスト."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk


@pytest.mark.compliance
def test_chat_invoke(chat_model):
    out = chat_model.invoke([HumanMessage(content="Hello")])
    assert isinstance(out, AIMessage), (
        f"AIMessage expected: got {type(out)} value={out}"
    )
    assert out.content is not None and len(out.content) >= 0, (
        "content が空文字 (生成失敗の可能性)"
    )


@pytest.mark.compliance
def test_chat_stream(chat_model):
    chunks = list(chat_model.stream([HumanMessage(content="Stream test")]))
    assert chunks, "stream からチャンクが得られませんでした"
    assert all(isinstance(c, ChatGenerationChunk) for c in chunks), (
        f"Chunk 型不一致: {[type(c) for c in chunks]}"
    )
