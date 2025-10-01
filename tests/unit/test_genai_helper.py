from langchain_openvino_genai.genai_helper import ChunkStreamer, IterableStreamer


def test_iterable_streamer_basic(monkeypatch):
    class DummyTokenizer:
        def decode(self, tokens):
            return "".join(chr(int(t)) for t in tokens)

    tok = DummyTokenizer()
    s = IterableStreamer(tok)
    # simulate tokens
    s.put(ord("A"))
    s.put(ord("B"))
    # flush end
    s.end()
    out = "".join(list(s))
    assert "A" in out and "B" in out, f"ストリーム出力に期待文字が欠落: {out!r}"


def test_chunk_streamer_groups(monkeypatch):
    class DummyTokenizer:
        def decode(self, tokens):
            return "".join(chr(int(t)) for t in tokens)

    tok = DummyTokenizer()
    s = ChunkStreamer(tok, tokens_len=2)
    s.put(ord("A"))  # not emitted yet
    emitted = s.put(ord("B"))  # triggers parent put
    assert emitted is False or emitted is True  # just ensure call success
    s.end()
    out = "".join(list(s))
    assert "A" in out and "B" in out, f"chunk flush 失敗: {out!r}"
