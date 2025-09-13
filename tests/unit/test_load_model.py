import importlib
from pathlib import Path

lm = importlib.import_module("langchain_openvino_genai.load_model")


def test_load_model_calls_snapshot(monkeypatch, tmp_path):
    called: dict = {}

    def fake_snapshot_download(**kwargs):  # capture kwargs
        called.update(kwargs)
        return str(tmp_path / "model")

    monkeypatch.setattr(lm, "snapshot_download", fake_snapshot_download)

    path = lm.load_model(repo_id="user/model", revision="main", download_path=tmp_path)
    assert Path(path).name == "model", f"戻り値パス不正: {path}"
    assert called.get("repo_id") == "user/model", f"repo_id 未反映: {called}"
    assert called.get("revision") == "main", f"revision 未反映: {called}"
