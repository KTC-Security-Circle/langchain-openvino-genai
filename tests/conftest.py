"""Global pytest fixtures for the project.

方針:
- openvino_genai が無くてもテストが走るように軽量モックを提供
- 実ライブラリが存在する場合は smoke 的に本物を利用可能
- 失敗した場合でも *何が期待値で何が返ったか* を assertion メッセージに詳細出力
"""

from __future__ import annotations

# 環境変数 FORCE_FAKE_OPENVINO=1 で常にモック使用可能
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pytest

FORCE_FAKE = os.environ.get("FORCE_FAKE_OPENVINO") == "1"

# ----- Conditional import / mock openvino_genai -----
try:  # pragma: no cover - import branch distinction
    if FORCE_FAKE:
        raise ImportError("Forced fake by env")
    import openvino_genai  # type: ignore
except Exception:  # noqa: BLE001

    class _FakeGenerationConfig:
        def __init__(self):
            self.stop_strings: set[str] = set()
            self.structured_output_config = None

    class _FakeTokenizer:
        def __init__(self):
            self.added_special_tokens = []

        def decode(self, tokens):
            # Very naive: join ints as chars modulo ASCII safe range
            chars = []
            for t in tokens:
                try:
                    if isinstance(t, (list, tuple)):
                        t = t[0]
                    chars.append(chr(int(t) % 90 + 32))
                except Exception:
                    chars.append("?")
            return "".join(chars)

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=False
        ):
            # Simplified: role:content\n...
            rendered = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            if add_generation_prompt:
                rendered += "\nassistant:"
            return rendered

        def batch_decode(self, token_ids, skip_special_tokens=True):
            return [self.decode(token_ids[0] if token_ids else [])]

    class _FakeTokenizedInputs:
        def __init__(self, tokens, attention_mask=None):
            self.tokens = tokens
            self.attention_mask = attention_mask

    class _FakeStreamerBase:
        def reset(self):
            pass

    class _FakeLLMPipeline:
        def __init__(self, model_path: str, device: str, **_: Any):  # store minimal
            self.model_path = model_path
            self.device = device
            self._config = _FakeGenerationConfig()
            self._tokenizer = _FakeTokenizer()

        def get_generation_config(self):
            return self._config

        def get_tokenizer(self):
            return self._tokenizer

        def generate(self, prompt, config, streamer=None, **kwargs):  # noqa: D401
            # Produce deterministic dummy output to make assertions stable.
            base_text = "DUMMY_RESPONSE"
            # Respect stop strings if appear in base_text for demonstration
            for stop in getattr(config, "stop_strings", []) or []:
                if stop in base_text:
                    base_text = base_text.split(stop)[0]
            if streamer is not None:
                for ch in base_text:
                    streamer.put(ord(ch))
                streamer.end()
                return types.SimpleNamespace(tokens=[[ord(c) for c in base_text]])
            return types.SimpleNamespace(
                tokens=[[ord(c) for c in base_text]], text=base_text
            )

    class _FakeStructuredOutputConfig:
        def __init__(self, json_schema: str):
            self.json_schema = json_schema

    # Expose fake module structure
    openvino_genai = types.SimpleNamespace(
        LLMPipeline=_FakeLLMPipeline,
        Tokenizer=_FakeTokenizer,
        TokenizedInputs=_FakeTokenizedInputs,
        StreamerBase=_FakeStreamerBase,
        GenerationConfig=_FakeGenerationConfig,
        StructuredOutputConfig=_FakeStructuredOutputConfig,
    )

# ----- Fixtures -----


@pytest.fixture(scope="session")
def has_real_openvino() -> bool:
    return (
        not isinstance(openvino_genai.LLMPipeline, type)
        or openvino_genai.LLMPipeline.__name__ != "_FakeLLMPipeline"
    )


def pytest_addoption(parser):  # pragma: no cover
    parser.addoption(
        "--ov-model-path",
        action="store",
        nargs="?",  # 値省略可
        const="__AUTO__",  # 指定のみで値なしの場合は同梱デフォルトを使用
        default=None,
        help=(
            "Path to an OpenVINO GenAI exported model directory. "
            "--ov-model-path <dir> で明示指定, --ov-model-path (値なし) で同梱デフォルトを自動探索。"
            " 自動探索順: llms/Qwen3-8B-int4-cw-ov → llms/ (直下に openvino_model.xml がある場合)"
            " 未指定の場合はモデル依存テストを skip。"
        ),
    )
    parser.addoption(
        "--ov-devices",
        action="store",
        default=None,
        help=(
            "Comma separated list of OpenVINO target devices to test (e.g. CPU,GPU,NPU). "
            "環境変数 OPENVINO_TEST_DEVICES でも指定可能。未指定時は CPU のみ。"
        ),
    )


@pytest.fixture(scope="session")
def dummy_model_path(pytestconfig):
    """Resolve the model path to use for real OpenVINO tests.

    解決優先順:
      1. コマンドライン --ov-model-path
      2. 環境変数 OPENVINO_MODEL_PATH
      3. リポジトリ同梱デフォルト: llms/

    いずれも存在しない場合は skip (実モデル不要テストのみ実行)。
    """
    cli: Optional[str] = pytestconfig.getoption("--ov-model-path")
    env_path = os.environ.get("OPENVINO_MODEL_PATH")

    repo_root = Path(__file__).parent.parent
    bundled_qwen = repo_root / "llms" / "Qwen3-8B-int4-cw-ov"
    bundled_flat = repo_root / "llms"

    # --ov-model-path を値なしで指定 → const の __AUTO__ が入る
    if cli == "__AUTO__":
        # 優先: サブディレクトリ(Qwen*) → フラット(llms/) 直下に openvino_model.xml があるか
        if bundled_qwen.exists():
            return str(bundled_qwen.resolve())
        if (bundled_flat / "openvino_model.xml").exists():
            return str(bundled_flat.resolve())
        pytest.skip(
            "--ov-model-path (値なし) 指定だが既定候補 (llms/Qwen3-8B-int4-cw-ov あるいは llms/ に openvino_model.xml) が見つかりません"
        )

    # 明示指定があれば最優先
    if cli:
        p = Path(cli)
        if p.exists():
            return str(p.resolve())
        raise pytest.UsageError(
            f"--ov-model-path で指定されたパスが存在しません: {cli}"
        )

    # 環境変数
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p.resolve())
        raise pytest.UsageError(
            f"OPENVINO_MODEL_PATH が存在しないパスを指しています: {env_path}"
        )

    pytest.skip(
        "OpenVINO モデル未指定のためモデル依存テストを skip (明示実行したい場合は --ov-model-path 或いは OPENVINO_MODEL_PATH を設定)"
    )


@pytest.fixture(scope="session")
def ov_test_devices(pytestconfig):
    raw = pytestconfig.getoption("--ov-devices") or os.environ.get(
        "OPENVINO_TEST_DEVICES", "CPU"
    )
    devices = [d.strip() for d in raw.split(",") if d.strip()]
    # 正規化 (大文字)
    devices = [d.upper() for d in devices]
    # 重複除去
    seen = set()
    uniq = []
    for d in devices:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


@pytest.fixture()
def ov_llm(dummy_model_path):
    """Instantiate real (or fake forced) OpenVINOLLM using resolved model path.

    モデルパスが存在しないケースは dummy_model_path フィクスチャで skip 済み。
    パイプライン初期化失敗時はテスト失敗として扱い、原因 (例外メッセージ) をそのまま伝播。
    """
    from langchain_openvino_genai.llm_model import OpenVINOLLM

    return OpenVINOLLM.from_model_path(model_path=dummy_model_path, device="CPU")


@pytest.fixture()
def chat_model(ov_llm):
    from langchain_openvino_genai.chat_model import ChatOpenVINO

    return ChatOpenVINO(llm=ov_llm)


# Structured output schema example
@dataclass
class _Pet:
    name: str
    age: int


@pytest.fixture()
def pet_schema():
    return {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }


# Tool stub for bind_tools
class _EchoTool:
    name = "echo"
    description = "Echo back the provided text"
    # args_schema needs model_json_schema & model_fields like pydantic model
    from pydantic import BaseModel

    class ArgsSchema(BaseModel):  # type: ignore
        text: str

    args_schema = ArgsSchema


@pytest.fixture()
def echo_tool():
    return _EchoTool()


@pytest.fixture()
def two_tools(echo_tool):
    class _UpperTool:
        name = "upper"
        description = "Convert text to upper-case"
        from pydantic import BaseModel

        class ArgsSchema(BaseModel):  # type: ignore
            text: str

        args_schema = ArgsSchema

    return [echo_tool, _UpperTool()]


# Helper to assert JSON schema presence
@pytest.fixture()
def assert_has_structured_output():
    def _assert(model, expect_keys: List[str]):
        cfg = getattr(model, "config", None)
        so = getattr(cfg, "structured_output_config", None)
        assert so is not None, (
            "structured_output_config が設定されていません (schema未設定)"
        )
        if hasattr(so, "json_schema"):
            js = so.json_schema
        else:
            js = getattr(so, "json_schema", None)
        assert js, "json_schema が空です"
        for k in expect_keys:
            assert k in js, f"期待キー {k} が schema に含まれない: json_schema={js}"

    return _assert


# Marker registration (for IDE auto-complete clarity)
def pytest_configure(config):  # pragma: no cover
    config.addinivalue_line("markers", "compliance: langchain-tests 仕様テスト")
    config.addinivalue_line("markers", "realmodel: 実 OpenVINO 利用")
    config.addinivalue_line("markers", "slow: 時間のかかるテスト")
