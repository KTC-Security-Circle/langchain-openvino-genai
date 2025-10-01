# tests/ README

本ディレクトリは `langchain_openvino_genai` パッケージ向けの pytest ベース自動テスト群です。

---

## クイックリファレンス (最短手順)

PowerShell 前提。

```powershell
# 依存関係同期 (プロジェクトルートで)
uv sync

# モデルをまだダウンロードしていない場合 (例: 既存 OpenVINO 変換済みモデルを ./llms 下へ配置)
# 例) llms/Qwen3-8B-int4-cw-ov/ openvino_model.xml 等が存在する想定

# モデル無しで *モック利用可能* なテストのみ実行 (モデル依存テストは skip)
uv run pytest -q

# 同梱/配置済みモデルを自動探索して実モデルテストも含めて実行
uv run pytest --ov-model-path -q

# 明示的にモデルパスを指定
uv run pytest --ov-model-path .\llms\Qwen3-8B-int4-cw-ov -q

# 特定カテゴリのみ (例: 単体テスト)
uv run pytest tests/unit -q

# デバイス (CPU,GPU,NPU) を指定して簡易動作確認
uv run pytest tests/features/test_device_support.py::test_llm_runs_on_devices --ov-model-path --ov-devices CPU,GPU,NPU -q -s
```

---

## テスト実行オプション

| オプション / 環境変数      | 目的                                     | 挙動                                                                      |
| -------------------------- | ---------------------------------------- | ------------------------------------------------------------------------- |
| `--ov-model-path <dir>`    | 実 OpenVINO GenAI モデルディレクトリ指定 | 例: `--ov-model-path llms/Qwen3-8B-int4-cw-ov`                            |
| `--ov-model-path` (値なし) | 自動探索                                 | `llms/Qwen3-8B-int4-cw-ov` → `llms/` 直下に `openvino_model.xml` 順で探索 |
| 未指定                     | モデル依存テスト skip                    | モデル不要テストのみ実行                                                  |
| `OPENVINO_MODEL_PATH`      | モデルパス環境指定                       | CLI 未指定時に利用                                                        |
| `--ov-devices CPU,GPU,NPU` | 複数デバイス動作検証                     | 各デバイスで LLM 初期化を試行                                             |
| `OPENVINO_TEST_DEVICES`    | デバイス一覧環境指定                     | CLI 未指定時に利用 (デフォルト CPU)                                       |
| `FORCE_FAKE_OPENVINO=1`    | 強制モック                               | 実ライブラリがあってもモックで実行                                        |

---

## マーカー

| マーカー     | 意味                                   | 備考                                   |
| ------------ | -------------------------------------- | -------------------------------------- |
| `compliance` | langchain インターフェイス準拠スモーク | langchain-tests 互換観点の軽量チェック |
| `realmodel`  | 実モデル前提テスト (将来拡張位置)      | 現状付与箇所は最小                     |
| `slow`       | 実行に時間がかかる                     | デバイス複数検証などに付与検討         |

フィルタ例: `pytest -m compliance -q` / `pytest -m 'compliance and not slow'`。

---

## テストカテゴリ

1. 単体 (`tests/unit/`)
   - ラッパークラス (`OpenVINOLLM`, `ChatOpenVINO`) の初期化・invoke・stream
   - ストラクチャ出力設定・ツールバインド補助 (`output_parsers`, `genai_helper`) の純粋ロジック
   - HF ダウンロードラッパーのパッチテスト (`test_load_model.py`)
2. 機能 (`tests/features/`)
   - Structured Output: JSON Schema 設定が `structured_output_config` に反映されるか
   - Tool Binding: `bind_tools` で組み立てた system prompt / tool schema の体裁
   - Device Support: CPU/GPU/NPU で初期化 → 短推論 (max_new_tokens/length≈30) の可否
3. コンプライアンス (`tests/compliance/`)
   - LangChain 標準的パターン( invoke / stream 擬似 )を破壊していないかのスモーク

---

## 主要フィクスチャ (`conftest.py`)

| フィクスチャ                   | 目的                                                            |
| ------------------------------ | --------------------------------------------------------------- |
| `dummy_model_path`             | モデルパス解決 (CLI/ENV/自動探索)。見つからなければ skip を発火 |
| `ov_llm`                       | `OpenVINOLLM` インスタンス (CPU)                                |
| `chat_model`                   | `ChatOpenVINO` ラッパー                                         |
| `pet_schema`                   | Structured Output 用サンプル JSON Schema                        |
| `echo_tool` / `two_tools`      | Tool binding 用ダミーツール                                     |
| `assert_has_structured_output` | schema 埋め込み検証ヘルパ                                       |
| `ov_test_devices`              | `--ov-devices` / ENV からデバイス一覧生成                       |
| `has_real_openvino`            | 実ライブラリ検出フラグ                                          |

### モック挙動

`openvino_genai` が無い/`FORCE_FAKE_OPENVINO=1` の場合、簡易 LLMPipeline モックを挿入。`generate()` は安定文字列 `DUMMY_RESPONSE` を返し、stop 文字列を簡易適用。これにより不安定な推論結果へ依存せずテストが複製可能。

---

## 個別テストファイル概要

| ファイル                             | 概要                                                        |
| ------------------------------------ | ----------------------------------------------------------- |
| `unit/test_llm_model.py`             | LLM ラッパーの基本・停止語処理・ストリーム動作              |
| `unit/test_chat_model.py`            | チャットプロンプト整形・ツールバインド経路                  |
| `unit/test_genai_helper.py`          | `IterableStreamer` / `ChunkStreamer` のバッファ & 終端制御  |
| `unit/test_output_parsers.py`        | ToolCallParser の JSON→ 構造化 変換成功/失敗パス            |
| `unit/test_load_model.py`            | `snapshot_download` パッチによるダウンロードラッパ挙動      |
| `features/test_structured_output.py` | JSON Schema 付与確認と最小出力検証                          |
| `features/test_bind_tools.py`        | Tool binding による system prompt/内部構造の一貫性          |
| `features/test_device_support.py`    | デバイス (CPU,GPU,NPU) 初期化 + 短推論。失敗は xfail 可視化 |
| `compliance/test_llm_compliance.py`  | LLM 基本 API スモーク                                       |
| `compliance/test_chat_compliance.py` | Chat モデル API スモーク                                    |

---

## デバイス検証テストの詳細

`test_device_support.py` は `ov_test_devices` の各デバイスで:

1. `OpenVINOLLM.from_model_path(..., device=<DEV>)` 初期化
2. GenerationConfig の `max_new_tokens` / `max_length` / `max_tokens` のいずれかを 30 に制限
3. `invoke("Hello")` → 非空文字列を期待
4. `invoke("Hello", stop=["RESP"])` → 生成結果に `RESP` が含まれないこと

初期化例外 (デバイス非対応等) は `xfail` でレポートに残して静かに許容 (CI で対応状況を可視化)。

---

## よくあるシナリオ

| 目的                           | コマンド例                                                  |
| ------------------------------ | ----------------------------------------------------------- |
| モデル無しで高速にロジック回帰 | `uv run pytest -q`                                          |
| 実モデル含め総合               | `uv run pytest --ov-model-path -q`                          |
| 特定ファイルのみ               | `uv run pytest tests/features/test_bind_tools.py -q`        |
| compliance のみ                | `uv run pytest -m compliance -q`                            |
| GPU/NPU を含むデバイス検証     | `uv run pytest --ov-model-path --ov-devices CPU,GPU,NPU -q` |

---

## トラブルシューティング

| 症状                             | 原因/対処                                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| ほとんど skip される             | モデルパス未指定。`--ov-model-path` か `OPENVINO_MODEL_PATH` を設定                                          |
| `UsageError --ov-model-path`     | 指定パスが存在しない。ディレクトリを再確認                                                                   |
| GPU/NPU で xfail                 | ドライバ / OpenVINO ランタイム未整備。サポート後 pass へ変化する想定                                         |
| Structured output が失敗         | schema に required フィールド不足・`with_structured_output` 呼び出し確認                                     |
| Tool binding 関連 assertion 失敗 | `bind_tools` の引数ツールの `args_schema` (pydantic) 定義漏れ                                                |
| stop が効かない                  | モック利用時は `DUMMY_RESPONSE` に stop 文字が含まれるか要確認。実モデルでは tokenizer/stop_strings 伝播確認 |

---

## 拡張アイデア (未実装)

- `realmodel` マーカーの体系化と CI マトリクス (CPU/GPU)
- pytest-cov 導入し閾値管理
- 生成ストリーム chunk の順序/タイミング計測
- 追加のエラーパス (不正 JSON schema, ツール引数バリデーションエラー) テスト

---

## ライセンス

本テストスイートはリポジトリ LICENSE (MIT など) と同一条件に従います。
