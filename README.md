# langchain-openvino-genai

LangChain から OpenVINO GenAI を手軽に使うための軽量ラッパーです。LLM 用の `OpenVINOLLM` と Chat 用の `ChatOpenVINO` を提供し、Hugging Face Hub からのモデル取得ヘルパー `load_model` も同梱しています。
LangChainにすでに実装されている [LangChain OpenVINO](https://python.langchain.com/docs/integrations/llms/openvino/) と異なる部分は、OpenVINO GenAI の LLMPipeline を利用している点です。
これにより、NPU での推論が可能になります。
ただし、すべてのLLMモデルが対応しているわけではないので、使用するモデルの互換性を確認してください。

このリポジトリにあるコードは IntelCorporation の [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks) プロジェクトにあるコードを基にしています。詳細は `NOTICE` ファイルを参照してください。

## 特長

- OpenVINO GenAI の LLMPipeline を LangChain LLM/ChatModel として利用
- ストリーミング出力対応（トークンチャンク）
- ChatML 互換のテンプレート組み立て（System/Human/AI メッセージ）
- Hugging Face Hub からのモデル自動ダウンロード（`llms/` 直下に配置）

## 対応要件

- Python 3.12+
- openvino-genai >= 2025.3.0.0

Windows/Intel NPU を利用する場合は、対応 NPU ドライバのインストールが必要になる場合があります（例: Intel NPU Driver for Windows）。CPU/GPU/NPU などのデバイス指定はコードの `device` 引数で切り替えます。

### 動作確認済みのモデル例

Hugging Face Hub 上のIR変換済みモデル:

- [OpenVINO/Qwen3-8B-int4-cw-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov)

IR変換を行い動作を確認できたモデル:

- まだありません。コントリビューション歓迎します。

## インストール

uv を利用（推奨）

```powershell
uv add git+https://github.com/KTC-Security-Circle/langchain-openvino-genai.git

# or use tag
uv add git+https://github.com/KTC-Security-Circle/langchain-openvino-genai.git --tag 0.0.1
```

pip を利用（代替）

```powershell
pip install git+https://github.com/KTC-Security-Circle/langchain-openvino-genai.git
```

## 使い方（最小例）

`scripts/` 以下にサンプルコードを用意しています。
実際に実行する場合はコードをコピーして使用するか、`langchain-openvino-genai` をクローンして実行してください。

LLMをローカルにダウンロードする際にダウンロードパスを指定しなければ、カレントディレクトリに`llms/` フォルダが作成され、そこにモデルが保存されます。
.gitignore に `llms/` を追加しておくことを忘れないでください。

### クローンして実行する方法

```powershell
git clone https://github.com/KTC-Security-Circle/langchain-openvino-genai.git
cd langchain-openvino-genai
uv sync

# llm 例
uv run scripts/run_llm_model.py
# chat 例
uv run scripts/run_chat_model.py
# chat ストリーミング例
uv run scripts/run_stream_chat.py
# 構造化出力例
uv run scripts/run_structured_output.py
```

## ライセンス

このプロジェクトは Apache License 2.0 で提供されます。詳細は `LICENCE` および `NOTICE` を参照してください。
