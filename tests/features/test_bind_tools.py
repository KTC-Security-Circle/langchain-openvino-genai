from langchain_core.messages import HumanMessage


def test_bind_tools_execution(chat_model, two_tools):
    chat_model.bind_tools(two_tools)
    # Runnable チェーンが構築され schema が設定される
    cfg = chat_model.llm.config.structured_output_config
    assert cfg is not None, "structured_output_config 未設定"
    js = getattr(cfg, "json_schema", None)
    assert js and "tool_name" in js, f"tool_name が schema に存在しない: {js}"
    # system 追加
    assert chat_model._additional_system_message is not None, (
        "system ツールメッセージ未設定"
    )
    content = chat_model._additional_system_message.content
    # 各ツールシグネチャの一部
    for tool in two_tools:
        assert tool.name in content, (
            f"ツール {tool.name} が system prompt に出現しない: {content}"
        )


def test_bind_tools_invoke_roundtrip(chat_model, two_tools):
    chat_model.bind_tools(two_tools)
    # 実際の LLM は JSON を確実に返すとは限らないため
    # ここでは invoke 実行が例外なく走ることと、失敗時に情報を出す
    try:
        _ = chat_model.invoke([HumanMessage(content="echo this text please")])
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"bind_tools 後の invoke が失敗: {e}")
