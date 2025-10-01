from langchain_core.messages import HumanMessage


def test_chat_prompt_conversion(chat_model):
    msg = [HumanMessage(content="こんにちは")]  # 最低1つの HumanMessage
    prompt = chat_model._to_chat_prompt(msg)
    assert isinstance(prompt, str), f"文字列が期待: type={type(prompt)}"
    assert "user:" in prompt or "user" in prompt, (
        f"user メッセージがテンプレートに反映されていません: {prompt!r}"
    )


def test_chat_error_on_last_not_human(chat_model):
    # 最後がAIならエラーになるはず
    from langchain_core.messages import AIMessage

    bad = [HumanMessage(content="Q"), AIMessage(content="A")]
    try:
        chat_model._to_chat_prompt(bad)
    except ValueError as e:
        assert "Last message" in str(e)
    else:
        raise AssertionError("最後がHumanMessageでないのにエラーになりませんでした")


def test_chat_generate_basic(chat_model):
    out = chat_model.invoke([HumanMessage(content="Hi")])
    from langchain_core.messages import AIMessage

    assert isinstance(out, AIMessage), (
        f"AIMessage が期待されます: type={type(out)} 値={out}"
    )
    assert out.content is not None, "content が空です"


def test_bind_tools_injects_system(echo_tool, chat_model):
    runnable = chat_model.bind_tools([echo_tool])
    # system メッセージが追加される: _additional_system_message
    assert chat_model._additional_system_message is not None, (
        "ツール system メッセージが設定されていません"
    )
    assert "echo(" in chat_model._additional_system_message.content, (
        "ツールシグネチャが含まれていません"
    )
    # runnable が Runnable チェーンであること
    from langchain_core.runnables import Runnable

    assert isinstance(runnable, Runnable)


def test_bind_tools_schema(chat_model, echo_tool):
    chat_model.bind_tools([echo_tool])
    cfg = chat_model.llm.config
    so = cfg.structured_output_config
    assert so is not None, "structured_output_config が None"
    js = getattr(so, "json_schema", None)
    assert js, "json_schema が空"
    assert "tool_name" in js and "arguments" in js, f"schema に必要キー不足: {js}"
