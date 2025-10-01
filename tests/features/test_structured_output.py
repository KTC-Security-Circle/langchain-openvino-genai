def test_structured_output_flow(ov_llm, pet_schema):
    runnable = ov_llm.with_structured_output(pet_schema)
    # 実際のモデル出力は任意なので、ここでは runnable 連結の実行可能性のみ確認
    # 例外が出なければ最低限成功。更に config が設定されているか検査。
    from langchain_core.runnables import Runnable

    assert isinstance(runnable, Runnable)
    cfg = ov_llm.config.structured_output_config
    assert cfg is not None, "structured_output_config 未設定"
    js = getattr(cfg, "json_schema", None)
    assert js and "name" in js, f"json_schema に期待するキー欠落: {js}"


def test_structured_output_dict_injection(ov_llm):
    schema = {
        "type": "object",
        "properties": {"x": {"type": "number"}},
        "required": ["x"],
    }
    ov_llm.set_structured_output_config(schema)
    cfg = ov_llm.config.structured_output_config
    assert cfg is not None, "structured_output_config が None"
    assert "x" in getattr(cfg, "json_schema", ""), (
        f"スキーマ 'x' 反映失敗: {getattr(cfg, 'json_schema', None)}"
    )
