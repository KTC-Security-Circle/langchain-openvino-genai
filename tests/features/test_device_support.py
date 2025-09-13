"""デバイス(CPU/GPU/NPU)サポートの簡易検証テスト。

動作方針:
- 指定された各デバイス(フィクスチャ ov_test_devices)で from_model_path を試す
- max_length(相当) を小さく (例: config.max_new_tokens / generation_config の属性があれば設定) して高速終了
- 成功: 1トークン以上のテキストが返ること
- 失敗: RuntimeError や NotImplementedError 等 → xfail 扱い (レポートで可視化)

注意:
- 実モデルが無い場合は上位フィクスチャで全体 skip 済み
- デバイス毎に初期化コストが高い場合はマーカー slow を付与可 (必要なら)
"""

from __future__ import annotations

import pytest

from langchain_openvino_genai.llm_model import OpenVINOLLM


@pytest.mark.compliance
def test_llm_runs_on_devices(dummy_model_path, ov_test_devices):
    """各デバイスを順次検証し *GPU 失敗で NPU がスキップされる* ことを防ぐ集約方式。

    ポリシー:
      - 初期化例外 ⇒ そのデバイスは "xfail" (非対応/ドライバ未設定) と記録し続行
      - 推論/アサーション失敗 ⇒ "fail" と記録 (他デバイスは続行し最後にまとめて失敗)
      - 1 つでも PASS があればテスト全体は PASS (ただし fail があれば失敗)
      - 全て xfail (PASS なし) の場合のみ pytest.xfail で終了
    """
    results = []  # list of dict(status, device, detail)
    for dev in ov_test_devices:
        label = f"[device={dev}]"
        try:
            try:
                llm = OpenVINOLLM.from_model_path(
                    model_path=dummy_model_path, device=dev
                )
            except Exception as e:  # 初期化不可 → xfail 扱い (他継続)
                print(f"DEVICE {dev}: XFAIL init_error={e}")
                results.append({"device": dev, "status": "xfail", "detail": str(e)})
                continue

            # 生成長短縮 (存在する最初の属性を 30 に)
            cfg = llm.config
            for attr_name in ["max_new_tokens", "max_length", "max_tokens"]:
                if hasattr(cfg, attr_name):
                    setattr(cfg, attr_name, 30)
                    break

            out = llm.invoke("Hello")
            if not isinstance(out, str):
                raise AssertionError(f"{label} 出力型が str ではない: {type(out)}")
            if not out:
                raise AssertionError(f"{label} 出力が空 (推論失敗の可能性)")
            short = llm.invoke("Hello", stop=["RESP"])  # type: ignore[arg-type]
            if "RESP" in short:
                raise AssertionError(f"{label} stop が効いていない: {short!r}")
            print(f"DEVICE {dev}: PASS len={len(out)}")
            results.append(
                {"device": dev, "status": "pass", "detail": f"len={len(out)}"}
            )
        except AssertionError as ae:  # 生成後の検証失敗
            print(f"DEVICE {dev}: FAIL {ae}")
            results.append({"device": dev, "status": "fail", "detail": str(ae)})
        except Exception as e:  # その他例外 (念のため)
            print(f"DEVICE {dev}: ERROR {e}")
            results.append({"device": dev, "status": "fail", "detail": str(e)})

    any_pass = any(r["status"] == "pass" for r in results)
    any_fail = any(r["status"] == "fail" for r in results)
    if not any_pass:
        # 全部 xfail (非対応) もしくは fail だけの場合で pass 無し
        # fail が含まれるなら assertion で失敗させ、xfail のみなら xfail
        if any_fail:
            summary = ", ".join(f"{r['device']}={r['status']}" for r in results)
            pytest.fail(f"全デバイスで成功なし: {summary}")
        else:
            summary = ", ".join(f"{r['device']}=xfail" for r in results)
            pytest.xfail(f"全デバイス非対応 (PASS 無し): {summary}")
    elif any_fail:
        # 一部成功したが失敗デバイスあり → 失敗させて気づけるようにする
        summary = ", ".join(f"{r['device']}={r['status']}" for r in results)
        pytest.fail(f"一部デバイス失敗: {summary}")
