"""评估修复的回归测试（P1/P2 七项）。"""
from types import SimpleNamespace

from jue.backends.openai_compat import OpenAICompatBackend
from jue.calibration import fit_temperature, shrink_temperature
from jue.cli import _backend_kwargs
from jue.prompt import probs_from_next_token_logits


def _b(base_url, **kw):
    return OpenAICompatBackend(base_url=base_url, api_key="test-key", model="m", **kw)


def test_url_normalization():
    # 回归：标准 DashScope base 曾被拼出双重 /v1
    assert _b("https://dashscope.aliyuncs.com/compatible-mode/v1").base_url == \
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert _b("https://token-plan.cn-beijing.maas.aliyuncs.com").base_url == \
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert _b("http://127.0.0.1:8000/v1").base_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert _b("http://127.0.0.1:8080").base_url == "http://127.0.0.1:8080/v1/chat/completions"
    full = "http://x/v1/chat/completions"
    assert _b(full).base_url == full


def test_extra_body_autodetect():
    assert _b("https://dashscope.aliyuncs.com/compatible-mode/v1").extra_body == {"enable_thinking": False}
    assert _b("http://127.0.0.1:8000/v1").extra_body == {}  # vLLM/本地端点不注入未知参数
    assert _b("http://x/v1", extra_body={"foo": 1}).extra_body == {"foo": 1}


def test_shrink_temperature_math():
    assert shrink_temperature(3.25, 46, 200) < 1.4          # 小样本强收缩
    assert abs(shrink_temperature(3.25, 100000, 200) - 3.25) < 0.01  # 大样本≈原值
    assert shrink_temperature(3.25, 46, 0) == 3.25          # n0=0 禁用
    assert shrink_temperature(1.0, 10) == 1.0


def test_shrink_fixes_small_sample_overfit():
    # 42 题：40 个高置信正确 + 2 个高置信错误 → 纯 NLL 拟合 T>2，收缩后应接近 1
    recs = [([0.999, 0.001], 0) for _ in range(40)] + [([0.999, 0.001], 1)] * 2
    t = fit_temperature(recs)
    assert t > 2
    assert shrink_temperature(t, len(recs), 200) < 1.5


def test_empty_letter_ids_guard():
    logits = [0.0] * 50
    logits[10] = 5.0
    p = probs_from_next_token_logits(logits, [[10], []], 2)  # B 无候选 token
    assert abs(sum(p) - 1) < 1e-9 and p[0] > 0.99
    p2 = probs_from_next_token_logits(logits, [[10]], 2)    # 候选列表不足也补齐
    assert len(p2) == 2


def test_cli_kwargs_per_backend():
    hf = SimpleNamespace(backend="hf", model="Qwen/Qwen3-0.6B", model_name=None,
                         base_url=None, api_key=None)
    assert _backend_kwargs(hf) == {"model_name": "Qwen/Qwen3-0.6B"}  # 回归：曾 TypeError
    oa = SimpleNamespace(backend="openai", model="m", model_name=None,
                         base_url="u", api_key="k")
    assert _backend_kwargs(oa) == {"base_url": "u", "api_key": "k", "model": "m"}


def test_decide_requires_state():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "jue.cli", "decide", "--backend", "mock",
                        "--questions", '{"q":{"type":"noul","instructions":"x?"}}'],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and "state" in (r.stderr + r.stdout)
