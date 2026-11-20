from jue.prompt import (_norm_token, is_uniform, letter_index_from_text,
                        probs_from_letter_fallback, probs_from_top_logprobs)
import math


def test_norm_fullwidth_and_punct():
    assert _norm_token("Ａ") == "A"
    assert _norm_token("ｂ。") == "B"
    assert _norm_token(" C ") == "C"


def test_letter_index_from_text():
    assert letter_index_from_text("Ｂ", 3) == 1
    assert letter_index_from_text("A.", 3) == 0
    assert letter_index_from_text("我觉得选C", 3) is None  # 首字符不是字母
    assert letter_index_from_text("E", 3) is None          # 超出选项数


def test_fullwidth_in_top_logprobs():
    top = [{"token": "Ｂ", "logprob": math.log(0.7)}, {"token": "A", "logprob": math.log(0.2)}]
    p = probs_from_top_logprobs(top, 3)
    assert p[1] > p[0] > p[2]


def test_uniform_and_fallback():
    assert is_uniform([0.25] * 4)
    assert not is_uniform([0.9, 0.1])
    f = probs_from_letter_fallback(1, 3)
    assert abs(sum(f) - 1) < 1e-9 and f[1] == 0.9


# --- 原有覆盖（合并保留） ---
from jue.prompt import build_messages, probs_from_next_token_logits, render_question
from jue.schema import validate_question

Q = validate_question({"type": "choice", "instructions": "选哪个？",
                       "criteria": {"a": "选项A", "b": "选项B", "c": "选项C"}})


def test_render_and_messages():
    text = render_question(Q)
    assert "A. a：选项A" in text and "C." in text
    msgs = build_messages({"x": 1}, Q)
    assert msgs[0]["role"] == "system" and "业务状态" in msgs[1]["content"]


def test_top_logprobs_variants_merge():
    top = [{"token": "A", "logprob": math.log(0.6)},
           {"token": " a", "logprob": math.log(0.1)},
           {"token": "B", "logprob": math.log(0.2)}]
    p = probs_from_top_logprobs(top, 3)
    assert abs(sum(p) - 1) < 1e-9
    assert p[0] > p[1] > p[2]
    assert p[2] < 0.01


def test_top_logprobs_empty_uniform():
    p = probs_from_top_logprobs([], 4)
    assert all(abs(x - 0.25) < 1e-9 for x in p)


def test_next_token_logits():
    logits = [0.0] * 100
    logits[10], logits[11] = 5.0, 4.0
    logits[20] = 3.0
    letter_ids = [[10, 11], [20], [30]]
    p = probs_from_next_token_logits(logits, letter_ids, 3)
    assert abs(sum(p) - 1) < 1e-9 and p[0] > p[1] > p[2]
