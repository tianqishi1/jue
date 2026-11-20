import math
from jue.calibration import (apply_temperature, brier, ece, fit_temperature, nll,
                             abstention_curve, summarize)


def test_temperature_normalizes():
    p = apply_temperature([0.9, 0.1], 2.0)
    assert abs(sum(p) - 1) < 1e-9
    assert p[0] < 0.9  # T>1 使分布更平


def test_brier_nll():
    assert abs(brier([1.0, 0.0], 0)) < 1e-12
    assert abs(brier([0.5, 0.5], 0) - 0.5) < 1e-12
    assert abs(nll([0.5, 0.5], 1) - math.log(2)) < 1e-9


def test_ece_perfect_and_overconfident():
    # 完全校准：置信度==正确率
    confs = [0.8] * 10
    corrects = [True] * 8 + [False] * 2
    assert ece(confs, corrects) < 1e-9
    # 过度自信：全说0.99但只对一半
    confs2 = [0.99] * 10
    corrects2 = [True] * 5 + [False] * 5
    assert ece(confs2, corrects2) > 0.4


def test_fit_temperature_fixes_overconfidence():
    # 模型总说 [0.95, 0.05]，实际一半对一半错 → 应拟合出 T > 1
    records = [([0.95, 0.05], i % 2) for i in range(40)]
    t = fit_temperature(records)
    assert t > 1.5
    p = apply_temperature([0.95, 0.05], t)
    assert p[0] < 0.75


def test_summarize_and_abstention():
    recs = [
        {"probs": [0.9, 0.1], "label": 0, "correct": True, "domain": "a", "type": "noul"},
        {"probs": [0.4, 0.6], "label": 0, "correct": False, "domain": "b", "type": "noul"},
    ]
    s = summarize(recs)
    assert s["n"] == 2 and s["accuracy"] == 0.5
    assert s["by_domain"]["a"]["accuracy"] == 1.0
    curve = abstention_curve(recs, thresholds=(0.5, 0.9))
    assert curve[0]["coverage"] == 1.0
    assert curve[1]["coverage"] == 0.5 and curve[1]["selective_accuracy"] == 1.0


def test_score_adjacent_accuracy():
    recs = [
        {"probs": [0.1, 0.8, 0.1, 0.0], "label": 2, "correct": False, "domain": "r", "type": "score"},
        {"probs": [0.7, 0.2, 0.1, 0.0], "label": 0, "correct": True, "domain": "r", "type": "score"},
        {"probs": [0.9, 0.1], "label": 1, "correct": False, "domain": "r", "type": "noul"},
    ]
    s = summarize(recs)
    assert s["accuracy"] == 1 / 3
    assert s["score_adjacent_accuracy"] == 1.0  # score 错的那题是相邻等级
