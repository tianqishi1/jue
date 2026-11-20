"""校准与指标：温度缩放、ECE、Brier、NLL、拒答曲线。

校准是本项目的核心卖点：模型说 0.9 时，应当真的对 90%。
"""
from __future__ import annotations

import math


def apply_temperature(probs: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0")
    powered = [max(p, 1e-12) ** (1.0 / temperature) for p in probs]
    total = sum(powered)
    return [p / total for p in powered]


def nll(probs: list[float], label: int) -> float:
    return -math.log(max(probs[label], 1e-12))


def brier(probs: list[float], label: int) -> float:
    return sum((p - (1.0 if i == label else 0.0)) ** 2 for i, p in enumerate(probs))


def ece(confidences: list[float], corrects: list[bool], bins: int = 10) -> float:
    """Expected Calibration Error（按最大概率置信度分箱）。"""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idxs = [
            i for i, c in enumerate(confidences)
            if (lo < c <= hi) or (b == 0 and c == 0.0)
        ]
        if not idxs:
            continue
        acc = sum(corrects[i] for i in idxs) / len(idxs)
        conf = sum(confidences[i] for i in idxs) / len(idxs)
        total += len(idxs) / n * abs(acc - conf)
    return total


def fit_temperature(records: list[tuple[list[float], int]], grid_size: int = 60) -> float:
    """在 records 上最小化平均 NLL 拟合单参数温度（粗网格 + 细化）。"""
    if not records:
        return 1.0

    def loss(t: float) -> float:
        return sum(nll(apply_temperature(p, t), y) for p, y in records) / len(records)

    best_t, best_l = 1.0, loss(1.0)
    lo, hi = 0.05, 8.0
    for _ in range(2):
        step = (hi - lo) / grid_size
        for i in range(grid_size + 1):
            t = lo + i * step
            l = loss(t)
            if l < best_l:
                best_t, best_l = t, l
        lo, hi = max(0.05, best_t - step * 2), best_t + step * 2
    return best_t


def summarize(records: list[dict]) -> dict:
    """records: [{probs, label, correct, domain, type, ...}] → 汇总指标。"""
    n = len(records)
    if n == 0:
        return {"n": 0}
    confs = [max(r["probs"]) for r in records]
    corrects = [bool(r["correct"]) for r in records]
    out = {
        "n": n,
        "accuracy": sum(corrects) / n,
        "ece": ece(confs, corrects),
        "brier": sum(brier(r["probs"], r["label"]) for r in records) / n,
        "nll": sum(nll(r["probs"], r["label"]) for r in records) / n,
        "mean_confidence": sum(confs) / n,
    }
    for key in ("domain", "type"):
        groups: dict = {}
        for r in records:
            groups.setdefault(r.get(key, "?"), []).append(r)
        out[f"by_{key}"] = {
            g: {"n": len(rs), "accuracy": sum(x["correct"] for x in rs) / len(rs)}
            for g, rs in sorted(groups.items())
        }
    score_recs = [r for r in records if r.get("type") == "score"]
    if score_recs:
        adj = sum(
            abs(max(range(len(r["probs"])), key=r["probs"].__getitem__) - r["label"]) <= 1
            for r in score_recs
        )
        out["score_adjacent_accuracy"] = adj / len(score_recs)
    return out


def abstention_curve(records: list[dict], thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)) -> list[dict]:
    """拒答曲线：max prob < 阈值 → 拒答。coverage=保留比例，selective_accuracy=保留部分准确率。"""
    out = []
    n = len(records)
    for t in thresholds:
        kept = [r for r in records if max(r["probs"]) >= t]
        out.append({
            "threshold": t,
            "coverage": len(kept) / n if n else 0.0,
            "selective_accuracy": (sum(r["correct"] for r in kept) / len(kept)) if kept else None,
            "abstained": n - len(kept),
        })
    return out
