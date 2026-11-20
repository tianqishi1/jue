"""Jue 决策请求/响应 schema 与校验。

题型与 Jev 兼容：choice（N 选 1）、noul（是/否）、score（有序等级评分）。
Phase 0 限制：单题选项数 2-26（选项字母 A-Z）。
"""
from __future__ import annotations

import json
from typing import Any

MAX_OPTIONS = 26
QUESTION_TYPES = ("choice", "noul", "score")


class SchemaError(ValueError):
    """请求或标签不符合 schema。"""


def normalize_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def validate_question(q: Any) -> dict:
    if not isinstance(q, dict):
        raise SchemaError("question 必须是对象")
    t = q.get("type")
    if t not in QUESTION_TYPES:
        raise SchemaError(f"未知题型: {t!r}（支持 {QUESTION_TYPES}）")
    ins = q.get("instructions")
    if not isinstance(ins, str) or not ins.strip():
        raise SchemaError("question.instructions 必须是非空字符串")
    out: dict = {"type": t, "instructions": ins}
    if t == "choice":
        c = q.get("criteria")
        if not isinstance(c, dict) or not (2 <= len(c) <= MAX_OPTIONS):
            raise SchemaError("choice.criteria 必须是 2-26 项的对象 {选项: 描述}")
        out["criteria"] = {str(k): str(v) for k, v in c.items()}
    elif t == "noul":
        c = q.get("criteria") or {}
        if not isinstance(c, dict):
            raise SchemaError("noul.criteria 必须是对象（可选 true/false 描述）")
        out["criteria"] = {k: str(v) for k, v in c.items() if k in ("true", "false")}
    else:  # score
        c = q.get("criteria")
        if not isinstance(c, list) or not (2 <= len(c) <= MAX_OPTIONS):
            raise SchemaError("score.criteria 必须是 2-26 项的有序列表（低→高）")
        out["criteria"] = [str(x) for x in c]
    return out


def validate_questions(questions: Any) -> dict:
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("questions 必须是非空对象 {question_id: question}")
    return {str(qid): validate_question(q) for qid, q in questions.items()}


def question_options(q: dict) -> list[tuple[str, str]]:
    """返回 (规范化 key, 渲染文本) 列表；顺序即概率向量顺序。"""
    t = q["type"]
    if t == "choice":
        return [(k, f"{k}：{v}" if v else k) for k, v in q["criteria"].items()]
    if t == "noul":
        c = q.get("criteria") or {}
        return [
            ("true", "是" + (f"（{c['true']}）" if c.get("true") else "")),
            ("false", "否" + (f"（{c['false']}）" if c.get("false") else "")),
        ]
    return [(str(i), s) for i, s in enumerate(q["criteria"])]


def label_index(q: dict, label: Any) -> int:
    """把数据集标签映射为概率向量下标。"""
    keys = [k for k, _ in question_options(q)]
    t = q["type"]
    if t == "noul":
        if isinstance(label, bool):
            return 0 if label else 1
        s = str(label).strip().lower()
        if s in ("true", "是", "yes", "1"):
            return 0
        if s in ("false", "否", "no", "0"):
            return 1
        raise SchemaError(f"noul 标签无法识别: {label!r}")
    if t == "score":
        if isinstance(label, int) and 0 <= label < len(keys):
            return label
        s = str(label).strip()
        if s in q["criteria"]:
            return q["criteria"].index(s)
        if s.isdigit() and 0 <= int(s) < len(keys):
            return int(s)
        raise SchemaError(f"score 标签无法识别: {label!r}")
    s = str(label).strip()
    if s in keys:
        return keys.index(s)
    raise SchemaError(f"choice 标签 {label!r} 不在选项中: {keys}")


def format_answer(q: dict, probs: list[float], abstain_threshold: float | None = None) -> dict:
    """概率向量 → Jev 风格的结构化答案。"""
    keys = [k for k, _ in question_options(q)]
    best = max(range(len(probs)), key=probs.__getitem__)
    abstained = abstain_threshold is not None and probs[best] < abstain_threshold
    t = q["type"]
    if t == "choice":
        return {
            "type": "choice",
            "answer": None if abstained else keys[best],
            "probs": {k: round(p, 6) for k, p in zip(keys, probs)},
            "abstained": abstained,
        }
    if t == "noul":
        return {
            "type": "noul",
            "answer": None if abstained else best == 0,
            "prob": round(probs[0], 6),
            "abstained": abstained,
        }
    expected = sum(i * p for i, p in enumerate(probs))
    return {
        "type": "score",
        "answer": None if abstained else q["criteria"][best],
        "probs": [round(p, 6) for p in probs],
        "expected": round(expected, 4),
        "abstained": abstained,
    }
