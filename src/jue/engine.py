"""决策引擎：backend 概率 → 结构化答案（含拒答）。"""
from __future__ import annotations

from .backends.base import DecisionBackend
from .schema import format_answer, normalize_state, validate_questions


def run_decision(backend: DecisionBackend, state, questions: dict,
                 abstain_threshold: float | None = None) -> dict:
    qs = validate_questions(questions)
    state_str = normalize_state(state)
    answers = {}
    for qid, q in qs.items():
        probs = backend.option_probs(state_str, q)
        answers[qid] = format_answer(q, probs, abstain_threshold)
    return answers
