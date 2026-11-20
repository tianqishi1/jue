"""确定性 mock 后端：仅用于测试与 CI，不代表任何模型能力。"""
from __future__ import annotations

import hashlib
import math

from ..schema import question_options
from .base import DecisionBackend


class MockBackend(DecisionBackend):
    name = "mock"

    def option_probs(self, state, question: dict) -> list[float]:
        opts = question_options(question)
        weights = []
        for key, _ in opts:
            seed = f"{state}|{question['instructions']}|{key}".encode("utf-8")
            h = int(hashlib.sha256(seed).hexdigest()[:8], 16)
            weights.append(1.0 + (h % 1000) / 1000.0)
        total = sum(weights)
        return [w / total for w in weights]
