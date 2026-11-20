"""后端抽象：所有后端把 (state, question) 映射为选项概率向量。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class DecisionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def option_probs(self, state, question: dict) -> list[float]:
        """返回与 schema.question_options(question) 顺序对齐、和为 1 的概率。"""

    def close(self) -> None:  # pragma: no cover
        pass
