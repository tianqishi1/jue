"""SystemOne 原生决策 API 后端（Jev / 百炼 decision-model-preview）。

直接消费服务端返回的校准概率，用作 teacher 与对照基线。
响应结构做防御性解析（不同厂商字段略有差异）。实验性。
"""
from __future__ import annotations

import json
import os
import urllib.request

from ..schema import question_options
from .base import DecisionBackend


class SystemOneBackend(DecisionBackend):
    name = "systemone"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str = "decision-model-preview", timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("JUE_SYSTEMONE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("JUE_SYSTEMONE_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise ValueError("SystemOne 后端需要 base_url 与 api_key（JUE_SYSTEMONE_URL/JUE_SYSTEMONE_KEY）")

    def option_probs(self, state, question: dict) -> list[float]:
        keys = [k for k, _ in question_options(question)]
        q = {k: v for k, v in question.items() if k != "label"}
        body = {"model": self.model, "state": state, "questions": {"q": q}}
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return self._extract_probs(data, keys, question["type"])

    @staticmethod
    def _extract_probs(data: dict, keys: list[str], qtype: str) -> list[float]:
        node = data
        for wrapper in ("output", "result", "data"):
            if isinstance(node, dict) and wrapper in node and isinstance(node[wrapper], dict):
                node = node[wrapper]
        answers = node.get("answers") or node.get("questions") or node
        a = answers.get("q") if isinstance(answers, dict) else None
        if a is None:
            raise RuntimeError(f"无法解析 SystemOne 响应: {json.dumps(data, ensure_ascii=False)[:300]}")
        if qtype == "noul":
            p = a.get("prob", a.get("probability"))
            if p is None:
                probs = a.get("probs") or a.get("probabilities") or {}
                p = probs.get("true", probs.get("是"))
            if p is None:
                raise RuntimeError(f"noul 响应缺少概率: {a}")
            return [float(p), 1.0 - float(p)]
        probs = a.get("probs") or a.get("probabilities")
        if isinstance(probs, dict):
            vals = [float(probs.get(k, 0.0)) for k in keys]
        elif isinstance(probs, list):
            vals = [float(x) for x in probs]
        else:
            raise RuntimeError(f"响应缺少 probs: {a}")
        total = sum(vals)
        if total <= 0:
            raise RuntimeError(f"概率和为 0: {a}")
        return [v / total for v in vals]
