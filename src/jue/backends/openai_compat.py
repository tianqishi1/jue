"""OpenAI 兼容 chat/completions + logprobs 后端（零训练路线主力）。

支持任意 OpenAI 兼容端点（DashScope compatible-mode、vLLM、Ollama、llama.cpp server 等）。
默认从环境变量或 ~/.bailian/config.json 的活动 profile 读取配置。

鲁棒性（Phase 0 实测教训）：
- 思考型模型需 enable_thinking=False，否则首 token 可能不是字母；
- 服务端偶发返回空 top_logprobs → 自动重试；
- 重试仍无 logprobs 但 content 是字母 → 降级兜底（degraded 计数，概率非真实校准值）。
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

from ..prompt import (build_messages, is_uniform, letter_index_from_text,
                      probs_from_letter_fallback, probs_from_top_logprobs)
from ..schema import question_options
from .base import DecisionBackend


def bailian_profile() -> dict | None:
    p = pathlib.Path.home() / ".bailian" / "config.json"
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        prof = cfg.get(cfg.get("active_config", ""), {})
        if prof.get("api_key") and prof.get("base_url"):
            return prof
    except Exception:
        pass
    return None


class OpenAICompatBackend(DecisionBackend):
    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        top_logprobs: int = 5,
        timeout: float = 60.0,
        max_retries: int = 3,
        readout_retries: int = 2,
        extra_body: dict | None = None,
    ):
        prof = bailian_profile() or {}
        self.base_url = (base_url or os.environ.get("JUE_BASE_URL") or prof.get("base_url", "")).rstrip("/")
        if self.base_url and not self.base_url.endswith("/chat/completions"):
            if self.base_url.endswith("/v1"):
                self.base_url += "/chat/completions"
            elif "token-plan" in self.base_url:
                self.base_url += "/compatible-mode/v1/chat/completions"
            else:
                self.base_url += "/v1/chat/completions"
        self.api_key = api_key or os.environ.get("JUE_API_KEY") or prof.get("api_key", "")
        self.model = model or os.environ.get("JUE_MODEL") or prof.get("default_text_model", "qwen-flash")
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.max_retries = max_retries
        self.readout_retries = readout_retries
        if extra_body is None:
            host = self.base_url.lower()
            # 仅 DashScope 系端点需要关思考；vLLM/OpenAI 等对未知参数可能 400
            self.extra_body = {"enable_thinking": False} if ("aliyuncs" in host or "dashscope" in host) else {}
        else:
            self.extra_body = dict(extra_body)
        if not self.base_url or not self.api_key:
            raise ValueError("缺少 base_url/api_key：请传参、设 JUE_BASE_URL/JUE_API_KEY，或配置 bl auth")
        self.calls = 0
        self.degraded = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def option_probs(self, state, question: dict) -> list[float]:
        n = len(question_options(question))
        body = {
            "model": self.model,
            "messages": build_messages(state, question),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": min(self.top_logprobs, 5),
            **self.extra_body,
        }
        fallback = None
        for attempt in range(self.readout_retries + 1):
            data = self._post(body)
            self.calls += 1
            usage = data.get("usage") or {}
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            choice = data["choices"][0]
            lp = (choice.get("logprobs") or {}).get("content") or []
            top = lp[0].get("top_logprobs", []) if lp else []
            probs = probs_from_top_logprobs(top, n)
            if not is_uniform(probs):
                return probs
            idx = letter_index_from_text(choice.get("message", {}).get("content", ""), n)
            if idx is not None:
                fallback = probs_from_letter_fallback(idx, n)
        if fallback is not None:
            self.degraded += 1
            return fallback
        return [1.0 / n] * n

    def _post(self, body: dict) -> dict:
        payload = json.dumps(body).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.base_url,
                data=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                last_err = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_err from e
            except Exception as e:  # 网络抖动重试
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        raise last_err  # type: ignore[misc]
