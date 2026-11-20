"""本地决策服务：POST /v1/decisions（Jev 风格），数据不出本机。

注意：本模块刻意不用 `from __future__ import annotations`——
DecisionRequest 定义在工厂函数内部，延迟注解会让 FastAPI 无法解析Body模型。
"""
from .backends.base import DecisionBackend
from .engine import run_decision
from .schema import SchemaError


def create_app(backend: DecisionBackend, abstain_threshold: float | None = None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Any

    app = FastAPI(title="Jue Decision Server", version="0.0.1")

    class DecisionRequest(BaseModel):
        state: Any
        questions: dict
        abstain_threshold: float | None = None

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "backend": backend.name, "model": getattr(backend, "model", None)}

    @app.post("/v1/decisions")
    def decisions(req: DecisionRequest):
        try:
            answers = run_decision(
                backend, req.state, req.questions,
                abstain_threshold=req.abstain_threshold if req.abstain_threshold is not None else abstain_threshold,
            )
        except SchemaError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"backend error: {e}")
        return {"model": getattr(backend, "model", backend.name), "answers": answers}

    return app
