"""Jue (决) — 中文优先、校准透明、端侧可跑的开源决策模型。"""
__version__ = "0.0.1"

from .engine import run_decision
from .schema import SchemaError

__all__ = ["run_decision", "SchemaError", "__version__"]
