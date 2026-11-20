from .base import DecisionBackend
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend
from .systemone import SystemOneBackend

__all__ = ["DecisionBackend", "MockBackend", "OpenAICompatBackend", "SystemOneBackend"]


def make_backend(name: str, **kwargs) -> DecisionBackend:
    if name == "mock":
        return MockBackend()
    if name == "openai":
        return OpenAICompatBackend(**kwargs)
    if name == "systemone":
        return SystemOneBackend(**kwargs)
    if name == "hf":
        from .hf_local import HFLocalBackend
        return HFLocalBackend(**kwargs)
    raise ValueError(f"未知后端: {name}（mock/openai/systemone/hf）")
