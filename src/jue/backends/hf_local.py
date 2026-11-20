"""HuggingFace 本地 CPU/GPU 后端（端侧路线）。

零训练 logits 读出，支持 Qwen3 系列小模型（0.6B/1.7B/4B）。
需要 pip install jue[hf]；首次运行会下载模型权重。
"""
from __future__ import annotations

from ..prompt import LETTERS, build_messages, probs_from_next_token_logits
from ..schema import question_options
from .base import DecisionBackend


class HFLocalBackend(DecisionBackend):
    name = "hf"

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "cpu", dtype: str = "auto"):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
        self.model.eval()
        self.device = device
        self._letter_ids: list[list[int]] = []
        for letter in LETTERS:
            ids = []
            for variant in (letter, letter.lower(), " " + letter, " " + letter.lower()):
                enc = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(enc) == 1 and enc[0] not in ids:
                    ids.append(enc[0])
            self._letter_ids.append(ids)

    def option_probs(self, state, question: dict) -> list[float]:
        torch = self.torch
        n = len(question_options(question))
        messages = build_messages(state, question)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0, -1].float().cpu()
        return probs_from_next_token_logits(logits, self._letter_ids, n)
