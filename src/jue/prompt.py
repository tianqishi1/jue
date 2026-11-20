"""零训练 logits 路线的 prompt 构造与首 token 概率解析。

方法（SemIf 式）：把每个问题渲染成单选题，要求模型只输出选项字母，
读取第一个生成 token 在选项字母上的 logprobs，归一化为选项概率。
"""
from __future__ import annotations

import math

from .schema import normalize_state, question_options

SYSTEM_PROMPT = (
    "你是一个严格的决策引擎。根据给定的业务状态对问题做出判断，"
    "只输出一个选项字母，禁止输出任何解释、标点、空格或其他内容。"
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PUNCT = " .。:：、!！?？,，\n\t\r"


def _norm_token(tok: str) -> str:
    """全角字母转半角、去空白与常见标点、转大写。"""
    out = []
    for c in str(tok):
        o = ord(c)
        if 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
            c = chr(o - 0xFEE0)
        out.append(c)
    return "".join(out).strip(_PUNCT).upper()


def render_question(q: dict) -> str:
    lines = [f"问题：{q['instructions']}"]
    if q["type"] == "score":
        lines.append("选项（按程度从低到高排列）：")
    else:
        lines.append("选项：")
    for i, (_, text) in enumerate(question_options(q)):
        lines.append(f"{LETTERS[i]}. {text}")
    lines.append("只输出一个选项字母。")
    return "\n".join(lines)


def build_messages(state, q: dict) -> list[dict]:
    user = f"【业务状态】\n{normalize_state(state)}\n\n【待判断问题】\n{render_question(q)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def letter_index_from_text(text: str, n_options: int) -> int | None:
    """从模型输出文本提取选项字母下标（容错：全角/标点/多字符取首字母）。"""
    t = _norm_token(text)
    if t and "A" <= t[0] <= "Z":
        idx = ord(t[0]) - ord("A")
        if idx < n_options:
            return idx
    return None


def is_uniform(probs: list[float]) -> bool:
    return bool(probs) and max(probs) - min(probs) < 1e-9


def probs_from_top_logprobs(top_logprobs: list[dict], n_options: int) -> list[float]:
    """OpenAI 风格 top_logprobs → 选项概率。

    - 同一字母的大小写/带空格/全角变体概率合并；
    - 未出现在 top_logprobs 中的选项给一个极小地板值（避免 0 概率导致 NLL 爆炸）；
    - 全部缺失时退化为均匀分布（调用方应重试或走兜底）。
    """
    mass = [0.0] * n_options
    for entry in top_logprobs or []:
        tok = _norm_token(entry.get("token", ""))
        if len(tok) == 1 and "A" <= tok <= "Z":
            idx = ord(tok) - ord("A")
            if idx < n_options:
                mass[idx] += math.exp(float(entry.get("logprob", -100.0)))
    present = [m for m in mass if m > 0]
    if not present:
        return [1.0 / n_options] * n_options
    floor = min(present) * 0.01
    mass = [m if m > 0 else floor for m in mass]
    total = sum(mass)
    return [m / total for m in mass]


def probs_from_letter_fallback(idx: int, n_options: int, confidence: float = 0.9) -> list[float]:
    """logprobs 缺失但 content 是字母时的降级兜底（标记 degraded，非真实校准概率）。"""
    rest = (1.0 - confidence) / max(n_options - 1, 1)
    return [confidence if i == idx else rest for i in range(n_options)]


def probs_from_next_token_logits(logits, letter_token_ids: list[list[int]], n_options: int) -> list[float]:
    """HF 本地路线：next-token logits + 每个字母的候选 token id 组 → 选项概率。"""
    import math as _m

    ids_list = list(letter_token_ids[:n_options])
    while len(ids_list) < n_options:
        ids_list.append([])
    scores = []
    for ids in ids_list:
        if not ids:
            scores.append(-1e9)  # 该字母无单 token 变体 → 极低分而非崩溃
            continue
        vals = [float(logits[i]) for i in ids]
        mx = max(vals)
        scores.append(mx + _m.log(sum(_m.exp(v - mx) for v in vals)))
    mx = max(scores)
    exps = [_m.exp(s - mx) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
