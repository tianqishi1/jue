# Jue（决）

**中文优先 · 校准透明 · 端侧可跑的开源决策模型。**

Jue 是一个 "System One" 决策模型项目（品类由 [Jev](https://jev.ai) 开创）：
输入业务状态 + 类型化问题，输出**结构化答案 + 校准概率**——不生成文本，供程序直接消费。

```
if answer.prob > 0.95:  自动执行
elif answer.prob > 0.7: 辅助人工
else:                   拒答 / 转人工
```

这条自动化规则成立的前提是概率可信。**让每个概率都敢用**，就是 Jue 要做的事。

> 状态：Phase 0（零训练基线 + 评测体系）。模型权重训练在 Phase 1，详见
> [产品设计方案](docs/product-design.md) 与 [技术实现方案](docs/technical-design.md)。

## 为什么是 Jue

开源决策模型生态已有 Kev / Laya / SemIf 等优秀项目，Jue 不做通用复刻，专注三个没人占的组合：

| 支柱 | 含义 |
|---|---|
| 中文优先 | 中文业务语料训练 + 中文决策评测集（客服/电商/审核/风控/Agent 路由） |
| 校准透明 | 发布必须附带 ECE / Brier / 拒答曲线，不允许只报准确率 |
| 端侧可跑 | GGUF 量化 + llama.cpp/ONNX，无独显笔记本 CPU 可用，数据不出本机 |

## 快速开始

```bash
pip install -e .            # 核心零依赖；服务端 pip install -e .[server]

# 决策（默认读 ~/.bailian/config.json 或 JUE_BASE_URL/JUE_API_KEY/JUE_MODEL）
jue decide --state "订单三天没发货，用户要求退款" \
  --questions '{"intent":{"type":"choice","instructions":"核心诉求？","criteria":{"urge":"催发货","refund":"退款"}},"angry":{"type":"noul","instructions":"用户是否愤怒？"}}'

# 跑中文评测集（92 题 / 5 领域）
jue eval --data data/eval/zh-decision-mini.jsonl --backend openai --out reports/my-run

# 本地决策服务（数据不出本机）
jue serve --backend openai --port 8000
# POST http://127.0.0.1:8000/v1/decisions  {"state": ..., "questions": ...}
```

题型与 Jev 兼容：`choice`（N 选 1）、`noul`（是/否）、`score`（有序评分）。
`--abstain 0.7`：最大概率低于阈值 → `answer=null, abstained=true`（拒答是一等公民）。

## 后端

| 后端 | 路线 | 说明 |
|---|---|---|
| `openai` | 零训练 logits | 任意 OpenAI 兼容端点（DashScope/vLLM/llama.cpp/Ollama），读首 token logprobs |
| `hf` | 零训练 logits | 本地 transformers，CPU 可跑（`pip install -e .[hf]`） |
| `systemone` | 原生决策 API | Jev / 百炼 decision-model，作 teacher 与对照（实验性） |
| `mock` | 确定性伪概率 | 仅测试/CI |

## Phase 0 基线结果（2026-11-20 实测）

`zh-decision-mini`（40 states / 92 questions）× qwen3.8-max 零训练 logits 读出：

| 指标 | 数值 |
|---|---|
| 准确率 | **94.6%**（choice 97.6% / noul 100% / score 75%） |
| ECE / Brier（原始） | **0.047** / 0.104（原始 logits 已相当校准） |
| ECE（收缩温度缩放后） | **0.040**（T=3.25→收缩 1.25，小样本防过拟合） |
| score 相邻等级准确率 | 100%（score 错误全部是相邻等级） |
| 拒答 τ=0.95 | 覆盖 62%，保留部分准确率 98.3% |

详见 [Phase 0 实测发现](docs/phase0-findings.md)。关键教训已固化进代码：
思考型模型必须 `enable_thinking=False`；服务端偶发空 logprobs 需重试；
小样本温度缩放会过拟合 → 已用收缩先验修复（`--shrink-n0`，报告同时给原始/校准后指标）。

## 评测集

`data/eval/zh-decision-mini.jsonl`：5 领域 × 8 states，含反讽、隐含意图、
条件性诉求、差评≠辱骂、常识≠诊疗建议、最低还款≠高风险等边界用例。
指标：Accuracy / ECE / Brier / NLL / 拒答曲线；校准/holdout 按 id 哈希确定性切分。

## 路线图

- [x] Phase 0：schema/引擎/校准/评测 harness + 零训练基线 + mini 评测集
- [ ] Phase 1：中文数据合成 → LoRA + pointer head 训练 → 校准 → HF 权重 + GGUF
- [ ] Phase 2：2000+ 题评测集、SDK/文档、社区发布

## 许可证

Apache-2.0。名称 "Jue/决" 为暂定名，发布前做商标与名称冲突检查；与 TypeSafe AI 的 Jev 无关联。
