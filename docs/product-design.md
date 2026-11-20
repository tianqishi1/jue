# Jue（决）产品设计方案

> 版本 v0.1 · 2026-11-20 · 状态：Phase 0 开发中
> 一句话：**中文优先、校准透明、端侧可跑的开源决策模型。**

## 1. 背景与定位

Jev（TypeSafe AI，2026-09）开创了 "System One" 决策模型品类：输入业务状态 + 类型化问题，
输出结构化答案 + 校准概率，不生成文本。两个月内开源生态已拥挤（Kev、Laya、SemIf、OpenJev 等），
OpenAI / Google 也已推出商业决策模型，OpenRouter 上架 13 个决策模型。

**通用复刻没有生态位。Jue 的差异化定位是三个支柱的组合：**

| 支柱 | 回答的问题 | 现状空隙 |
|---|---|---|
| 中文优先 | 判断得准不准 | 开源模型英文语料为主；中文最强的百炼决策模型闭源 |
| 校准透明 | 概率敢不敢用 | 开源复刻普遍只报准确率，不报 ECE/Brier，无拒答机制 |
| 端侧可跑 | 在哪跑、多少钱 | 现有部署路线 = vLLM（N卡）/ MLX（Mac），CPU/边缘无人认真做 |

三者互相咬合：端侧约束 → 小模型 → 容量有限 → 更依赖中文专项训练与校准把有限能力用足。

**不做什么**：不做通用聊天、不做文本生成、不做"又一个 Jev 英文复刻"、Phase 1 前不做多模态。

## 2. 目标用户与场景

| 用户 | 场景 | 关键诉求 |
|---|---|---|
| 中小电商/客服团队 | 工单分流、退款意图识别、情绪分级 | 中文准、便宜、可私有化 |
| 内容平台/社区 | 评论审核、广告识别、风险分级 | 数据不出域、低延迟、可解释概率 |
| Agent 开发者 | 工具路由、结果校验、意图判断 | 毫秒级、结构化输出、可本地跑 |
| 金融/风控团队 | 交易拦截、欺诈分类、审批分流 | 校准概率支撑阈值自动化、可审计 |
| 研究者 | 决策模型校准研究 | 开放权重、开放评测集、可复现 |

核心自动化范式（产品存在的理由）：

```
if answer.prob > 0.95:  自动执行
elif answer.prob > 0.7: 辅助人工
else:                   拒答/转人工
```

这条规则成立的前提是**概率可信**——校准是产品功能，不是论文指标。

## 3. 产品形态（交付物）

1. **模型权重**：中文决策微调小模型（0.6B–4B，Qwen 系底座），HF 发布，含 GGUF 量化版。
2. **Python SDK + CLI**：`pip install jue`；`jue decide / eval / serve`。
3. **本地决策服务**：`jue serve` 起 OpenAI 风格 `POST /v1/decisions`，数据不出本机。
4. **中文决策评测集**：`zh-decision-mini`（Phase 0，92 题）→ `zh-decision-bench`（Phase 2，目标 2000+ 题），
   含准确率 + ECE + Brier + 拒答曲线四类指标，成为中文决策模型的"尺子"。
5. **训练管线**：数据合成脚本 + LoRA + pointer head + 校准，Apache-2.0 全开源。

### API 设计（Jev 兼容）

```json
POST /v1/decisions
{
  "state": "订单付款三天未发货，用户要求退款并威胁投诉",
  "questions": {
    "department": {"type": "choice", "instructions": "转给哪个团队？",
                   "criteria": {"logistics": "物流问题", "refund": "退款诉求", "complaint": "投诉维权"}},
    "refund_intent": {"type": "noul", "instructions": "是否要求退款？"},
    "severity": {"type": "score", "instructions": "严重程度？",
                 "criteria": ["轻微", "一般", "严重", "紧急"]}
  },
  "abstain_threshold": 0.7
}
```

响应：每题 `{type, answer, probs/prob, abstained}`；score 额外给 `expected`（期望等级）。
`abstain_threshold` 之下 answer=null + abstained=true —— **拒答是一等公民**。

## 4. 命名与品牌

- 名称：**Jue（决）**。决 = 决策、判断；与 Jev 品类呼应但独立，避开 Jev 商标。
- 发布前检查 GitHub/HF/PyPI 名称冲突与商标风险；备选：决明（Jueming）。
- Slogan：`让每个概率都敢用。` / `Probabilities you can act on.`

## 5. 成功指标

| 阶段 | 指标 | 目标 |
|---|---|---|
| Phase 0 | 零训练基线跑通 + 评测报告 | 92 题全量出数（本报告） |
| Phase 1 | 中文任务准确率 | 微调 4B ≥ 零训练 qwen-max 基线 + 5pp |
| Phase 1 | 校准 | ECE ≤ 0.05（温度缩放后，holdout） |
| Phase 1 | 端侧 | 4B-Q4 在 16GB 无独显笔记本 CPU 上单题 < 3s |
| Phase 2 | 生态 | HF 下载 1k+/月；评测集被 ≥ 3 个外部项目引用 |

## 6. 路线图

- **Phase 0（本周期）**：schema/引擎/校准/评测 harness + 零训练 API 基线 + mini 评测集。✅ 开发中
- **Phase 1（1–2 周）**：合成 1–10 万条中文决策数据 → 租卡训 0.6B/4B LoRA + pointer head →
  温度缩放/拒答 → HF 权重 + GGUF → 本地 CPU 实测。
- **Phase 2**：评测集扩到 2000+ 题（含对抗/反讽/分布外子集）、OpenAI 兼容 server 完善、
  SDK 文档、上架 HF/GitHub、登记 jev-decision-index、社区运营。
- **Phase 3（可选）**：垂直领域包（客服/审核/风控）、多模态 state、蒸馏到 0.3B 级。

## 7. 竞争与风险

| 风险 | 等级 | 对策 |
|---|---|---|
| 赛道同质化/迭代快 | 高 | 只做中文+校准+端侧组合，不做通用复刻；评测集建立引用壁垒 |
| 大厂与百炼免费挤压 | 中 | 开源价值在私有化/可微调/可审计，不与免费 API 拼价格 |
| 校准数据难获得 | 高 | 合成数据带软标签 + 人工抽检；评测集含真实分布子集 |
| 小模型能力上限 | 中 | 领域聚焦（不做开放域判断）；Go/No-Go 用数字说话 |
| 名称/商标冲突 | 低 | 发布前检索；Apache-2.0 复用 Kev/SemIf 代码保留声明 |

## 8. Go/No-Go 决策点（Phase 0 结束时）

- 若零训练大模型基线在中文任务上已很强（acc>90% 且 ECE 可控）→ 训练增量空间小，
  重心转向**端侧蒸馏 + 评测集**（把大模型能力蒸到 CPU 可跑的小模型）。
- 若基线明显偏弱或过度自信 → 按 Phase 1 训练路线投入。
- 判据以 `reports/phase0-*/report.md` 实测数字为准。

**实测结论（2026-11-20）**：零训练 qwen3.8-max 基线 acc 94.6% / ECE 0.047，触发第一条分支——
Phase 1 重心转向端侧蒸馏 + score 题型专项 + 评测集扩建，详见 docs/phase0-findings.md §5。
