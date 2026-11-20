# Jue（决）技术实现方案

> 版本 v0.1 · 2026-11-20 · 对应代码：`src/jue/`

## 1. 总体架构

```
                 ┌────────────────────────────────────────────┐
 state+questions │  schema.py   校验/归一化（choice/noul/score）│
                 └──────────────┬─────────────────────────────┘
                                ▼
                 ┌────────────────────────────────────────────┐
                 │  engine.py   逐题调用后端 → 概率 → 结构化答案 │
                 │              （argmax / 拒答 / expected）    │
                 └──────────────┬─────────────────────────────┘
                                ▼
        ┌───────────────┬───────────────┬────────────────┬──────────────┐
        │ openai_compat │ hf_local      │ systemone      │ mock         │
        │ 零训练 logits  │ 零训练 logits  │ 原生决策 API    │ 测试专用      │
        │ (API/vLLM/    │ (CPU/GPU,     │ (Jev/百炼,     │              │
        │  llama.cpp)   │  transformers)│  teacher/基线) │              │
        └───────────────┴───────────────┴────────────────┴──────────────┘
                                ▼
                 ┌────────────────────────────────────────────┐
                 │ calibration.py  温度缩放 / ECE / Brier / 拒答 │
                 │ eval.py         评测 harness + 报告           │
                 └────────────────────────────────────────────┘
```

**后端统一接口**：`option_probs(state, question) -> list[float]`（与选项顺序对齐、和为 1）。
引擎与后端解耦：换模型/换路线不改业务代码。

## 2. 推理路线

### 2.1 零训练 logits 读出（Phase 0 主力，SemIf 式）

1. 每题渲染为单选题：system 约束"只输出选项字母"，user = 业务状态 + 问题 + A/B/C 选项。
2. `max_tokens=1, temperature=0, logprobs=True`，读第一个生成 token 的 top_logprobs。
3. 字母变体合并（`A`/`a`/` A`/` a`），未上榜选项给地板值（min×0.01），归一化 → 选项概率。
4. noul → A.是/B.否；score → 字母按等级低→高排列，额外输出期望等级。

约束与对策：
- DashScope top_logprobs 上限 5 → Phase 0 单题选项 ≤5；>5 选项走分组两阶段（Phase 1）。
- 首 token 不是字母（模型不听话）→ 地板值兜底 + 均匀分布降级，评测中计为低置信。
- 每问一次前向，无共享前缀优化（Phase 1 在本地后端实现 KV 复用）。

### 2.2 训练路线（Phase 1，Kev/Bongard 式）

- 底座：Qwen3-0.6B / 1.7B / 4B（Apache-2.0）。
- 结构：LoRA 微调骨干 + **pointer head**（复用 lm_head 对选项字母 token 的打分，
  训练目标 = 选项字母上的交叉熵；等价于把"生成"约束为"指向"）。
- 共享前缀：state 编码一次，问题分支并行（参考 Jev 逆向架构：shared-prefix KV + 并行分支）。
  训练时按 state 分组打包；推理时 `past_key_values` 复用。
- 损失：交叉熵 + label smoothing(0.05)；校准敏感场景实验 focal loss。
- 算力：单卡 4090/A100（租用），4B LoRA 单轮数小时；预算每轮几十~几百元。
- 数据：见 §4。

### 2.3 端侧部署（Phase 1 交付）

- 导出 GGUF（Q4_K_M/Q5），llama.cpp server 提供 logprobs → 直接复用 openai_compat 后端。
- 备选 ONNX Runtime（int8）；0.6B 目标手机级延迟。
- 验收：16GB 无独显 Windows 笔记本，4B-Q4 单题 <3s，0.6B <500ms。
- 量化后必须重跑校准评测（量化会改变概率分布 → 重新拟合温度）。

## 3. 校准（核心差异化）

| 手段 | 阶段 | 说明 |
|---|---|---|
| 温度缩放 | Phase 0 起 | 单参数 T，最小化 NLL；calib/holdout 按 id 哈希 50/50 切分，防泄漏 |
| 小样本收缩 | Phase 0 起 | T_final = T_fit^(n/(n+n0))，n0=200 伪计数；防小样本高置信错误主导 NLL（实测 holdout ECE 0.081→0.028） |
| label smoothing / focal | Phase 1 训练期 | 抑制过度自信 |
| 分组温度 | Phase 2 | 按题型/领域分别拟合（noul 与 score 的置信度分布不同） |
| 拒答 | Phase 0 起 | max prob < τ → answer=null；报告 coverage/selective-accuracy 曲线 |
| OOD 检测 | Phase 2 | 选项概率熵 + 状态困惑度双信号，分布外整体降置信 |

指标：Accuracy、ECE(10 bins)、Brier、NLL、平均置信度、拒答曲线。
**发布规则：任何权重发布必须附带 holdout ECE/Brier，不允许只报准确率。**

## 4. 数据管线（Phase 1）

1. **合成**：大模型按领域模板生成 (state, questions, 硬标签+软标签)；
   软标签 = 多次采样投票分布，保留歧义样本的真实不确定性（校准训练的关键）。
2. **领域模板**：客服分流/电商售后/内容审核/金融风控/Agent 路由（与评测集同构）。
3. **难例注入**：反讽、隐含意图、混合意图、边界案例、分布外样本（目标占比 20%）。
4. **质检**：schema 校验 + 标签可映射校验（已有测试）+ 大模型交叉复审 + 人工抽检 5%。
5. **防污染**：评测集 state 与训练合成种子隔离；评测集哈希清单随仓库发布。
6. 规模：Phase 1 目标 1–10 万题；`scripts/synth_data.py`（Phase 1 实现）。

## 5. 评测协议

- 数据集：JSONL，`{id, domain, state, questions:{qid:{...,label}}}`；标签映射统一走 `label_index`。
- 切分：`sha256(id|qid)` 哈希 → calib_frac 比例进校准集，其余为 holdout；确定性、跨运行一致。
- 报告：`report.json`（机器读）+ `report.md`（人读）+ `records.jsonl`（逐题概率，可复算）。
- 对照：同一评测集跑 Jue 各权重 / 零训练基线 / SystemOne teacher（Jev、百炼）。
- Phase 0 评测集：`data/eval/zh-decision-mini.jsonl`，40 states / 92 questions / 5 领域，
  含反讽（cs-002）、条件性退款（cs-001）、差评≠辱骂（mod-001）、常识≠诊疗建议（mod-004）、
  最低还款≠高风险（risk-006）等边界用例。

## 6. 服务与接口

- `jue serve`：FastAPI，`POST /v1/decisions` + `GET /healthz`；后端可插拔。
- 兼容策略：请求/响应字段与 Jev 对齐（state/questions/answers），迁移成本低。
- 安全：默认绑定 127.0.0.1；API key 只从环境/配置读取，不落仓库。

## 7. 仓库结构

```
jue/
├── docs/                  产品设计方案 / 技术实现方案（本文件）
├── src/jue/
│   ├── schema.py          题型校验、选项渲染、标签映射、答案格式化
│   ├── prompt.py          单选题 prompt + 首token概率解析（API/HF 两条）
│   ├── engine.py          决策编排（拒答在此生效）
│   ├── calibration.py     温度缩放/ECE/Brier/NLL/拒答曲线
│   ├── eval.py            评测 harness + JSON/Markdown 报告
│   ├── cli.py             jue decide / eval / serve
│   ├── server.py          FastAPI /v1/decisions
│   └── backends/          base / mock / openai_compat / systemone / hf_local
├── data/eval/             zh-decision-mini.jsonl
├── tests/                 18 项单测（schema/校准/prompt/端到端/数据集完整性）
├── scripts/               probe_api.py（端点探测）；synth_data.py（Phase 1）
└── reports/               评测产物（gitignore）
```

## 8. 当前状态与已知限制（Phase 0）

- ✅ Phase 0 完成：全链路代码 + 23 项测试通过 + 92 题实测（acc 94.6% / ECE 0.047，
  见 docs/phase0-findings.md）；读出协议加固（enable_thinking=False / 重试 / 全角归一化 / 降级兜底）。
- ✅ Token Plan 端点验证：qwen3.8-max 支持 logprobs（top≤5），零训练路线可用。
- ⏳ hf_local 后端已实现，本机无 torch，未做实机验证（`pip install jue[hf]` 后可跑）。
- ⏳ systemone 后端响应解析为防御性实现，待有 workspace 凭据后实测。
- ⚠️ 选项数上限 26（字母方案）；>5 选项在 top_logprobs=5 的 API 上概率不全（地板值兜底）。
- ⚠️ 温度缩放为单参数全局拟合 + 小样本收缩；Phase 0 样本量小，holdout 数字波动大，只作方向性结论。
- ✅ 评估修复轮（2026-11-20）：7 项修复（温度收缩/URL 归一化/CLI 后端参数/extra_body 自动检测/
  输入校验/HF 空候选守卫/UTF-8 输出），34 项测试通过，收缩校准经离线对比实证。
