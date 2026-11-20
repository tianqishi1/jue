# 贡献指南

感谢关注 Jue（决）！本项目目标：**中文优先、校准透明、端侧可跑的开源决策模型**。

## 开发环境

```bash
git clone https://github.com/tianqishi1/jue.git
cd jue
pip install -e ".[dev,server]"   # 核心零依赖；server  extras 含 fastapi/uvicorn
pytest -q                        # 34 项测试全部离线可跑，不需要任何 API key
```

可选：`pip install -e ".[hf]"` 安装 torch/transformers 以使用本地 CPU 后端。

## 我可以贡献什么

| 方向 | 入口 | 说明 |
|---|---|---|
| **评测样本**（最欢迎） | `data/eval/zh-decision-mini.jsonl` | 中文业务判断难例：反讽、隐含意图、边界案例。用 issue 模板「评测样本」提交 |
| 新后端 | `src/jue/backends/` | 实现 `option_probs(state, question) -> list[float]` 即可接入引擎/评测/服务 |
| 校准方法 | `src/jue/calibration.py` | 分组温度、ECE 目标拟合、OOD 拒答等，附离线对比实验 |
| 文档/翻译 | `docs/`、`README.md` | 中文优先，英文欢迎 |

## 评测样本规范

JSONL，每行一个 state：

```json
{"id":"cs-009","domain":"customer_service","state":"业务状态文本","questions":{"qid":{"type":"choice|noul|score","instructions":"问题","criteria":{},"label":"标签"}}}
```

- `choice.criteria` 为 {选项: 描述}（2-26 项）；`score.criteria` 为低→高有序列表；`noul` 可选 true/false 描述
- 标签必须可被 `jue.schema.label_index` 映射（提交前跑 `pytest tests/test_eval.py` 会自动校验全量数据集）
- **标签要有可辩护的唯一最佳答案**；真正模糊的样本请标注争议点，我们计划引入软标签
- 领域：customer_service / ecommerce / moderation / risk / agent_routing，新领域请先开 issue 讨论
- 禁止提交真实用户隐私数据；合成或脱敏样本

## 提交规范

- 分支 + PR，一个 PR 做一件事；commit message 用中文或英文均可，说清「为什么」
- 新功能必须带测试；修 bug 必须带回归测试（参考 `tests/test_fixes.py`）
- 涉及概率/校准的改动，附 `jue eval` 或 `scripts/rebuild_report.py` 的前后对比数字
- **不要提交任何密钥**；API 凭据只走环境变量或 `~/.bailian/config.json`
- 评测报告产物在 `reports/`（gitignore）；有发布价值的报告精选进 `docs/benchmarks/`

## 许可证

Apache-2.0。提交贡献即表示同意以 Apache-2.0 授权。复用 Kev/SemIf 等兼容许可代码时保留原声明。
