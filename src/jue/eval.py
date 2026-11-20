"""评测 harness：跑数据集 → 原始指标 → 温度缩放校准 → 校准后指标 → 拒答曲线。

数据集格式（JSONL）：
{"id","domain","state","questions":{qid:{type,instructions,criteria,label}}}
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

from .backends.base import DecisionBackend
from .calibration import abstention_curve, apply_temperature, fit_temperature, summarize
from .schema import label_index, question_options, validate_questions


def load_dataset(path: str | pathlib.Path) -> list[dict]:
    items = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _in_calib_split(item_id: str, qid: str, frac: float) -> bool:
    h = int(hashlib.sha256(f"{item_id}|{qid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def run_eval(backend: DecisionBackend, items: list[dict], calib_frac: float = 0.5,
             progress: bool = True) -> dict:
    records, errors = [], []
    t0 = time.time()
    for item in items:
        qs = validate_questions({k: {kk: vv for kk, vv in q.items() if kk != "label"}
                                 for k, q in item["questions"].items()})
        for qid, q in qs.items():
            raw_q = item["questions"][qid]
            try:
                probs = backend.option_probs(item["state"], q)
                y = label_index(q, raw_q["label"])
                best = max(range(len(probs)), key=probs.__getitem__)
                records.append({
                    "id": item["id"], "qid": qid, "domain": item.get("domain", "?"),
                    "type": q["type"], "probs": probs, "label": y, "correct": best == y,
                    "calib_split": _in_calib_split(item["id"], qid, calib_frac),
                })
            except Exception as e:
                errors.append({"id": item["id"], "qid": qid, "error": str(e)[:200]})
            if progress and (len(records) + len(errors)) % 10 == 0:
                print(f"  ... {len(records) + len(errors)} questions done", flush=True)
    elapsed = time.time() - t0
    return build_report(records, errors, len(items), elapsed, backend.name,
                        getattr(backend, "model", None), calib_frac)


def build_report(records: list[dict], errors: list[dict], n_items: int,
                 elapsed: float, backend_name: str, model, calib_frac: float = 0.5) -> dict:
    """从逐题记录构建报告（可离线复算，见 scripts/rebuild_report.py）。"""
    raw = summarize(records)
    fit_recs = [(r["probs"], r["label"]) for r in records if r["calib_split"]]
    eval_recs = [r for r in records if not r["calib_split"]]
    temperature = fit_temperature(fit_recs) if fit_recs else 1.0
    calibrated = []
    for r in records:
        p = apply_temperature(r["probs"], temperature)
        best = max(range(len(p)), key=p.__getitem__)
        calibrated.append({**r, "probs": p, "correct": best == r["label"]})

    report = {
        "backend": backend_name,
        "model": model,
        "n_items": n_items, "n_questions": len(records), "n_errors": len(errors),
        "elapsed_sec": round(elapsed, 1),
        "temperature": round(temperature, 4),
        "calib_frac": calib_frac,
        "raw": raw,
        "calibrated": summarize(calibrated),
        "calibrated_holdout": summarize([r for r in calibrated if not r["calib_split"]]),
        "abstention_raw": abstention_curve(records),
        "abstention_calibrated": abstention_curve(calibrated),
        "errors": errors,
    }
    return {"report": report, "records": records, "calibrated_records": calibrated}


def save_results(result: dict, out_dir: str | pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in result["calibrated_records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "records_raw.jsonl").open("w", encoding="utf-8") as f:
        for r in result["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(render_markdown(result["report"]), encoding="utf-8")
    return out


def render_markdown(rep: dict) -> str:
    raw, cal = rep["raw"], rep["calibrated"]
    lines = [
        f"# Jue 评测报告",
        "",
        f"- 后端: `{rep['backend']}` / 模型: `{rep.get('model')}`",
        f"- 样本: {rep['n_items']} states / {rep['n_questions']} questions（失败 {rep['n_errors']}）",
        f"- 耗时: {rep['elapsed_sec']}s，拟合温度 T = {rep['temperature']}（calib_frac={rep['calib_frac']}）",
        "",
        "## 总体指标",
        "",
        "| 指标 | 原始 | 温度缩放后 |",
        "|---|---|---|",
    ]
    for k, label in (("accuracy", "准确率"), ("ece", "ECE↓"), ("brier", "Brier↓"), ("nll", "NLL↓"), ("mean_confidence", "平均置信度")):
        lines.append(f"| {label} | {raw.get(k, 0):.4f} | {cal.get(k, 0):.4f} |")
    if "score_adjacent_accuracy" in raw:
        lines.append(f"| score相邻等级准确率 | {raw['score_adjacent_accuracy']:.4f} | {cal.get('score_adjacent_accuracy', 0):.4f} |")
    lines += ["", "## 分领域准确率（校准后）", "", "| 领域 | n | 准确率 |", "|---|---|---|"]
    for g, s in (cal.get("by_domain") or {}).items():
        lines.append(f"| {g} | {s['n']} | {s['accuracy']:.4f} |")
    lines += ["", "## 分题型准确率（校准后）", "", "| 题型 | n | 准确率 |", "|---|---|---|"]
    for g, s in (cal.get("by_type") or {}).items():
        lines.append(f"| {g} | {s['n']} | {s['accuracy']:.4f} |")
    lines += ["", "## 拒答曲线（校准后）", "", "| 阈值 | 覆盖率 | 保留部分准确率 | 拒答数 |", "|---|---|---|---|"]
    for row in rep["abstention_calibrated"]:
        sa = row["selective_accuracy"]
        lines.append(f"| {row['threshold']} | {row['coverage']:.3f} | {sa:.4f} | {row['abstained']} |" if sa is not None
                     else f"| {row['threshold']} | {row['coverage']:.3f} | - | {row['abstained']} |")
    if rep["errors"]:
        lines += ["", "## 错误", ""] + [f"- {e['id']}/{e['qid']}: {e['error']}" for e in rep["errors"][:20]]
    return "\n".join(lines) + "\n"
