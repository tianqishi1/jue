import json
import pathlib

from jue.backends import make_backend
from jue.engine import run_decision
from jue.eval import load_dataset, run_eval, save_results, render_markdown

DATA = pathlib.Path(__file__).parent.parent / "data" / "eval" / "zh-decision-mini.jsonl"


def test_dataset_integrity():
    from jue.schema import label_index, validate_questions
    items = load_dataset(DATA)
    assert len(items) == 40
    n = 0
    for item in items:
        qs = validate_questions({k: {kk: vv for kk, vv in q.items() if kk != "label"}
                                 for k, q in item["questions"].items()})
        for qid, q in qs.items():
            label_index(q, item["questions"][qid]["label"])  # 标签必须可映射
            n += 1
    assert n == 92


def test_engine_with_mock():
    backend = make_backend("mock")
    answers = run_decision(backend, "测试状态", {
        "q1": {"type": "noul", "instructions": "是否？"},
        "q2": {"type": "choice", "instructions": "选哪个？", "criteria": {"x": "X", "y": "Y"}},
    })
    assert set(answers) == {"q1", "q2"}
    assert isinstance(answers["q1"]["answer"], bool)
    assert abs(sum(answers["q2"]["probs"].values()) - 1) < 1e-5


def test_engine_abstain():
    backend = make_backend("mock")
    answers = run_decision(backend, "s", {"q": {"type": "noul", "instructions": "是否？"}},
                           abstain_threshold=0.999)
    assert answers["q"]["abstained"] and answers["q"]["answer"] is None


def test_eval_end_to_end(tmp_path):
    items = load_dataset(DATA)[:4]
    result = run_eval(make_backend("mock"), items, progress=False)
    rep = result["report"]
    assert rep["n_questions"] == sum(len(i["questions"]) for i in items)
    assert rep["n_errors"] == 0
    assert 0 <= rep["raw"]["accuracy"] <= 1
    assert rep["temperature"] > 0
    assert "by_domain" in rep["calibrated"]
    out = save_results(result, tmp_path / "run")
    assert (out / "report.json").exists() and (out / "records.jsonl").exists()
    md = render_markdown(rep)
    assert "ECE" in md and "拒答曲线" in md
    json.loads((out / "report.json").read_text(encoding="utf-8"))
