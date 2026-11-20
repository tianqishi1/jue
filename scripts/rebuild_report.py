"""从已保存的 records_raw.jsonl 离线重建评测报告（不消耗 API）。

用法: python scripts/rebuild_report.py reports/phase0-qwen3.8-max-v2
"""
import json
import pathlib
import sys

from jue.eval import build_report, save_results


def main():
    run_dir = pathlib.Path(sys.argv[1])
    records = [json.loads(l) for l in (run_dir / "records_raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    result = build_report(records, meta.get("errors", []), meta["n_items"], meta["elapsed_sec"],
                          meta["backend"], meta.get("model"), meta.get("calib_frac", 0.5))
    save_results(result, run_dir)
    rep = result["report"]
    print(f"rebuilt: acc={rep['raw']['accuracy']:.4f} ece={rep['raw']['ece']:.4f} "
          f"T={rep['temperature']} score_adj={rep['raw'].get('score_adjacent_accuracy')}")


if __name__ == "__main__":
    main()
