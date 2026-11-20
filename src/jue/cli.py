"""Jue CLI：jue decide / jue eval / jue serve"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _backend_kwargs(args) -> dict:
    backend = getattr(args, "backend", "openai")
    if backend == "hf":
        name = getattr(args, "model_name", None) or getattr(args, "model", None)
        return {"model_name": name} if name else {}
    kw = {}
    if getattr(args, "base_url", None):
        kw["base_url"] = args.base_url
    if getattr(args, "api_key", None):
        kw["api_key"] = args.api_key
    if getattr(args, "model", None):
        kw["model"] = args.model
    return kw


def cmd_decide(args):
    from .backends import make_backend
    from .engine import run_decision

    state = args.state
    if args.state_file:
        state = open(args.state_file, encoding="utf-8").read()
    if state is None:
        raise SystemExit("错误：需要 --state 或 --state-file")
    if not args.questions and not args.questions_file:
        raise SystemExit("错误：需要 --questions 或 --questions-file")
    questions = json.loads(args.questions) if args.questions else json.load(open(args.questions_file, encoding="utf-8"))
    backend = make_backend(args.backend, **_backend_kwargs(args))
    answers = run_decision(backend, state, questions, abstain_threshold=args.abstain)
    print(json.dumps({"model": getattr(backend, "model", backend.name), "answers": answers}, ensure_ascii=False, indent=2))


def cmd_eval(args):
    from .backends import make_backend
    from .eval import load_dataset, run_eval, save_results

    items = load_dataset(args.data)
    if args.limit:
        items = items[: args.limit]
    backend = make_backend(args.backend, **_backend_kwargs(args))
    print(f"评测 {len(items)} 条 state，后端={args.backend} ...", flush=True)
    result = run_eval(backend, items, calib_frac=args.calib_frac, shrink_n0=args.shrink_n0)
    out = save_results(result, args.out)
    rep = result["report"]
    print(f"完成: acc={rep['raw']['accuracy']:.4f} ece={rep['raw']['ece']:.4f} "
          f"→ 校准后 acc={rep['calibrated']['accuracy']:.4f} ece={rep['calibrated']['ece']:.4f} (T={rep['temperature']})")
    print(f"报告: {out}")


def cmd_serve(args):
    import uvicorn

    from .backends import make_backend
    from .server import create_app

    backend = make_backend(args.backend, **_backend_kwargs(args))
    app = create_app(backend, abstain_threshold=args.abstain)
    uvicorn.run(app, host=args.host, port=args.port)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows GBK 控制台中文乱码
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="jue", description="Jue (决) — 开源中文决策模型")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_backend_args(sp, default="openai"):
        sp.add_argument("--backend", default=default, choices=["mock", "openai", "systemone", "hf"])
        sp.add_argument("--base-url")
        sp.add_argument("--api-key")
        sp.add_argument("--model")
        sp.add_argument("--model-name", help="hf 后端的 HF 模型名")
        sp.add_argument("--abstain", type=float, default=None, help="拒答阈值（max prob 低于此值则拒答）")

    d = sub.add_parser("decide", help="对一条 state 做决策")
    d.add_argument("--state")
    d.add_argument("--state-file")
    d.add_argument("--questions")
    d.add_argument("--questions-file")
    add_backend_args(d)
    d.set_defaults(fn=cmd_decide)

    e = sub.add_parser("eval", help="跑评测集")
    e.add_argument("--data", required=True)
    e.add_argument("--out", default="reports/run")
    e.add_argument("--limit", type=int)
    e.add_argument("--calib-frac", type=float, default=0.5)
    e.add_argument("--shrink-n0", type=float, default=200.0, help="温度收缩伪计数（0=禁用收缩）")
    add_backend_args(e)
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("serve", help="起本地 OpenAI 风格决策服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    add_backend_args(s)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
