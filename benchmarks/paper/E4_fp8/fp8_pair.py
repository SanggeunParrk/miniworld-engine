"""E4: the default (bf16 / fp16) path against the FP8 path of TriMul and TriAttn inference, in one process, alternating.

    python fp8_pair.py --out records/<card>/fp8.<gates>.r<k>.jsonl [--gates shipped|none] [--model af3] [selection overrides]

Per case of the E2 grid (inference, ``master`` dtype: fp32 masters, bf16 activations, bf16 autocast), two copies of one module with the same
weights and inputs: ``cuda`` (the default route) and ``engine-fp8`` (``miniworld_engine.fp8``: the engine ``cuda`` backend with the ``safe``
preset's FP8 routes on; the TriAttn FP8 routes are validated per model, ``--model``, default ``af3``). The FP8 copy is calibrated first (two eager
calls: static scales are taken on a process's first call of each site) and ``fp8.check()`` is read after them; then ROUNDS rounds alternate
the two routes (order flips each round), each round one ``miniworld_engine.measure.speed.measure`` (CUDA-graph capture, replay, L2 flush,
warmup, median of ``iterations`` replays). ``fp8.check()`` is read again after the replays: True = some route saturated the e4m3 range
(``saturated``) and its scales were raised: the case is recorded, never re-routed. ``fp8.route_stats()`` says which FP8 routes fired
(``routes_fired``): a case below a route's length gate runs the default kernels in both copies (ratio ~ 1) and says so.

``--gates shipped``: the routes' production gates (what ``engine-fp8`` does for a user). ``--gates none``: every length / S gate removed
(MINIWORLD_TRIMUL_FP8_MINL=0 and the TriAttn gates of h100_sm90 cleared), as the FP8 measurements of the 3.0 tree did.
On a card without FP8 routes (not an H100) the FP8 copy runs the default kernels: ``routes_fired`` is False and the records say so.
Training is not measured: no FP8 backward exists, ``engine-fp8`` runs the default path in training.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROUNDS = 3


def _parse(argv):
    from paper_common import cfg as cfgmod

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "cfg_pair.yaml"))
    p.add_argument("--gates", choices=["shipped", "none"], default="shipped")
    p.add_argument(
        "--model", default="af3", help="the model whose TriAttn FP8 routes engine-fp8 runs (miniworld_engine.fp8.TRIATTN_MODELS)"
    )
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--budget-seconds", type=float, default=0.0)
    cfgmod.add_selection_args(p)
    return p.parse_args(argv)


def _steps(base, inputs, autocast):
    import contextlib

    import torch

    def mk(module):
        fwd = module._forward_cuda  # the instance's _forward_cuda carries engine-fp8 (what a runtime that calls _forward_<backend> gets)

        def step():
            with torch.no_grad(), torch.autocast("cuda", dtype=autocast) if autocast is not None else contextlib.nullcontext():
                return fwd(**inputs)

        return step

    default = copy.deepcopy(base).set_backend("cuda").eval()
    fp8 = copy.deepcopy(base).set_backend("engine-fp8").eval()
    return default, mk(default), fp8, mk(fp8)


def main(argv=None) -> int:
    args = _parse(argv)
    if args.gates == "none":
        os.environ["MINIWORLD_TRIMUL_FP8_MINL"] = "0"  # before the first import of the kernels
        os.environ["MINIWORLD_TRIATTN_B200_FP8_MINS"] = "0"  # the B200 TriAttn gate (a no-op on every other card)
        os.environ["MINIWORLD_TRIATTN_SM120_FP8_MINL"] = "0"  # the RTX 5090 e4m3 TriAttn gate (a no-op on every other card)
        os.environ["MINIWORLD_TRIATTN_SM120_FP8_PVONLY"] = "1"  # and its PV-only core (unfused kinds; off when shipped)
    import torch
    from paper_common import cfg as cfgmod
    from paper_common import provenance
    from paper_common.run_measure import case_key

    if not torch.cuda.is_available():
        print("[fp8] no CUDA device", file=sys.stderr)
        return 2
    from miniworld_engine import fp8
    from miniworld_engine.measure import config, speed
    from miniworld_engine.modules.primitives import unpadded_masks

    t0 = time.time()
    cfg = cfgmod.load(args.config, args)
    out = Path(args.out)
    done_marker = Path(f"{out}.done")
    if done_marker.exists():
        print(f"[fp8] {done_marker} exists: nothing to do")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = provenance.describe() | {"gates": args.gates, "model": args.model, "rounds": args.rounds, "config": args.config}
    Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    have = set()
    if out.exists():
        have = {case_key(json.loads(x)) for x in out.read_text().splitlines() if x.strip()}
    if args.gates == "none":
        try:
            from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

            hs.FP8_PV_MIN_S, hs.FP8_PV_FRONT_MIN_S, hs.FP8_PV_FUSED_MIN_S = {}, {}, 0
        except Exception as exc:  # recorded in the sidecar: the TriAttn gates could not be cleared on this card
            meta["triattn_gates_not_cleared"] = f"{type(exc).__name__}: {exc}"
            Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    fp8.set_model(args.model)
    dev = torch.device(cfg.device)
    speed_cfg = config.speed_settings(cfg)
    warmup, iterations = int(speed_cfg.get("warmup", 5)), int(speed_cfg.get("iterations", 100))
    timer = speed._Timer(dev, bool(speed_cfg.get("flush_l2", True)))
    torch.backends.cuda.matmul.allow_tf32 = True  # cuda is a TF32_BACKENDS member
    print(f"[fp8] {json.dumps(meta, default=str)}", flush=True)
    cases = [c for c in cfgmod.limit(cfg, args.max_cases) if case_key(c) not in have and c.mode == "eval"]
    finished = True
    with open(out, "a") as sink:
        for case in cases:
            if args.budget_seconds and time.time() - t0 > args.budget_seconds:
                finished = False
                print("[fp8] budget used up", flush=True)
                break
            rec = {**config.describe(case, cfg), "gates": args.gates, "model": args.model, "meta": {"env": config.environment()}}
            try:
                base = config.build(case, device=dev)
                with unpadded_masks():
                    inputs = config.inputs(case, base, device=dev)
                default, step_d, _f8mod, step_f = _steps(base, inputs, config.autocast(case))
                with (
                    torch.no_grad(),
                    (
                        torch.autocast("cuda", dtype=config.autocast(case))
                        if config.autocast(case)
                        else __import__("contextlib").nullcontext()
                    ),
                ):
                    why = default.unsupported(**inputs)
                if why is not None:
                    raise RuntimeError(f"unsupported: {why}")
                fp8.reset_route_stats()
                for _ in range(2):  # calibration (static scales on the first eager call of each site) and a warm call
                    step_f()
                torch.cuda.synchronize(dev)
                sat0 = fp8.check()
                eager_stats = fp8.route_stats()
                ms = {"cuda": [], "engine-fp8": []}
                for r in range(args.rounds):
                    for name, step in (
                        (("cuda", step_d), ("engine-fp8", step_f)) if r % 2 == 0 else (("engine-fp8", step_f), ("cuda", step_d))
                    ):
                        torch._dynamo.reset()
                        ms[name].append(speed.measure(step, timer, timing="graph", warmup=warmup, iterations=iterations)["median_ms"])
                sat1 = fp8.check()
                fired = {
                    "trimul": eager_stats["trimul"]["calls"] > 0,
                    "triattn": any(v > 0 for k, v in eager_stats["triattn"].items() if k.startswith("pv") or k == "qk"),
                }
                rec.update(
                    status="ok",
                    ms=dict(ms),
                    ratio_default_over_fp8=statistics.median(ms["cuda"]) / statistics.median(ms["engine-fp8"]),
                    saturated_calibration=sat0,
                    saturated_replay=sat1,
                    routes_fired=fired,
                    fp8_route_fired=fired["trimul"] if case.module == "TriangleMultiplication" else fired["triattn"],
                    route_stats={k: v for k, v in eager_stats.items() if k in ("trimul", "triattn")},
                )
            except Exception as exc:  # recorded, never re-routed
                oom = isinstance(exc, torch.cuda.OutOfMemoryError)
                rec.update(status="skipped" if oom else "failed", reason=f"{type(exc).__name__}: {str(exc).strip()[:500]}")
            finally:
                base = default = inputs = None
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()
            sink.write(json.dumps(rec, default=str) + "\n")
            sink.flush()
            print(
                f"{rec['label']} | gates={args.gates} | "
                + (
                    f"default {statistics.median(rec['ms']['cuda']):.4f} ms  fp8 {statistics.median(rec['ms']['engine-fp8']):.4f} ms  "
                    f"x{rec['ratio_default_over_fp8']:.3f}  fired={rec['fp8_route_fired']} saturated={rec['saturated_calibration'] or rec['saturated_replay']}"
                    if rec["status"] == "ok"
                    else f"{rec['status']}: {rec.get('reason', '')[:200]}"
                ),
                flush=True,
            )
    if finished:
        done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
