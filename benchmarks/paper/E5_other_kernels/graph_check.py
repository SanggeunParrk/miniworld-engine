"""CUDA-graph validity check of every (backend, case) of E5.  The measurer times ``graph.replay()`` of a step captured on a side stream; a library whose kernels launch on the
legacy default stream (or otherwise leave the capture) is then timed on an EMPTY graph. This check runs the same step (same module, inputs, backend, train / eval) and verifies:

  (a) the output of two graph replays equals the eager output (eval: relative rms; train: output norm ratio, the module's dropout is random);
  (b) train: the parameter gradients after a replay are non-zero and of the eager norm (a parameter with a non-zero eager gradient and a zero / missing graph gradient fails);
  (c) the replay time is of the size of the eager GPU kernel sum (replay < 0.5 x kernel sum: suspect);
  (d) torch.profiler: the largest eager kernels (>= 10 % of the eager kernel time, up to 3) appear in the replay's kernel list.

A failed check -> verdict ``invalid_graph`` with the reasons; the record also holds the eager timing (CUDA events, same L2 flush / warmup / iterations as the measurer's ``eager`` mode) for
every checked cell: for an invalid cell it is the corrected time (``collect.py`` excludes the graph time and uses it, marked eager).

    python graph_check.py cfg_e5_all.yaml --out records/<card>/graphcheck.r1.jsonl --ext <names> [--module ...] [--backend ...] [--budget-seconds N]

Resumable (cases already in --out are skipped; <out>.done at the end). pytorch is not checked (plain torch ops).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch
from paper_common import cfg as cfgmod
from paper_common import csvio, provenance


def _floats(tree):
    from torch.utils import _pytree

    return [t for t in _pytree.tree_leaves(tree) if isinstance(t, torch.Tensor) and t.is_floating_point()]


def _kernels(fn, n: int = 1) -> dict[str, float]:
    """{kernel name: total device ms} of ``n`` calls of ``fn`` (all CUDA activity: every stream, legacy default included)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    out: dict[str, float] = {}
    for e in prof.events():
        if str(getattr(e, "device_type", "")).endswith("CUDA"):
            t = getattr(e, "device_time", None)
            if t is None:
                t = getattr(e, "cuda_time", 0.0)
            out[e.name] = out.get(e.name, 0.0) + float(t) / 1e3 / n
    return out


def _samp(t: torch.Tensor) -> torch.Tensor:
    """A strided sample of at most ~4M elements (the checks compare norms / relative errors; a full copy of a [1, L, L, d] output does not fit next to the graph pool on a 32 GB card)."""
    f = t.detach().reshape(-1)
    return f[:: max(1, f.numel() // (1 << 22))].clone()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30))


def check(step, module, timer, *, train: bool, warmup: int, iterations: int) -> dict:
    from miniworld_engine.measure import speed

    rec: dict = {}
    reasons: list[str] = []
    side = torch.cuda.Stream(timer.device)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    # capture exactly as the measurer does (the first backward must have run on the side stream: autograd nodes remember their stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        ret_g = step()
    torch.cuda.synchronize()
    g_outs = _floats(ret_g)

    def snapshot():
        return [_samp(t) for t in g_outs], (
            {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None} if train else {}
        )

    g.replay()
    torch.cuda.synchronize()
    o1, gr1 = snapshot()
    g.replay()
    torch.cuda.synchronize()
    o2, _ = snapshot()
    # eager reference on the CURRENT stream (a library that launches on the legacy default stream races with a non-blocking side stream)
    ret = step()
    torch.cuda.synchronize()
    e_out = [_samp(t) for t in _floats(ret)]
    e_grad = {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None} if train else {}
    ek = _kernels(step)
    e_sum = sum(ek.values())
    rec["eager_kernel_ms"] = e_sum
    rec["eager"] = speed.measure(step, timer, timing="eager", warmup=warmup, iterations=iterations)["median_ms"]
    # (a)
    if len(o1) != len(e_out):
        reasons.append(f"(a) {len(o1)} graph outputs vs {len(e_out)} eager outputs")
    else:
        if not train:
            err = max(_rel(a, b) for a, b in zip(o1, e_out, strict=True))
            rec["out_rel_vs_eager"] = err
            if not err < 5e-2:
                reasons.append(f"(a) graph output differs from eager: relative rms {err:.2e}")
        else:
            ratios = [float(a.double().norm() / b.double().norm().clamp_min(1e-30)) for a, b in zip(o1, e_out, strict=True)]
            rec["out_norm_ratio"] = min(ratios), max(ratios)
            if not all(0.8 < r < 1.25 for r in ratios):
                reasons.append(f"(a) graph output norm / eager output norm = {min(ratios):.3f}..{max(ratios):.3f}")
        rr = max(_rel(a, b) for a, b in zip(o2, o1, strict=True)) if not train else 0.0
        rec["replay2_rel_vs_replay1"] = rr
        if not rr < 1e-2:
            reasons.append(f"(a) the second replay differs from the first: relative rms {rr:.2e}")
        if any(not torch.isfinite(t).all() for t in o1):
            reasons.append("(a) non-finite graph output")
    # (b)
    if train:
        bad, n_nz = [], 0
        for n, ge in e_grad.items():
            ne = float(ge.double().norm())
            if ne == 0.0:
                continue
            n_nz += 1
            gg = gr1.get(n)
            ng = 0.0 if gg is None else float(gg.double().norm())
            if ng == 0.0 or not 0.5 < ng / ne < 2.0:
                bad.append(f"{n} {ng:.2e}/{ne:.2e}")
        rec["grad_params_nonzero_eager"] = n_nz
        rec["grad_params_bad"] = len(bad)
        if n_nz == 0:
            reasons.append("(b) eager gradients are all zero (nothing to compare)")
        if bad:
            reasons.append(f"(b) {len(bad)} of {n_nz} parameter gradients zero / off in the graph vs eager (e.g. {'; '.join(bad[:3])})")
    # (c) replay time vs eager kernel sum
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t_graph = speed._stats(timer.run(g.replay, max(5, iterations // 2)))["median_ms"]
    rec["graph"] = t_graph
    rec["graph_over_kernel_sum"] = t_graph / e_sum if e_sum > 0 else None
    if e_sum > 0 and t_graph < 0.5 * e_sum:
        reasons.append(f"(c) graph replay {t_graph:.3f} ms is below half of the eager GPU kernel time {e_sum:.3f} ms")
    # (d)
    gk = _kernels(g.replay)
    top = [k for k, v in sorted(ek.items(), key=lambda kv: -kv[1]) if v >= 0.1 * e_sum][:3]
    miss = [k[:60] for k in top if k not in gk]
    rec["replay_kernel_ms"] = sum(gk.values())
    if miss:
        reasons.append(f"(d) largest eager kernel(s) absent from the graph replay: {'; '.join(miss)}")
    del g
    # (d) is advisory: CUPTI does not list every kernel of a replayed graph (the engine's own backward kernels are sometimes missing from the replay profile while the replay
    # time equals the eager time and the gradients match); the verdict rests on (a), (b), (c)
    rec["verdict"] = "invalid_graph" if any(not r.startswith("(d)") for r in reasons) else "valid"
    rec["reasons"] = reasons
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--ext", nargs="*", default=[])
    p.add_argument("--budget-seconds", type=float, default=0.0)
    p.add_argument(
        "--redo-unsupported",
        action="store_true",
        help="repeat the cells an earlier pass recorded unsupported (a library tree that was not installed then)",
    )
    cfgmod.add_selection_args(p)
    args = p.parse_args(argv)
    out = Path(args.out)
    done = Path(f"{out}.done")
    if done.exists():
        print(f"[check] {done} exists")
        return 0
    from miniworld_engine.measure import config, speed
    from miniworld_engine.modules.primitives import BackendUnsupported, unpadded_masks

    t0 = time.time()
    cfg = cfgmod.load(args.config, args)
    if args.ext:
        import externals

        externals.install(args.ext)
    dev = torch.device("cuda")
    sp = config.speed_settings(cfg)
    warmup, iterations = int(sp.get("warmup", 5)), int(sp.get("iterations", 20))
    timer = speed._Timer(dev, bool(sp.get("flush_l2", True)))
    # done = a real record (any graphcheck*.jsonl of the card, so a later pass only repeats what failed) or two placeholders of a check the process died in
    have, died = set(), {}
    for f in sorted(out.parent.glob("graphcheck*.jsonl")):
        for r in csvio.read_jsonl(str(f)):
            k = (r["row"], json.dumps(r["kwargs"], sort_keys=True), r["length"], r["dtype"], r["mode"], r["backend"])
            if r.get("verdict") != "check_error" and not (args.redo_unsupported and r.get("verdict") == "unsupported"):
                have.add(k)
            elif str(r.get("reason", "")).startswith("the process died"):
                died[k] = died.get(k, 0) + 1
    have |= {k for k, n in died.items() if n >= 3}
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.meta.json").write_text(
        json.dumps(provenance.describe() | {"config": str(Path(args.config).resolve())}, indent=1, default=str)
    )
    stopped = False
    with open(out, "a") as sink:
        for case in cfgmod.limit(cfg, args.max_cases):
            todo = [
                b
                for b in config.backends_of(cfg, case)
                if b != "pytorch" and (case.row, json.dumps(case.kwargs, sort_keys=True), case.length, case.dtype, case.mode, b) not in have
            ]
            if not todo:
                continue
            if args.budget_seconds and time.time() - t0 > args.budget_seconds:
                stopped = True
                break
            base = config.build(case, device=dev)
            for backend in todo:
                rec = {**config.describe(case, cfg), "backend": backend}
                # placeholder first: if the process dies in the check (a segfault), the next call skips this (backend, case) and the record says so; the real record follows it
                sink.write(
                    json.dumps(
                        {**rec, "status": "failed", "verdict": "check_error", "reason": "the process died during the check (crash)"},
                        default=str,
                    )
                    + "\n"
                )
                sink.flush()
                module = inputs = step = None
                try:
                    if backend not in type(base).backends:
                        raise BackendUnsupported(f"{type(base).__name__} has no backend {backend!r}")
                    torch._dynamo.reset()
                    module = copy.deepcopy(base)
                    with unpadded_masks():
                        inputs = config.inputs(case, module, device=dev)
                    step = speed.make_step(
                        module,
                        inputs,
                        backend=backend,
                        train=case.mode == "train",
                        autocast=config.autocast(case),
                        seed=config.seed(case),
                        compile_mode=None,
                        keep_expanded=config.keeps_expanded(case),
                    )
                    rec.update(
                        status="checked", **check(step, module, timer, train=case.mode == "train", warmup=warmup, iterations=iterations)
                    )
                except BackendUnsupported as exc:
                    rec.update(status="unsupported", verdict="unsupported", reason=str(exc))
                except Exception as exc:
                    rec.update(
                        status="skipped" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed",
                        verdict="check_error",
                        reason=f"{type(exc).__name__}: {str(exc)[:400]}",
                    )
                    traceback.print_exc()
                finally:
                    step = module = inputs = None
                    torch._dynamo.reset()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                sink.write(json.dumps(rec, default=str) + "\n")
                sink.flush()
                print(
                    f"[check] {rec['label']} | {backend:12s} | {rec.get('verdict')} | graph {rec.get('graph', float('nan')):.3f} eager {rec.get('eager', float('nan')):.3f} kern {rec.get('eager_kernel_ms', float('nan')):.3f} | {'; '.join(rec.get('reasons', [])) or rec.get('reason', '')}"[
                        :400
                    ],
                    flush=True,
                )
            del base
            torch.cuda.empty_cache()
    if not stopped:
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
