"""The comparison table: D = 128 Transition (y = x + Ws(silu(Wa LN x) * Wb LN x)), bf16, B200, CUDA-graph replay median.

Inference = forward under no_grad.  Training = forward + backward (dx and all five parameter gradients).
Rows: PyTorch eager, torch.compile, Anthropic's own Transition rows (uplifting-biomolecular-modeling opt_core; forward-only -- the
provider ships no backward with parameter gradients), and the fused sm_100a kernels of this experiment.

  python bench_all.py --lengths 384 768 [--anthropic-root <refs/uplifting-biomolecular-modeling>] [--save records/table-vN.json]
"""
import argparse, json, os, sys, traceback
import torch
import torch.nn.functional as F
from common import D, H, make_inputs, fp32_fwd, rel, graph_time

p = argparse.ArgumentParser()
p.add_argument("--lengths", type=int, nargs="+", default=[384, 768])
p.add_argument("--anthropic-root", default=os.environ.get("ANTHROPIC_ROOT", "/NHNHOME/WORKSPACE/26mohw002_A/psk6950/refs/uplifting-biomolecular-modeling"))
p.add_argument("--anthropic-rows", nargs="+", default=["v2", "af3_fused", "pf", "lnl", "v1"])
p.add_argument("--no-compile", action="store_true")
p.add_argument("--save", default=None)
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False


def torch_transition(x, wa, wb, ws, g, b):
    xn = F.layer_norm(x, (D,), g, b, 1e-5)
    return x + F.linear(F.silu(F.linear(xn, wa)) * F.linear(xn, wb), ws)


def train_step_fn(fn, x, params, dy):
    """fwd + bwd closure; grads land in .grad (set_to_none before capture so replay rewrites the same buffers)."""
    def step():
        y = fn(x, *params)
        y.backward(dy)
    return step


results = {"device": torch.cuda.get_device_name(), "rows": {}}
sys.path.insert(0, os.path.join(a.anthropic_root, "common", "opt_core"))

for L in a.lengths:
    M = L * L
    x, wa, wb, ws, gamma, beta = make_inputs(L)
    ref = fp32_fwd(x, wa, wb, ws, gamma, beta)
    g16, b16 = gamma.to(torch.bfloat16), beta.to(torch.bfloat16)
    dy = torch.randn_like(x)

    def put(name, mode, us, err=None, note=None):
        results["rows"].setdefault(name, {})[f"{mode}_L{L}"] = {"us": us, "rel_vs_fp32": err, "note": note}
        print(f"L{L} {mode:9s} {name:34s} {('%.1f us' % us) if us else 'n/a':>10s}  {'' if err is None else 'rel %.2e' % err}  {note or ''}", flush=True)

    # ---------------- PyTorch eager / compile
    for name, fn in [("PyTorch eager", torch_transition)] + ([] if a.no_compile else [("torch.compile", torch.compile(torch_transition))]):
        try:
            with torch.no_grad():
                y = fn(x, wa, wb, ws, g16, b16)
                put(name, "inference", graph_time(lambda: fn(x, wa, wb, ws, g16, b16)), rel(y, ref))
            xl = x.clone().requires_grad_(True)
            prm = [t.clone().requires_grad_(True) for t in (wa, wb, ws, g16, b16)]
            step = train_step_fn(fn, xl, prm, dy)
            for _ in range(3):
                step()
            for t in [xl] + prm:
                t.grad = None
            put(name, "training", graph_time(step))
        except Exception as exc:  # noqa: BLE001
            put(name, "inference", None, note=repr(exc)[:100])

    # ---------------- Anthropic rows (forward only)
    try:
        from opt_core.kernels import transition as T
        W = T.pack(w_o=ws, w_a=wa, w_b=wb, ln_w=gamma, ln_b=beta, eps=1e-5, device=x.device)
        for row in a.anthropic_rows:
            name = f"Anthropic {row}"
            try:
                with torch.no_grad():
                    fn = lambda: T.transition(x, W, word=row, residual=True, n_tokens=L, timing="graph")[0]
                    y = fn()
                    put(name, "inference", graph_time(fn), rel(y, ref))
                put(name, "training", None, note="no backward in the Anthropic provider")
            except Exception as exc:  # noqa: BLE001
                put(name, "inference", None, note=repr(exc)[:120])
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    # ---------------- this experiment
    try:
        from fwd_op import FusedFwd
        f = FusedFwd()
        f.set_weights(wa, wb, ws)
        run_i, out, *_ = f.bind(x, gamma, beta, save=False)
        run_i(); torch.cuda.synchronize()
        put("fused sm_100a (this work)", "inference", graph_time(run_i), rel(out, ref))
        try:
            from bwd_op import FusedTrain
            tr = FusedTrain(f)
            step = tr.bind(x, gamma, beta, dy)
            step(); torch.cuda.synchronize()
            put("fused sm_100a (this work)", "training", graph_time(step))
        except ImportError:
            put("fused sm_100a (this work)", "training", None, note="backward not built yet")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        put("fused sm_100a (this work)", "inference", None, note=repr(exc)[:120])

if a.save:
    with open(a.save, "w") as fh:
        json.dump(results, fh, indent=1)
