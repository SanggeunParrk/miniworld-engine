"""Baselines on this GPU for the augmented pair-bias attention core (A = 48, H = 16, D = 48): inference (no grad) and
training (forward + backward: dq, dk, dv, dbias). Rows: PyTorch SDPA (bias as a float mask broadcast over A), torch.compile of
the eager formula, the engine's bf16 Triton core, Anthropic's fpf_apb (forward only; the provider has no backward).
    python bench_base.py [--lengths 384 768] [--anthropic-root ...]"""
import argparse, math, os, sys, traceback
import torch
import torch.nn.functional as F
from common import H, D, make, reference, rel, graph_time

p = argparse.ArgumentParser()
p.add_argument("--lengths", type=int, nargs="+", default=[384, 768])
p.add_argument("--A", type=int, default=48)
p.add_argument("--anthropic-root", default=os.environ.get("ANTHROPIC_ROOT", "/NHNHOME/WORKSPACE/26mohw002_A/psk6950/refs/uplifting-biomolecular-modeling"))
p.add_argument("--no-compile", action="store_true")
a = p.parse_args()
A = a.A


def eager(q, k, v, bias):                               # q, k, v [A, 1, L, H, D]; bias [H, L, L]
    qh, kh, vh = (t[:, 0].transpose(1, 2) for t in (q, k, v))
    s = (qh.float() @ kh.float().transpose(-1, -2)) / math.sqrt(D) + bias.float()[None]
    return (torch.softmax(s, -1) @ vh.float()).to(q.dtype).transpose(1, 2)[:, None]


def sdpa(q, k, v, bias):
    qh, kh, vh = (t[:, 0].transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias[None].expand(q.shape[0], -1, -1, -1)).transpose(1, 2)[:, None]


def triton_core(q, k, v, bias):
    from miniworld_engine.kernels.augmented_attention import triton_augmented_attention_pair_bias as f
    return f(q, k, v, bias.permute(1, 2, 0)[None].contiguous() if bias.dim() == 3 else bias, None)


def put(L, name, mode, us, err=None, note=""):
    print(f"L{L} {mode:9s} {name:34s} {('%8.1f us' % us) if us else '     n/a'}  {'' if err is None else 'rel %.2e' % err}  {note}", flush=True)


for L in a.lengths:
    q, k, v, bias = make(A, L)
    ref = reference(q, k, v, bias)
    rows = [("PyTorch eager", eager), ("PyTorch SDPA", sdpa)]
    if not a.no_compile:
        rows.append(("torch.compile (eager formula)", torch.compile(eager)))
    rows.append(("engine Triton bf16 core", triton_core))
    for name, fn in rows:
        try:
            bias_in = bias.permute(1, 2, 0)[None].contiguous() if fn is triton_core else bias
            f2 = (lambda qq, kk, vv, bb: fn(qq, kk, vv, bb)) if fn is not triton_core else (lambda qq, kk, vv, bb: triton_core(qq, kk, vv, bb))
            with torch.no_grad():
                o = f2(q, k, v, bias_in if fn is triton_core else bias)
                put(L, name, "inference", graph_time(lambda: f2(q, k, v, bias_in if fn is triton_core else bias)), rel(o, ref))
            qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
            bb = (bias_in if fn is triton_core else bias).detach().clone().requires_grad_(True)
            do = torch.randn_like(q)
            def step():
                out = f2(qq, kk, vv, bb)
                out.backward(do)
            for _ in range(2): step()
            for t in (qq, kk, vv, bb): t.grad = None
            put(L, name, "training", graph_time(step))
        except Exception as exc:  # noqa: BLE001
            put(L, name, "inference", None, note=repr(exc)[:150])
    # Anthropic fpf_apb (forward only)
    try:
        sys.path.insert(0, os.path.join(a.anthropic_root, "common", "opt_core"))
        from opt_core.kernels.apb.fpf_apb import apb_triton as T
        qv, kv, vv = (t[:, 0] for t in (q, k, v))              # [A, L, H, D] views
        fn = lambda: T.apb_views(qv, kv, vv, bias, scale=1.0 / math.sqrt(D))
        with torch.no_grad():
            o = fn()
            put(L, "Anthropic fpf_apb", "inference", graph_time(fn), rel(o.reshape(A, 1, L, H, D), ref))
        put(L, "Anthropic fpf_apb", "training", None, note="no backward in the Anthropic provider")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        put(L, "Anthropic fpf_apb", "inference", None, note=repr(exc)[:150])
