"""E5: accuracy of the external TriMul backends against the module's own fp64 PyTorch reference (the engine's accuracy standard).

    python E5_other_kernels/acc_trimul.py --out records/<card>/ext_trimul_acc.jsonl [--backends cuda pytorch fast_trimul gpumode geonn] [--cases 128:128 768:64 ...]

For each case ``L:d_pair`` (outgoing, d_hidden = d_pair, bf16 activations over fp32 masters, random non-degenerate weights), with and without padding
positions in the mask: the backend's output against the fp64 PyTorch result of the SAME weights and inputs, as relative rms ``||e|| / ||ref||`` and
relative max ``max|e| / max|ref|``, next to the PyTorch backend's own error (bf16 autocast). A refused case is recorded with the reason.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import externals as reg


def build(d: int, seed: int):
    from miniworld_engine.modules.triangle import TriangleMultiplication

    g = torch.Generator(device="cuda").manual_seed(seed)
    m = TriangleMultiplication(d, outgoing=True, master_weights=True, p_drop=0.0).cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g, device="cuda") / d**0.5)
            elif name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g, device="cuda"))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g, device="cuda"))
    return m.eval(), g


def errs(out, ref, base=None):
    """Relative rms / max error of ``out`` against ``ref``; with ``base`` (the residual input) also the error relative to the UPDATE ``ref - base``
    (the residual stream dominates ``out``, so the output-relative error hides most of a kernel's own error)."""
    e = (out.double() - ref).abs()
    r = {"rel_rms": float(e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), "rel_max": float(e.max() / ref.abs().max())}
    if base is not None:
        u = ref - base.double()
        r |= {"upd_rel_rms": float(e.pow(2).mean().sqrt() / u.pow(2).mean().sqrt()), "upd_rel_max": float(e.max() / u.abs().max())}
    return r


def grads(m, pair, mask, backend):
    """Gradients of sum(out * w) w.r.t. the pair and the output-projection weight (fixed random w), or the refusal."""
    m.set_backend(backend)
    x = pair.detach().clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    out = m(x, mask)
    w = torch.randn(out.shape, generator=torch.Generator(device="cuda").manual_seed(7), device="cuda", dtype=torch.float32)
    (out.float() * w).sum().backward()
    return x.grad, m.to_out.weight.grad, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backends", nargs="+", default=["pytorch", "cuda", *reg.TRIMUL_REGISTRY])
    ap.add_argument(
        "--grads", action="store_true", help="training: input and W_o gradients against the fp64 reference (cases without padding)"
    )
    ap.add_argument("--cases", nargs="+", default=["128:128", "512:128", "768:128", "768:64", "768:256", "768:512"])
    a = ap.parse_args()
    reg.install([b for b in a.backends if b in reg.TRIMUL_REGISTRY])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in a.cases:
        n, d = (int(v) for v in case.split(":"))
        m, g = build(d, seed=n + d)
        pair = torch.randn(1, n, n, d, generator=g, device="cuda").bfloat16()
        for padded in (False, True):
            pad = n // 8 if padded else 0
            mask = (torch.arange(n, device="cuda") < n - pad)[None].contiguous()
            ref_m = copy.deepcopy(m).double()
            ref_m.set_backend("pytorch")
            try:
                with torch.no_grad():
                    ref = ref_m(pair.double(), mask)
            except torch.OutOfMemoryError:
                print(
                    json.dumps({"case": case, "padded_mask": padded, "status": "skipped", "reason": "fp64 reference out of memory"}),
                    flush=True,
                )
                continue
            finally:
                del ref_m
            for b in a.backends:
                rec = {"case": case, "L": n, "d": d, "padded_mask": padded, "backend": b, "gpu": torch.cuda.get_device_name()}
                m.set_backend(b)
                with torch.no_grad():
                    why = m.unsupported(pair, mask) if b != "pytorch" else None
                    if why:
                        rec |= {"status": "unsupported", "reason": why}
                    else:
                        try:
                            rec |= {"status": "ok", **errs(m(pair, mask), ref, pair)}
                        except Exception as exc:
                            rec |= {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
                rows.append(rec)
                print(json.dumps(rec), flush=True)
                if a.grads and not padded and b in ("pytorch", "cuda", "fast_trimul"):
                    rec = {
                        "case": case,
                        "L": n,
                        "d": d,
                        "padded_mask": padded,
                        "backend": b,
                        "kind": "grad",
                        "gpu": torch.cuda.get_device_name(),
                    }
                    try:
                        mt = copy.deepcopy(m).train()
                        mt.p_drop = 0.0
                        why = mt.set_backend(b).unsupported(pair, mask) if b != "pytorch" else None
                        if why:
                            rec |= {"status": "unsupported", "reason": why}
                        else:
                            gx, gw, w = grads(mt, pair, mask, b)
                            rt = copy.deepcopy(m).double().train()
                            rt.p_drop = 0.0
                            x64 = pair.double().requires_grad_(True)
                            (rt.set_backend("pytorch")(x64, mask) * w.double()).sum().backward()
                            rec |= {"status": "ok", "dx": errs(gx, x64.grad), "dWo": errs(gw, rt.to_out.weight.grad)}
                    except Exception as exc:
                        rec |= {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
