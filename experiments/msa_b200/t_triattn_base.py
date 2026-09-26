"""Triangle-attention forward baselines on this B200 (sustained, power-capped): SDPA, opt_core k2b / flash (Triton), cueq if present.
q/k/v [1, N, H, S, D] bf16, bias [1, 1, H, S, S] fp32, no mask; N = S = L, H = 4, D = 32."""
import os, sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
sys.path.insert(0, os.environ["OPT_CORE_DIR"])
from opt_core.kernels import triattn as ta
H, D = 4, 32
for L in (384, 768):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(1, L, H, L, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))
    bias = torch.randn(1, 1, H, L, L, device="cuda", generator=g)
    ref = ta.sdpa_reference(q.float(), k.float(), v.float(), bias)
    rows = [("sdpa (torch)", lambda: ta.sdpa_reference(q, k, v, bias))]
    for w in ("k2b", "flash"):
        rows.append((w, (lambda w=w: ta.triangle_attention(q, k, v, bias, None, None, word=w))))
    try:
        import cuequivariance_ops_torch as cq
        rows.append(("cueq", lambda: cq.triangle_attention(q, k, v, bias, None, None)))
    except Exception as e:
        print("cueq unavailable:", str(e)[:80])
    fl = 4 * L * H * L * L * D
    for name, fn in rows:
        try:
            out = fn(); err = ((out.float() - ref).norm() / ref.norm()).item()
            r = sustained(fn, secs=2.0)
            print(f"L={L} {name:14s} {r['ms']*1e3:8.1f} us  {r['J']*1e3:7.2f} mJ  {r['W']:5.0f} W  {fl/(r['ms']*1e-3)/1e12:6.1f} TF/s  rel.err {err:.2e}", flush=True)
        except Exception as e:
            print(f"L={L} {name:14s} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
