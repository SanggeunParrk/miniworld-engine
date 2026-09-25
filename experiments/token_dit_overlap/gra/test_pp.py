"""Correctness of gemm_resgate_adaln_pp against fp32 torch and the mm + resgate_adaln_rows path, incl. repeat-determinism
and a few persistent-grid sizes (so clusters walk 1, 2 and many tiles)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdit import kernels as K  # noqa
from gra import gemm_resgate_adaln_pp  # noqa

torch.manual_seed(0)
dev, bf, D = "cuda", torch.bfloat16, 768
bad = 0
for L, S in ((384, 5), (768, 5), (128, 3)):
    M = L * S
    for Kd, sa in ((768, 3072), (1536, 1536)):
        for adaln in (True, False):
            for mc in (0, 7, 2):
                abuf = torch.randn(M, sa, device=dev, dtype=bf)
                a = abuf[:, :Kd]
                w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5 * 0.25).to(bf)
                x0 = torch.randn(M, D, device=dev)
                g = torch.randn(L, 4, D, device=dev, dtype=bf)
                gl, ms, mb = g[:, 0], g[:, 1], g[:, 2]
                tok = torch.arange(M, device=dev) % L
                xr = x0 + torch.sigmoid(gl.float()[tok]) * (a.float() @ w.float().t())
                xar = torch.nn.functional.layer_norm(xr, (D,), eps=1e-5) * torch.sigmoid(ms.float()[tok]) + mb.float()[tok]
                outs = []
                for rep in range(3):
                    x = x0.clone(); xa = torch.zeros(M, D, device=dev, dtype=bf)
                    gemm_resgate_adaln_pp(a, w, x, gl, ms if adaln else None, mb if adaln else None,
                                          xa if adaln else None, L, max_clusters=mc)
                    torch.cuda.synchronize()
                    outs.append((x.clone(), xa.clone()))
                x, xa = outs[0]
                det = all(torch.equal(o[0], x) and torch.equal(o[1], xa) for o in outs[1:])
                ex = float((x - xr).norm() / (xr - x0).norm())
                ea = float((xa.float() - xar).norm() / xar.norm()) if adaln else 0.0
                y = torch.mm(a, w.t()); x2 = x0.clone(); xa2 = torch.empty_like(xa)
                K.resgate_adaln_rows(x2, y, gl, ms if adaln else None, mb if adaln else None, xa2, L)
                ex2 = float((x2 - xr).norm() / (xr - x0).norm())
                ea2 = float((xa2.float() - xar).norm() / xar.norm()) if adaln else 0.0
                ok = ex < 3e-3 and ea < 8e-3 and det
                bad += not ok
                print(f"L{L} S{S} K{Kd} adaln={int(adaln)} max_cl={mc}: x {ex:.1e} (rows {ex2:.1e})  xa {ea:.1e} "
                      f"(rows {ea2:.1e})  det={det}  {'ok' if ok else 'FAIL'}", flush=True)
print("ALL OK" if bad == 0 else f"{bad} FAILED", flush=True)
