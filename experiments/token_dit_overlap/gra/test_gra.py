"""Correctness of gemm_resgate_adaln against fp32 torch and against mm + resgate_adaln_rows."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdit import kernels as K  # noqa
from gra import gemm_resgate_adaln  # noqa

torch.manual_seed(0)
dev, bf, D = "cuda", torch.bfloat16, 768
for L, S in ((384, 5), (768, 5), (128, 3)):
    M = L * S
    for Kd, sa in ((768, 3072), (1536, 1536)):
        for nwg in (1, 2):
            for adaln in (True, False):
                abuf = torch.randn(M, sa, device=dev, dtype=bf)
                a = abuf[:, :Kd]
                w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5 * 0.25).to(bf)
                x0 = torch.randn(M, D, device=dev)
                g = torch.randn(L, 4, D, device=dev, dtype=bf)
                gl, ms, mb = g[:, 0], g[:, 1], g[:, 2]
                tok = torch.arange(M, device=dev) % L
                xr = x0 + torch.sigmoid(gl.float()[tok]) * (a.float() @ w.float().t())
                xar = torch.nn.functional.layer_norm(xr, (D,), eps=1e-5) * torch.sigmoid(ms.float()[tok]) + mb.float()[tok]
                x = x0.clone(); xa = torch.zeros(M, D, device=dev, dtype=bf)
                gemm_resgate_adaln(a, w, x, gl, ms if adaln else None, mb if adaln else None, xa if adaln else None, L, nwg=nwg)
                torch.cuda.synchronize()
                ex = float((x - xr).norm() / (xr - x0).norm())
                ea = float((xa.float() - xar).norm() / xar.norm()) if adaln else 0.0
                # the row-pass path for scale
                y = torch.mm(a, w.t()); x2 = x0.clone(); xa2 = torch.empty_like(xa)
                K.resgate_adaln_rows(x2, y, gl, ms if adaln else None, mb if adaln else None, xa2, L)
                ex2 = float((x2 - xr).norm() / (xr - x0).norm())
                ea2 = float((xa2.float() - xar).norm() / xar.norm()) if adaln else 0.0
                ok = ex < 3e-3 and ea < 8e-3
                print(f"L{L} S{S} K{Kd} nwg{nwg} adaln={int(adaln)}  x {ex:.1e} (rows {ex2:.1e})  xa {ea:.1e} (rows {ea2:.1e})  {'ok' if ok else 'FAIL'}", flush=True)
