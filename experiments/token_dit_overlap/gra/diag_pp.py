"""One gemm_resgate_adaln_pp call at one config (a CUDA fault kills the context, so one config a process)."""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gra import fused  # noqa
gemm_resgate_adaln_pp = fused()[0]
p = argparse.ArgumentParser()
p.add_argument("--L", type=int, default=384); p.add_argument("--K", type=int, default=768)
p.add_argument("--adaln", type=int, default=1); p.add_argument("--mc", type=int, default=0)
a = p.parse_args()
torch.manual_seed(0)
dev, bf, D, L = "cuda", torch.bfloat16, 768, a.L
M = 5 * L
A = torch.randn(M, a.K, device=dev, dtype=bf); W = (torch.randn(D, a.K, device=dev) * a.K ** -0.5 * 0.25).to(bf)
x0 = torch.randn(M, D, device=dev); g = torch.randn(L, 4, D, device=dev, dtype=bf); gl, ms, mb = g[:, 0], g[:, 1], g[:, 2]
tok = torch.arange(M, device=dev) % L
xr = x0 + torch.sigmoid(gl.float()[tok]) * (A.float() @ W.float().t())
x = x0.clone(); xa = torch.zeros(M, D, device=dev, dtype=bf)
gemm_resgate_adaln_pp(A, W, x, gl, ms if a.adaln else None, mb if a.adaln else None, xa if a.adaln else None, L, max_clusters=a.mc)
torch.cuda.synchronize()
ex = float((x - xr).norm() / (xr - x0).norm())
bad_rows = ((x - xr).abs().amax(1) > 1e-2).nonzero().flatten()
print(f"L{L} K{a.K} adaln={a.adaln} mc={a.mc}: x err {ex:.1e}, bad rows {bad_rows.numel()} "
      f"(tiles {sorted(set((bad_rows // 64).tolist()))[:12]})", flush=True)
