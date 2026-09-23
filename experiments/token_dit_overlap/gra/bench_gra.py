"""Kernel-level latency: gemm_resgate_adaln vs torch.mm alone vs torch.mm + resgate_adaln_rows.

Timed the engine's way (bench.us -> triton do_bench, L2 evicted between iterations). The earlier numbers here came from
CUDA-graph replays with everything hot in L2, which is the wrong comparison for a kernel whose whole point is to not
write an intermediate."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import us as t_us  # noqa
from tdit import kernels as K  # noqa
from gra import gemm_resgate_adaln  # noqa

nwgs = [int(v) for v in (sys.argv[1:] or ["1", "2"])]




dev, bf, D = "cuda", torch.bfloat16, 768
for L in (384, 768):
    M = 5 * L
    for Kd, sa, nm in ((768, 3072, "Wo"), (1536, 1536, "squeeze")):
        abuf = torch.randn(M, sa, device=dev, dtype=bf); a = abuf[:, :Kd]
        w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5).to(bf)
        x = torch.randn(M, D, device=dev); y = torch.empty(M, D, device=dev, dtype=bf); xa = torch.empty_like(y)
        g = torch.randn(L, 4, D, device=dev, dtype=bf); gl, ms, mb = g[:, 0], g[:, 1], g[:, 2]
        tm = t_us(lambda: torch.mm(a, w.t(), out=y))
        tr = t_us(lambda: K.resgate_adaln_rows(x, y, gl, ms, mb, xa, L))
        row = f"L{L} {nm:<7s} M{M} K{Kd}: mm {tm:6.1f}  rows {tr:6.1f}  mm+rows {tm + tr:6.1f}"
        for nwg in nwgs:
            tg = t_us(lambda: gemm_resgate_adaln(a, w, x, gl, ms, mb, xa, L, nwg=nwg))
            tg0 = t_us(lambda: gemm_resgate_adaln(a, w, x, gl, None, None, None, L, nwg=nwg))
            row += f" | gra nwg{nwg} {tg:6.1f} (no adaln {tg0:6.1f})"
        print(row + "  us", flush=True)
