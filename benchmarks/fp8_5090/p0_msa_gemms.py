"""Phase 0 for the RTX 5090 OPM / PWA e4m3 routes: the dominant GEMM of each module, default bf16 kernel against the cuBLASLt e4m3 TN GEMM
(the H100 routes' helpers ``opm_fp8.product`` / ``pwa_fp8.contract``: plain cuBLASLt, no sm_90 code).

    python p0_msa_gemms.py opm:L:S ... pwa:C:S:N ...

OPM: O [32 L, 32 L] = A^T B over K = S (default: torch.mm on the s-major bf16 A, B [S, 32 L], as ``opm_sm80._outer`` without a mask); e4m3: A2 / BT
[32 L, S] K-major. PWA: o [8, N, S C] = w v over K = N (default: ``gemm_sm120.contract``); e4m3: w [8, N, N], vT [8, S C, N]. CUDA graph, 128 MB L2
flush, median of 7; TFLOPS = 2 M N K / t."""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _msa_common import graph_time  # noqa: E402

from foldjet_engine.kernels.integrations import opm_fp8, pwa_fp8  # noqa: E402
from foldjet_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120  # noqa: E402

E4 = torch.float8_e4m3fn
print(torch.cuda.get_device_name(), flush=True)
for c in sys.argv[1:]:
    kind, *dims = c.split(":")
    if kind == "opm":
        L, S = map(int, dims)
        a = torch.randn(S, 32 * L, device="cuda").bfloat16()
        b = torch.randn(S, 32 * L, device="cuda").bfloat16()
        a8 = a.t().contiguous().to(E4)
        b8 = b.t().contiguous().to(E4)
        sab = (torch.ones(1, device="cuda"), torch.ones(1, device="cuda"))
        t0 = graph_time(lambda: torch.mm(a.t(), b))
        t1 = graph_time(lambda: opm_fp8.product(a8, b8, sab, L))
        fl = 2 * (32 * L) ** 2 * S
        print(f"opm L{L} S{S}: bf16 {t0:.0f} us ({fl / t0 / 1e6:.0f} TF)  e4m3 {t1:.0f} us ({fl / t1 / 1e6:.0f} TF)  ratio {t0 / t1:.2f}", flush=True)
        del a, b, a8, b8
    else:
        C, S, N = map(int, dims)
        w = torch.rand(8, N, N, device="cuda").bfloat16()
        v = torch.randn(8, N, S * C, device="cuda").bfloat16()
        wq = (w.float() * 256).to(E4)
        vt = v.transpose(1, 2).contiguous().to(E4)
        o = torch.empty(8, N, S * C, device="cuda", dtype=torch.bfloat16)
        one = torch.ones(1, device="cuda")
        t0 = graph_time(lambda: gemm_sm120.contract(w, v))
        t1 = graph_time(lambda: pwa_fp8.contract(wq, vt, o, one, one))
        fl = 2 * 8 * N * N * S * C
        print(f"pwa C{C} S{S} N{N}: bf16 {t0:.0f} us ({fl / t0 / 1e6:.0f} TF)  e4m3 {t1:.0f} us ({fl / t1 / 1e6:.0f} TF)  ratio {t0 / t1:.2f}", flush=True)
        del w, v, wq, vt, o
    torch.cuda.empty_cache()
