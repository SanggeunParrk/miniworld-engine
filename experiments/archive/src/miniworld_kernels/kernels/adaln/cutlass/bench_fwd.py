"""Bench CUTLASS TF32 fused fwd vs current materialize vs pytorch compile (fp32, token+atom)."""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F
import triton
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/psk6950/miniworld-kernels/_ct_cutlass/cutlass"
BUILD = "/home/psk6950/.cache/adaln_cutlass_fwd"
os.makedirs(BUILD, exist_ok=True)
mod = load(name="adaln_cutlass_fwd", sources=[os.path.join(HERE, "adaln_fwd.cu")],
           extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include")],
           extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90a", "--expt-relaxed-constexpr",
                              "--expt-extended-lambda", "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-DNDEBUG"],
           build_directory=BUILD, verbose=False)

import sys
sys.path.insert(0, "/home/psk6950/miniworld-kernels/src")
from miniworld_kernels.kernels.adaln.triton.inference import _cond_affine, adaln_inference
from miniworld_kernels.kernels.layernorm_linear.te_style import _ln_materialize

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
eps = 1e-5


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return m * 1000


def ref(x, cond, lnw, sw, sb, bw):
    xn = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    cn = F.layer_norm(cond, (cond.shape[-1],), lnw, None, eps)
    return torch.sigmoid(F.linear(cn, sw, sb)) * xn + F.linear(cn, bw, None)


def cutlass_fwd(x, cond, lnw, Ws, Wb, scale_b, ones_d, zeros_d):
    cond_aff = _cond_affine(cond, lnw, eps)
    x_hat = _ln_materialize(x, ones_d, zeros_d, eps)[0]
    y1 = mod.adaln_fwd1(cond_aff, Ws, scale_b, x_hat)
    return mod.adaln_fwd2(cond_aff, Wb, y1)


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    for tag, d, seqs in [("TOKEN", 768, [384, 512, 768, 1024]), ("ATOM", 128, [2048, 4096, 8192])]:
        print(f"\n#### {tag} d={d} fp32 forward")
        print(f"  {'seq':>5} {'M':>8} | {'compile':>8} {'materialize':>11} {'cutlass':>8} | {'ct/comp':>7} {'ct/mat':>7}")
        compiled = torch.compile(ref)
        for seq in seqs:
            M = 32 * seq
            x = torch.randn(M, d, device="cuda", dtype=torch.float32)
            cond = torch.randn(M, d, device="cuda", dtype=torch.float32)
            lnw = torch.randn(d, device="cuda", dtype=torch.float32)
            Ws = torch.randn(d, d, device="cuda", dtype=torch.float32) * d ** -0.5
            Wb = torch.randn(d, d, device="cuda", dtype=torch.float32) * d ** -0.5
            sb = torch.randn(d, device="cuda", dtype=torch.float32) * 0.1
            ones_d = torch.ones(d, device="cuda", dtype=torch.float32)
            zeros_d = torch.zeros(d, device="cuda", dtype=torch.float32)
            # accuracy
            r = ref(x, cond, lnw, Ws, sb, Wb)
            yc = cutlass_fwd(x, cond, lnw, Ws, Wb, sb, ones_d, zeros_d)
            c = torch.nn.functional.cosine_similarity(r.flatten(), yc.flatten(), dim=0).item()
            with torch.no_grad():
                tcomp = t(lambda: compiled(x, cond, lnw, Ws, sb, Wb))
                tmat = t(lambda: adaln_inference(x, cond, lnw, Ws, sb, Wb, eps, eps))
                tct = t(lambda: cutlass_fwd(x, cond, lnw, Ws, Wb, sb, ones_d, zeros_d))
            print(f"  {seq:>5} {M:>8} | {tcomp:>8.1f} {tmat:>11.1f} {tct:>8.1f} | "
                  f"{tcomp/tct:>6.2f}x {tmat/tct:>6.2f}x  cos={c:.5f}", flush=True)


if __name__ == "__main__":
    main()
