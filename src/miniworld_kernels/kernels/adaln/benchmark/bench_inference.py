"""Verify + benchmark inference-only adaLN (adaln_inference) vs pytorch eager / compile / current triton.

team-gm harness style: triton.testing.do_bench timing, parseable stdout table, fp32 main.
Run via srun on a compute node (NEVER login node). Forward-only (inference).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.adaln.triton.inference import (
    adaln_inference,
    adaln_inference_fused,
    adaln_inference_materialize,
)
from miniworld_kernels.kernels.adaln.triton.main import triton_adaptive_layer_norm

DEVICE = "cuda"

# Match the official harness (config/bench.yaml allow_tf32=true) so fp32 GEMMs use TF32 for
# ALL paths — otherwise eager/compile run true-fp32 while ours uses TF32 (unfair).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def do_bench(fn):
    med, p20, p80 = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return med


def make_weights(d_hidden, d_cond, dtype):
    lnw = torch.randn(d_cond, device=DEVICE, dtype=dtype)
    scale_w = torch.randn(d_hidden, d_cond, device=DEVICE, dtype=dtype) * (d_cond ** -0.5)
    scale_b = torch.randn(d_hidden, device=DEVICE, dtype=dtype) * 0.1
    bias_w = torch.randn(d_hidden, d_cond, device=DEVICE, dtype=dtype) * (d_cond ** -0.5)
    return lnw, scale_w, scale_b, bias_w


def ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w, eps):
    x_norm = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    cond_norm = F.layer_norm(cond, (cond.shape[-1],), lnw, None, eps)
    scale = F.linear(cond_norm, scale_w, scale_b)
    bias = F.linear(cond_norm, bias_w, None)
    return torch.sigmoid(scale) * x_norm + bias


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def verify(d_hidden, d_cond, M, dtype, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype)
    cond = torch.randn(M, d_cond, device=DEVICE, dtype=dtype)
    lnw, scale_w, scale_b, bias_w = make_weights(d_hidden, d_cond, dtype)
    ref = ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w, eps)
    out_m = adaln_inference_materialize(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps)
    out_f = adaln_inference_fused(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps)
    cm = cos(ref, out_m)
    cf = cos(ref, out_f)
    em = (ref.float() - out_m.float()).abs().max().item()
    ef = (ref.float() - out_f.float()).abs().max().item()
    print(f"  verify d={d_hidden} M={M} {dtype}: materialize cos={cm:.6f} maxabs={em:.3e} | "
          f"fused cos={cf:.6f} maxabs={ef:.3e}")
    return cm


def bench_case(tag, d_hidden, d_cond, seqs, dtype, n_augment=48):
    eps = 1e-5
    print(f"\n## {tag}  d={d_hidden} dtype={dtype}")
    print(f"{'seq':>6} {'M':>8} | {'pt_eager':>9} {'pt_compile':>11} {'cur_triton':>11} "
          f"{'ours_mat':>9} {'ours_fused':>11} | {'best_vs_comp':>12} {'best_vs_cur':>12}")
    # cache cat weights for ours (inference: weights fixed)
    for seq in seqs:
        M = n_augment * seq
        x = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype)
        cond = torch.randn(M, d_cond, device=DEVICE, dtype=dtype)
        lnw, scale_w, scale_b, bias_w = make_weights(d_hidden, d_cond, dtype)
        w_cat = torch.cat([scale_w, bias_w], dim=0).contiguous()
        b_cat = torch.cat([scale_b, scale_b.new_zeros(d_hidden)], dim=0).contiguous()

        def f_eager():
            return ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w, eps)

        compiled = torch.compile(ref_pytorch)

        def f_compile():
            return compiled(x, cond, lnw, scale_w, scale_b, bias_w, eps)

        def f_cur():
            return triton_adaptive_layer_norm(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps)

        def f_mat():
            return adaln_inference_materialize(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps,
                                               weight_cat=w_cat, bias_cat=b_cat)

        def f_fused():
            return adaln_inference_fused(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps)

        with torch.no_grad():
            t_eager = do_bench(f_eager)
            t_comp = do_bench(f_compile)
            try:
                t_cur = do_bench(f_cur)
            except Exception:  # noqa: BLE001
                t_cur = float("nan")
            t_mat = do_bench(f_mat)
            t_fused = do_bench(f_fused)
        best = min(t_mat, t_fused)
        print(f"{seq:>6} {M:>8} | {t_eager:>9.4f} {t_comp:>11.4f} {t_cur:>11.4f} "
              f"{t_mat:>9.4f} {t_fused:>11.4f} | {t_comp/best:>11.2f}x {t_cur/best:>11.2f}x")


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    print("=== correctness (fp32) ===")
    verify(768, 768, 32 * 512, torch.float32)
    verify(128, 128, 32 * 4096, torch.float32)
    print("=== correctness (bf16) ===")
    verify(768, 768, 32 * 512, torch.bfloat16)

    # token path
    bench_case("TOKEN fp32", 768, 768, [384, 512, 640, 768, 896, 1024], torch.float32)
    # atom path
    bench_case("ATOM fp32", 128, 128, [2048, 4096, 6144, 8192], torch.float32)


if __name__ == "__main__":
    main()
