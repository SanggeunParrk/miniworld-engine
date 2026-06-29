"""Verify (grad cos) + bench the FUSED triton dgrad (in-kernel GEMM + cond-LN-bwd) vs the cuBLAS
dgrad, both against pytorch eager/compile. Only dW (wgrad) stays cuBLAS in either path.

team-gm harness style. fp32 main, TF32 on (fair). Run via srun. A=48. Forward+backward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.adaln.triton.training import adaln_train, set_dgrad_triton

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
eps = 1e-5


def do_bench(fn, gtn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8],
                                      grad_to_none=gtn)
    return m


def make_params(d_hidden, d_cond, dtype):
    lnw = (torch.randn(d_cond, device=DEVICE, dtype=dtype)).requires_grad_()
    scale_w = (torch.randn(d_hidden, d_cond, device=DEVICE, dtype=dtype) * (d_cond ** -0.5)).requires_grad_()
    scale_b = (torch.randn(d_hidden, device=DEVICE, dtype=dtype) * 0.1).requires_grad_()
    bias_w = (torch.randn(d_hidden, d_cond, device=DEVICE, dtype=dtype) * (d_cond ** -0.5)).requires_grad_()
    return lnw, scale_w, scale_b, bias_w


def ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w):
    x_norm = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    cond_norm = F.layer_norm(cond, (cond.shape[-1],), lnw, None, eps)
    return torch.sigmoid(F.linear(cond_norm, scale_w, scale_b)) * x_norm + F.linear(cond_norm, bias_w, None)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def verify(d_hidden, d_cond, M, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype, requires_grad=True)
    cond = torch.randn(M, d_cond, device=DEVICE, dtype=dtype, requires_grad=True)
    lnw, scale_w, scale_b, bias_w = make_params(d_hidden, d_cond, dtype)
    dy = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype)
    params = (x, cond, lnw, scale_w, scale_b, bias_w)

    ref = ref_pytorch(*params)
    ref.backward(dy)
    g_ref = [t.grad.clone() for t in params]
    for t in params:
        t.grad = None

    set_dgrad_triton(True)
    out = adaln_train(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps)
    fwd_cos = cos(ref, out)
    out.backward(dy)
    g_our = [t.grad.clone() for t in params]

    names = ["dx", "dcond", "dlnw", "dWs", "dsb", "dWb"]
    cs = [cos(a, b) for a, b in zip(g_ref, g_our)]
    print(f"  [triton-dgrad] d={d_hidden} M={M} {dtype}: fwd={fwd_cos:.6f} | " +
          " ".join(f"{n}={c:.5f}" for n, c in zip(names, cs)), flush=True)
    return min([fwd_cos, *cs])


def bench_case(tag, d_hidden, d_cond, seqs, dtype, n_augment=48):
    print(f"\n## {tag}  d={d_hidden} dtype={dtype}  (fwd+bwd, A={n_augment})")
    print(f"{'seq':>6} {'M':>8} | {'pt_eager':>9} {'pt_compile':>11} {'cublas_dg':>10} {'triton_dg':>10} | "
          f"{'tri/comp':>9} {'tri/cub':>8}", flush=True)
    for seq in seqs:
        M = n_augment * seq
        x = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype, requires_grad=True)
        cond = torch.randn(M, d_cond, device=DEVICE, dtype=dtype, requires_grad=True)
        lnw, scale_w, scale_b, bias_w = make_params(d_hidden, d_cond, dtype)
        dy = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype)
        gtn = [x, cond, lnw, scale_w, scale_b, bias_w]

        def full(fn):
            for t in gtn:
                t.grad = None
            fn().backward(dy)

        compiled = torch.compile(ref_pytorch)

        def f_eager():
            full(lambda: ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w))

        def f_compile():
            full(lambda: compiled(x, cond, lnw, scale_w, scale_b, bias_w))

        def f_ours():
            full(lambda: adaln_train(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps))

        t_eager = do_bench(f_eager, gtn)
        t_comp = do_bench(f_compile, gtn)
        set_dgrad_triton(False)
        t_cub = do_bench(f_ours, gtn)
        set_dgrad_triton(True)
        t_tri = do_bench(f_ours, gtn)
        print(f"{seq:>6} {M:>8} | {t_eager:>9.4f} {t_comp:>11.4f} {t_cub:>10.4f} {t_tri:>10.4f} | "
              f"{t_comp/t_tri:>8.2f}x {t_cub/t_tri:>7.2f}x", flush=True)


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    print("=== grad correctness (triton dgrad) ===")
    verify(768, 768, 48 * 512, torch.float32)
    verify(128, 128, 48 * 4096, torch.float32)

    bench_case("TOKEN fp32", 768, 768, [384, 512, 640, 768, 896, 1024], torch.float32)
    bench_case("ATOM fp32", 128, 128, [2048, 4096, 6144, 8192], torch.float32)


if __name__ == "__main__":
    main()
