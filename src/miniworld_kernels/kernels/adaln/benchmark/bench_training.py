"""Verify (grad cosine) + benchmark training (fwd+bwd) adaLN vs pytorch eager/compile & current triton.

team-gm harness style. fp32 main, TF32 on (fair). Run via srun. Forward+backward ('full').
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.adaln.triton.training import adaln_train
from miniworld_kernels.kernels.adaln.triton.main import triton_adaptive_layer_norm

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def do_bench(fn, grad_to_none=None):
    med, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8],
                                        grad_to_none=grad_to_none or [])
    return med


def make_params(d_hidden, d_cond, dtype):
    lnw = (torch.randn(d_cond, device=DEVICE, dtype=dtype)).requires_grad_()
    scale_w = (torch.randn(d_hidden, d_cond, device=DEVICE, dtype=dtype) * (d_cond ** -0.5)).requires_grad_()
    scale_b = (torch.randn(d_hidden, device=DEVICE, dtype=dtype) * 0.1).requires_grad_()
    bias_w = (torch.randn(d_hidden, d_cond, device=DEVICE, dtype=dtype) * (d_cond ** -0.5)).requires_grad_()
    return lnw, scale_w, scale_b, bias_w


def ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w, eps):
    x_norm = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    cond_norm = F.layer_norm(cond, (cond.shape[-1],), lnw, None, eps)
    scale = F.linear(cond_norm, scale_w, scale_b)
    bias = F.linear(cond_norm, bias_w, None)
    return torch.sigmoid(scale) * x_norm + bias


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def verify(d_hidden, d_cond, M, dtype, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype, requires_grad=True)
    cond = torch.randn(M, d_cond, device=DEVICE, dtype=dtype, requires_grad=True)
    lnw, scale_w, scale_b, bias_w = make_params(d_hidden, d_cond, dtype)
    dy = torch.randn(M, d_hidden, device=DEVICE, dtype=dtype)

    ref = ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w, eps)
    ref.backward(dy)
    g_ref = [t.grad.clone() for t in (x, cond, lnw, scale_w, scale_b, bias_w)]
    for t in (x, cond, lnw, scale_w, scale_b, bias_w):
        t.grad = None

    out = adaln_train(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps)
    fwd_cos = cos(ref, out)
    out.backward(dy)
    g_our = [t.grad.clone() for t in (x, cond, lnw, scale_w, scale_b, bias_w)]

    names = ["dx", "dcond", "dlnw", "dWs", "dsb", "dWb"]
    cs = [cos(a, b) for a, b in zip(g_ref, g_our)]
    print(f"  d={d_hidden} M={M} {dtype}: fwd_cos={fwd_cos:.6f} | " +
          " ".join(f"{n}={c:.5f}" for n, c in zip(names, cs)))
    return min([fwd_cos, *cs])


def bench_case(tag, d_hidden, d_cond, seqs, dtype, n_augment=48):
    eps = 1e-5
    print(f"\n## {tag}  d={d_hidden} dtype={dtype}  (fwd+bwd)")
    print(f"{'seq':>6} {'M':>8} | {'pt_eager':>9} {'pt_compile':>11} {'cur_triton':>11} {'ours_train':>11} | "
          f"{'vs_comp':>8} {'vs_eager':>9}", flush=True)
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
            out = fn()
            out.backward(dy)
            return out

        compiled = torch.compile(ref_pytorch)

        def f_eager():
            return full(lambda: ref_pytorch(x, cond, lnw, scale_w, scale_b, bias_w, eps))

        def f_compile():
            return full(lambda: compiled(x, cond, lnw, scale_w, scale_b, bias_w, eps))

        def f_cur():
            return full(lambda: triton_adaptive_layer_norm(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps))

        def f_ours():
            return full(lambda: adaln_train(x, cond, lnw, scale_w, scale_b, bias_w, eps, eps))

        t_eager = do_bench(f_eager, gtn)
        t_comp = do_bench(f_compile, gtn)
        try:
            t_cur = do_bench(f_cur, gtn)
        except Exception:  # noqa: BLE001
            t_cur = float("nan")
        t_ours = do_bench(f_ours, gtn)
        print(f"{seq:>6} {M:>8} | {t_eager:>9.4f} {t_comp:>11.4f} {t_cur:>11.4f} {t_ours:>11.4f} | "
              f"{t_comp/t_ours:>7.2f}x {t_eager/t_ours:>8.2f}x", flush=True)


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    print("=== grad correctness (fp32) ===")
    verify(768, 768, 32 * 512, torch.float32)
    verify(128, 128, 32 * 4096, torch.float32)
    print("=== grad correctness (bf16) ===")
    verify(768, 768, 32 * 512, torch.bfloat16)

    bench_case("TOKEN fp32", 768, 768, [384, 512, 640, 768, 896, 1024], torch.float32)
    bench_case("ATOM fp32", 128, 128, [2048, 4096, 6144, 8192], torch.float32)


if __name__ == "__main__":
    main()
