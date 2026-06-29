"""Validate the existing cute layernorm_linear on the adaln forward shape (cond -> [scale|bias]).

Checks: (a) does it run for fp32 and bf16 inputs, (b) cos vs fp32 reference, (c) speed vs the
materialize cuBLAS GEMM. This gates whether the cute path is viable for adaln (esp. fp32 precision)
BEFORE building a custom sigmoid+x_hat epilogue.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import triton

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=20, rep=80, quantiles=[0.5, 0.2, 0.8])
    return m * 1000


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def run(d, seq, n_aug=32, eps=1e-5):
    from miniworld_kernels.kernels.layernorm_linear.cute import layernorm_linear as ll_cute
    M = n_aug * seq
    NX = NC = d
    print(f"\n## d={d} seq={seq} M={M}")
    for dt in (torch.float32, torch.bfloat16):
        cond = torch.randn(M, NC, device=DEVICE, dtype=dt)
        lnw = torch.randn(NC, device=DEVICE, dtype=dt)
        Ws = torch.randn(NX, NC, device=DEVICE, dtype=dt) * NC ** -0.5
        Wb = torch.randn(NX, NC, device=DEVICE, dtype=dt) * NC ** -0.5
        sb_b = torch.randn(NX, device=DEVICE, dtype=dt) * 0.1
        w_cat = torch.cat([Ws, Wb], 0).contiguous()                 # (2NX, NC)
        b_cat = torch.cat([sb_b, torch.zeros(NX, device=DEVICE, dtype=dt)], 0).contiguous()
        lnb0 = torch.zeros(NC, device=DEVICE, dtype=dt)

        # reference scale|bias in fp32
        cond_norm = F.layer_norm(cond.float(), (NC,), lnw.float(), None, eps)
        scale_ref = F.linear(cond_norm, Ws.float(), sb_b.float())
        bias_ref = F.linear(cond_norm, Wb.float(), None)
        sb_ref = torch.cat([scale_ref, bias_ref], dim=1)

        try:
            out = ll_cute(cond, lnw, lnb0, w_cat, b_cat, eps, save_stats=True)
            sb_cute = out[0] if isinstance(out, tuple) else out
            c = cos(sb_ref, sb_cute)
            tm = t(lambda: ll_cute(cond, lnw, lnb0, w_cat, b_cat, eps, save_stats=True))
            status = f"cos={c:.6f}  time={tm:.1f}us"
        except Exception as e:  # noqa: BLE001
            status = f"FAILED: {type(e).__name__}: {str(e)[:160]}"

        # baseline: materialize cuBLAS (cond_aff + F.linear)
        try:
            cond_aff = (F.layer_norm(cond.float(), (NC,), lnw.float(), None, eps)).to(dt)
            tm_cub = t(lambda: F.linear(F.layer_norm(cond, (NC,), lnw, None, eps), w_cat, b_cat))
            base = f"cuBLAS_materialize={tm_cub:.1f}us"
        except Exception as e:  # noqa: BLE001
            base = f"cuBLAS FAILED {e}"
        print(f"  {str(dt):16s}: {status}   | {base}")


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    run(768, 1024)
    run(128, 8192)


if __name__ == "__main__":
    main()
