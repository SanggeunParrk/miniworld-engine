"""fp32-IO adaln: TF32 matmul (cos=1.0) vs bf16 matmul (fp32 accum, cos~0.9999) — speed + accuracy.

The only lever left for fp32-IO speed (TF32 WGMMA custom kernel infeasible in this env). IO + LN
stay fp32; only the GEMM operands are downcast to bf16. Compares vs torch.compile baseline.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.adaln.triton import inference as INF
from miniworld_kernels.kernels.adaln.triton import training as TR

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def t(fn, gtn=None):
    m, _, _ = triton.testing.do_bench(fn, warmup=20, rep=80, quantiles=[0.5, 0.2, 0.8],
                                      grad_to_none=gtn or [])
    return m


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def ref(x, cond, lnw, sw, sb, bw, eps):
    xn = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    cn = F.layer_norm(cond, (cond.shape[-1],), lnw, None, eps)
    return torch.sigmoid(F.linear(cn, sw, sb)) * xn + F.linear(cn, bw, None)


def mk(d_h, d_c, dt=torch.float32):
    lnw = torch.randn(d_c, device=DEVICE, dtype=dt)
    sw = torch.randn(d_h, d_c, device=DEVICE, dtype=dt) * d_c ** -0.5
    sb = torch.randn(d_h, device=DEVICE, dtype=dt) * 0.1
    bw = torch.randn(d_h, d_c, device=DEVICE, dtype=dt) * d_c ** -0.5
    return lnw, sw, sb, bw


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    eps = 1e-5
    for tag, d, seqs in [("TOKEN", 768, [384, 512, 768, 1024]), ("ATOM", 128, [2048, 4096, 8192])]:
        print(f"\n#### {tag} d={d} fp32")
        # accuracy check at one size
        M0 = 32 * seqs[1]
        x = torch.randn(M0, d, device=DEVICE, dtype=torch.float32)
        cond = torch.randn(M0, d, device=DEVICE, dtype=torch.float32)
        lnw, sw, sb, bw = mk(d, d)
        r = ref(x, cond, lnw, sw, sb, bw, eps)
        INF.set_gemm_bf16(False); o_tf = INF.adaln_inference(x, cond, lnw, sw, sb, bw, eps, eps)
        INF.set_gemm_bf16(True);  o_bf = INF.adaln_inference(x, cond, lnw, sw, sb, bw, eps, eps)
        INF.set_gemm_bf16(False)
        print(f"  acc(inf): TF32 cos={cos(r, o_tf):.6f}  bf16mm cos={cos(r, o_bf):.6f}")

        print(f"  {'seq':>5} {'M':>8} | {'INF tf32':>9} {'INF bf16':>9} {'spd':>5} || "
              f"{'TR tf32':>9} {'TR bf16':>9} {'spd':>5}")
        for seq in seqs:
            M = 32 * seq
            x = torch.randn(M, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
            cond = torch.randn(M, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
            lnw, sw, sb, bw = mk(d, d)
            for p in (lnw, sw, sb, bw):
                p.requires_grad_()
            dy = torch.randn(M, d, device=DEVICE, dtype=torch.float32)
            gtn = [x, cond, lnw, sw, sb, bw]

            with torch.no_grad():
                INF.set_gemm_bf16(False); ti_tf = t(lambda: INF.adaln_inference(x, cond, lnw, sw, sb, bw, eps, eps))
                INF.set_gemm_bf16(True);  ti_bf = t(lambda: INF.adaln_inference(x, cond, lnw, sw, sb, bw, eps, eps))
                INF.set_gemm_bf16(False)

            def full(bf):
                TR.set_gemm_bf16(bf)
                for p in gtn:
                    p.grad = None
                o = TR.adaln_train(x, cond, lnw, sw, sb, bw, eps, eps)
                o.backward(dy)
            tt_tf = t(lambda: full(False), gtn)
            tt_bf = t(lambda: full(True), gtn)
            TR.set_gemm_bf16(False)
            print(f"  {seq:>5} {M:>8} | {ti_tf*1e3:>9.1f} {ti_bf*1e3:>9.1f} {ti_tf/ti_bf:>4.2f}x || "
                  f"{tt_tf*1e3:>9.1f} {tt_bf*1e3:>9.1f} {tt_tf/tt_bf:>4.2f}x", flush=True)


if __name__ == "__main__":
    main()
