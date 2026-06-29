"""Verify + bench adaln_fused3 (3 triton kernels) vs materialize vs pytorch compile (fp32)."""
from __future__ import annotations
import torch, torch.nn.functional as F, triton

from miniworld_kernels.kernels.adaln.triton.fused3 import adaln_fused3
from miniworld_kernels.kernels.adaln.triton.inference import adaln_inference_materialize

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
eps = 1e-5


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8]); return m * 1000


def ref(x, cond, lnw, sw, sb, bw):
    xn = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    cn = F.layer_norm(cond, (cond.shape[-1],), lnw, None, eps)
    return torch.sigmoid(F.linear(cn, sw, sb)) * xn + F.linear(cn, bw, None)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    for tag, d, seqs in [("TOKEN", 768, [384, 512, 768, 1024]), ("ATOM", 128, [2048, 4096, 8192])]:
        print(f"\n#### {tag} d={d} fp32 forward")
        print(f"  {'seq':>5} {'M':>8} | {'compile':>8} {'materialize':>11} {'fused3':>8} | {'f3/comp':>7} {'f3/mat':>7}  cos")
        compiled = torch.compile(ref)
        for seq in seqs:
            M = 32 * seq
            x = torch.randn(M, d, device=DEVICE, dtype=torch.float32)
            cond = torch.randn(M, d, device=DEVICE, dtype=torch.float32)
            lnw = torch.randn(d, device=DEVICE, dtype=torch.float32)
            sw = torch.randn(d, d, device=DEVICE, dtype=torch.float32) * d ** -0.5
            sb = torch.randn(d, device=DEVICE, dtype=torch.float32) * 0.1
            bw = torch.randn(d, d, device=DEVICE, dtype=torch.float32) * d ** -0.5
            r = ref(x, cond, lnw, sw, sb, bw)
            y3 = adaln_fused3(x, cond, lnw, sw, sb, bw, eps, eps)
            c = cos(r, y3)
            with torch.no_grad():
                tcomp = t(lambda: compiled(x, cond, lnw, sw, sb, bw))
                tmat = t(lambda: adaln_inference_materialize(x, cond, lnw, sw, sb, bw, eps, eps))
                t3 = t(lambda: adaln_fused3(x, cond, lnw, sw, sb, bw, eps, eps))
            print(f"  {seq:>5} {M:>8} | {tcomp:>8.1f} {tmat:>11.1f} {t3:>8.1f} | "
                  f"{tcomp/t3:>6.2f}x {tmat/t3:>6.2f}x  {c:.5f}", flush=True)


if __name__ == "__main__":
    main()
