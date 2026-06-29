"""Verify (grad cos) + bench fused3 training (fwd+bwd) vs eager/compile and vs adaln_train (cuBLAS)."""
from __future__ import annotations
import torch, torch.nn.functional as F, triton

from miniworld_kernels.kernels.adaln.triton.fused3 import adaln_fused3_train
from miniworld_kernels.kernels.adaln.triton.training import adaln_train

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
eps = 1e-5


def t(fn, gtn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8], grad_to_none=gtn)
    return m * 1000


def ref(x, cond, lnw, sw, sb, bw):
    xn = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    cn = F.layer_norm(cond, (cond.shape[-1],), lnw, None, eps)
    return torch.sigmoid(F.linear(cn, sw, sb)) * xn + F.linear(cn, bw, None)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def verify(d, M):
    torch.manual_seed(0)
    x = torch.randn(M, d, device=DEVICE, requires_grad=True)
    cond = torch.randn(M, d, device=DEVICE, requires_grad=True)
    lnw = (torch.randn(d, device=DEVICE)).requires_grad_()
    sw = (torch.randn(d, d, device=DEVICE) * d ** -0.5).requires_grad_()
    sb = (torch.randn(d, device=DEVICE) * 0.1).requires_grad_()
    bw = (torch.randn(d, d, device=DEVICE) * d ** -0.5).requires_grad_()
    dy = torch.randn(M, d, device=DEVICE)
    r = ref(x, cond, lnw, sw, sb, bw); r.backward(dy)
    gref = [p.grad.clone() for p in (x, cond, lnw, sw, sb, bw)]
    for p in (x, cond, lnw, sw, sb, bw): p.grad = None
    o = adaln_fused3_train(x, cond, lnw, sw, sb, bw, eps, eps)
    fc = cos(r, o); o.backward(dy)
    gour = [p.grad.clone() for p in (x, cond, lnw, sw, sb, bw)]
    nm = ["dx", "dcond", "dlnw", "dWs", "dsb", "dWb"]
    print(f"  d={d} M={M}: fwd={fc:.6f} | " + " ".join(f"{n}={cos(a,b):.5f}" for n, a, b in zip(nm, gref, gour)), flush=True)


def bench(tag, d, seqs):
    print(f"\n#### {tag} d={d} fp32 (fwd+bwd)")
    print(f"  {'seq':>5} {'M':>8} | {'eager':>8} {'compile':>8} {'adaln_train':>11} {'fused3':>8} | {'f3/comp':>7} {'f3/at':>6}")
    for seq in seqs:
        M = 32 * seq
        x = torch.randn(M, d, device=DEVICE, requires_grad=True)
        cond = torch.randn(M, d, device=DEVICE, requires_grad=True)
        lnw = torch.randn(d, device=DEVICE, requires_grad=True)
        sw = (torch.randn(d, d, device=DEVICE) * d ** -0.5).requires_grad_()
        sb = (torch.randn(d, device=DEVICE) * 0.1).requires_grad_()
        bw = (torch.randn(d, d, device=DEVICE) * d ** -0.5).requires_grad_()
        dy = torch.randn(M, d, device=DEVICE)
        gtn = [x, cond, lnw, sw, sb, bw]

        def full(fn):
            for p in gtn: p.grad = None
            fn().backward(dy)
        compiled = torch.compile(ref)
        te = t(lambda: full(lambda: ref(x, cond, lnw, sw, sb, bw)), gtn)
        tc = t(lambda: full(lambda: compiled(x, cond, lnw, sw, sb, bw)), gtn)
        ta = t(lambda: full(lambda: adaln_train(x, cond, lnw, sw, sb, bw, eps, eps)), gtn)
        t3 = t(lambda: full(lambda: adaln_fused3_train(x, cond, lnw, sw, sb, bw, eps, eps)), gtn)
        print(f"  {seq:>5} {M:>8} | {te:>8.1f} {tc:>8.1f} {ta:>11.1f} {t3:>8.1f} | {tc/t3:>6.2f}x {ta/t3:>5.2f}x", flush=True)


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    print("=== grad correctness ===")
    verify(768, 32 * 512)
    verify(128, 32 * 4096)
    bench("TOKEN", 768, [384, 512, 768, 1024])
    bench("ATOM", 128, [2048, 4096, 8192])


if __name__ == "__main__":
    main()
