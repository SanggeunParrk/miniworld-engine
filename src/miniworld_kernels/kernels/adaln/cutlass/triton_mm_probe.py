"""Triton autotuned TF32 matmul vs cuBLAS at adaln shapes — establish achievable ceiling + best config."""
from __future__ import annotations
import torch, triton, triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BM': bm, 'BN': bn, 'BK': bk, 'GM': 8}, num_stages=ns, num_warps=nw)
        for bm in (64, 128, 256) for bn in (64, 128, 256) for bk in (32, 64)
        for ns in (3, 4, 5) for nw in (4, 8)
        if bm * bn <= 256 * 128
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _mm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM); nn = tl.cdiv(N, BN)
    ng = GM * nn; g = pid // ng; fm = g * GM
    gsize = min(nm - fm, GM)
    pm = fm + ((pid % ng) % gsize); pn = (pid % ng) // gsize
    rm = (pm * BM + tl.arange(0, BM)) % M
    rn = (pn * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    a = A + (rm[:, None] * sam + rk[None, :] * sak)
    b = B + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        am = tl.load(a, mask=rk[None, :] < K - k * BK, other=0.0)
        bm = tl.load(b, mask=rk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(am, bm, acc, allow_tf32=True)
        a += BK * sak; b += BK * sbk
    c = C + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def tmm(A, B):
    M, K = A.shape; N = B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BM']) * triton.cdiv(N, meta['BN']),)
    _mm[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1))
    return C


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8]); return m * 1000


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    for (M, K, N) in [(32768, 768, 768), (32768, 768, 1536), (262144, 128, 256)]:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        Bt = torch.randn(N, K, device="cuda", dtype=torch.float32) * K ** -0.5
        B = Bt.t().contiguous()  # (K,N) for triton mm
        gf = 2 * M * K * N / 1e9
        ref = A @ B
        C = tmm(A, B)
        c = torch.nn.functional.cosine_similarity(ref.flatten(), C.flatten(), dim=0).item()
        tcub = t(lambda: A @ B)
        ttri = t(lambda: tmm(A, B))
        best = _mm.best_config
        print(f"\nM={M} K={K} N={N} ({gf:.0f}GF): cuBLAS={tcub:.1f}us ({gf/(tcub/1e6)/1e3:.0f}TF/s)  "
              f"triton={ttri:.1f}us ({gf/(ttri/1e6)/1e3:.0f}TF/s) cuBLAS/tri={tcub/ttri:.2f}x cos={c:.5f}", flush=True)
        print(f"  triton best: {best}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
