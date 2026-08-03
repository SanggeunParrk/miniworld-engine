"""Pick best CUTLASS config per shape for gemm_nt (dgrad/fwd) AND gemm_tn (wgrad), vs cuBLAS.
Emits a python dict literal (NT_BEST / TN_BEST) to paste into the orchestrator."""
import time, torch
import ct_gemm_ext as g
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm()*b.norm()+1e-20)).item()


def bench(fn, it=200, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e6


NCFG = g.num_cfg()
A = 48  # real batch multiplier: leading dim M = A * L (was tuned at M=L — 48x too small)
# ---- NT shapes: D = A@B^T, A:(M,K) B:(N,K). (M, K, N) ----  [M = 48*L, key stays L]
nt_shapes = {}
for L in (2048, 4096, 8192):
    M = A * L
    nt_shapes[("atom", L, "expand")] = (M, 128, 512)   # x@wcat^T : (M,128)->(M,512)
    nt_shapes[("atom", L, "squeeze")] = (M, 256, 128)  # h@ws^T
    nt_shapes[("atom", L, "scale")] = (M, 384, 128)    # cond@wsc^T
    nt_shapes[("atom", L, "dx")] = (M, 512, 128)       # dab@wcat
    nt_shapes[("atom", L, "dcond")] = (M, 128, 384)    # dscale@wsc
    nt_shapes[("atom", L, "dh")] = (M, 128, 256)       # dout@Ws (dgrad of squeeze)
for L in (384, 512, 768, 1024):
    M = A * L
    nt_shapes[("token", L, "expand")] = (M, 768, 3072)
    nt_shapes[("token", L, "squeeze")] = (M, 1536, 768)
    nt_shapes[("token", L, "scale")] = (M, 384, 768)
    nt_shapes[("token", L, "dx")] = (M, 3072, 768)
    nt_shapes[("token", L, "dcond")] = (M, 768, 384)
    nt_shapes[("token", L, "dh")] = (M, 768, 1536)     # dout@Ws (dgrad of squeeze)
# ---- TN shapes: D = A^T@B, A:(Mc,N) B:(Mc,K) -> (N,K). (Mc, N, K) ----  [Mc = 48*L]
tn_shapes = {}
for L in (2048, 4096, 8192):
    Mc = A * L
    tn_shapes[("atom", L, "dWcat")] = (Mc, 512, 128)   # dab^T@x : (Mc,512)^T@(Mc,128)
    tn_shapes[("atom", L, "dWs")] = (Mc, 128, 256)     # dout^T@h
    tn_shapes[("atom", L, "dWsc")] = (Mc, 128, 384)    # dscale^T@cond
for L in (384, 512, 768, 1024):
    Mc = A * L
    tn_shapes[("token", L, "dWcat")] = (Mc, 3072, 768)
    tn_shapes[("token", L, "dWs")] = (Mc, 768, 1536)
    tn_shapes[("token", L, "dWsc")] = (Mc, 768, 384)


def sweep(shapes, fn, mkref, mkargs, label):
    print(f"\n=== {label} (best cfg vs cuBLAS) ===")
    print(f"{'shape':30s} {'cfg':>4} {'cut_us':>8} {'cub_us':>8} {'ratio':>7} {'cos':>8}")
    best = {}
    for name, dims in shapes.items():
        A, B, ref = mkref(dims)
        bu, bc, bcos = 1e18, -1, 0
        for c in range(NCFG):
            try: D = fn(A, B, c)
            except Exception: continue
            cc = cos(D, ref)
            if cc < 0.99: continue
            t = bench(lambda: fn(A, B, c))
            if t < bu: bu, bc, bcos = t, c, cc
        tc = bench(lambda: mkargs(A, B))
        r = bu/tc if bc >= 0 else float("nan")
        print(f"{str(name):30s} {bc:>4} {bu:>8.2f} {tc:>8.2f} {r:>7.3f} {bcos:>8.5f}")
        best[name] = bc
    return best


def nt_ref(d):
    M, K, N = d; A = torch.randn(M, K, device=dev); B = torch.randn(N, K, device=dev)
    return A, B, A @ B.t()
def tn_ref(d):
    Mc, N, K = d; A = torch.randn(Mc, N, device=dev); B = torch.randn(Mc, K, device=dev)
    return A, B, A.t() @ B

nt_best = sweep(nt_shapes, g.gemm_nt, nt_ref, lambda A, B: A @ B.t(), "NT  D=A@B^T")
tn_best = sweep(tn_shapes, g.gemm_tn, tn_ref, lambda A, B: A.t() @ B, "TN  D=A^T@B (wgrad)")
import json
ntd = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in nt_best.items()}
tnd = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in tn_best.items()}
print("\nNT_BEST =", ntd)
print("TN_BEST =", tnd)
with open("/home/psk6950/miniworld-engine/_ct_cutlass/gemm_cfgs.json", "w") as f:
    json.dump({"NT": ntd, "TN": tnd}, f)
print("GEMM PICK DONE")
