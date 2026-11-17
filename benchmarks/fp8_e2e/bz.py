"""9c90 mode table: per trunk seed, bf16 and FP8 ipTM, RMSD of each to the bf16 trunk-seed-0 structure (mode A reference) and to the FP8-vs-bf16 same seed.  usage: bz.py DIR TARGET MODE"""
import sys, torch, itertools
d, t, m = sys.argv[1:4]
def kabsch(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, s, vt = torch.linalg.svd(a.T @ b)
    dd = torch.sign(torch.linalg.det(u @ vt)); D = torch.diag(torch.tensor([1, 1, dd]))
    return ((a @ (u @ D @ vt) - b) ** 2).sum(-1).mean().sqrt().item()
def ld(n):
    x = torch.load(f"{d}/{t}.{n}.pt"); return x["coords"].reshape(-1, 3), float(x["iptm"].float().mean())
seeds = list(range(int(sys.argv[4]) if len(sys.argv) > 4 else 12)); B = {s: ld(f"bf16_t{s}") for s in seeds}; F = {s: ld(f"{m}_t{s}") for s in seeds if __import__("os").path.exists(f"{d}/{t}.{m}_t{s}.pt")}
# modes: cluster bf16 structures greedily at 3 A
cl = []
for s in seeds:
    for c in cl:
        if kabsch(B[s][0], B[c[0]][0]) < 3.0: c.append(s); break
    else: cl.append([s])
print("bf16 clusters (3 A, greedy) by trunk seed:", cl)
ref = {i: c[0] for i, c in enumerate(cl)}
def assign(x):
    r = [kabsch(x, B[ref[i]][0]) for i in ref]; i = min(range(len(r)), key=lambda k: r[k]); return i, r[i]
print("seed | bf16 ipTM, cluster, rmsd-to-cluster-ref | fp8 ipTM, cluster, rmsd-to-ref | same-seed rmsd | flip(cluster differs)")
flips = 0; n = 0
for s in seeds:
    cb, rb = assign(B[s][0])
    if s in F:
        cf, rf = assign(F[s][0]); n += 1; flips += cb != cf
        print(f"{s:2d} | {B[s][1]:.3f} c{cb} {rb:.2f} | {F[s][1]:.3f} c{cf} {rf:.2f} | {kabsch(F[s][0], B[s][0]):.2f} | {'FLIP' if cb != cf else ''}")
    else: print(f"{s:2d} | {B[s][1]:.3f} c{cb} {rb:.2f} | -")
print(f"fp8 flips {flips}/{n}; bf16 seeds per cluster: {[len(c) for c in cl]}; fp8 per cluster: {[sum(1 for s in F if assign(F[s][0])[0] == i) for i in ref]}")
