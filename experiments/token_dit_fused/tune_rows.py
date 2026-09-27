"""Sweep (ROWS, num_warps) for the training row kernels at one shape; prints us and effective GB/s per config.
usage: run_b200.sh tune_rows.py --length 768 [--augment 48]"""
import argparse
import torch
import triton
from tdit import train_kernels as T

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); p.add_argument("--augment", type=int, default=48)
p.add_argument("--only-pair", action="store_true")
a = p.parse_args()
L, A = a.length, a.augment
M, D, dev, bf = A * L, 768, "cuda", torch.bfloat16
f32 = lambda *s: torch.randn(*s, device=dev)
b16 = lambda *s: torch.randn(*s, device=dev).to(bf)
x, x1, dout, O = f32(M, D), f32(M, D), f32(M, D), f32(M, D)
G, Gg, qkvg, dG, dGg, dqkvg = b16(M, 4 * D), b16(M, 2 * D), b16(M, 4 * D), b16(M, 4 * D), b16(M, 2 * D), b16(M, 4 * D)
y, z, xa, xt, dxt, dxa, dog, og = (b16(M, D) for _ in range(8))
st, st1 = torch.rand(M, 2, device=dev) + 0.5, torch.rand(M, 2, device=dev) + 0.5
bs, acc = f32(D), torch.zeros(4096, device=dev)
out, dx1, dx = f32(M, D), f32(M, D), f32(M, D)
dy, dz, dob = b16(M, D), b16(M, D), b16(M, D)
dd = f32(A, 16, L)
DQ, DK, DV = f32(M, D), f32(M, D), f32(M, D)
rqk = torch.rand(M, 32, device=dev) + 0.5
w48 = f32(48)
qn, kn, vc = b16(M, D), b16(M, D), b16(M, D)


def t_us(fn):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / 20


K = {
    "adaln_a": (lambda R, W: T._adaln_a[T.grid(M, R)](x, G, G.stride(0), bs, xa, st, M, N=D, BN=1024, ROWS=R, eps=1e-5, num_warps=W), M * D * (4 + 4 + 2)),
    "res_adaln_b": (lambda R, W: T._res_adaln_b[T.grid(M, R)](x, y, Gg, Gg.stride(0), bs, G, G.stride(0), bs, x1, xt, st1, M, N=D, BN=1024, ROWS=R, eps=1e-5, num_warps=W), M * D * (4 + 2 + 2 + 4 + 4 + 2)),
    "res_c": (lambda R, W: T._res_c[T.grid(M, R)](x1, z, Gg, Gg.stride(0), bs, out, M, N=D, BN=1024, ROWS=R, num_warps=W), M * D * (4 + 2 + 2 + 4)),
    "res_c_bwd": (lambda R, W: T._res_c_bwd[T.grid(M, R)](dout, z, Gg, Gg.stride(0), bs, dz, dGg, dGg.stride(0), acc, M, N=D, BN=1024, ROWS=R, num_warps=W), M * D * (4 + 2 + 2 + 2 + 2)),
    "res_adaln_b_bwd": (lambda R, W: T._res_adaln_b_bwd[T.grid(M, R)](dout, dxt, x1, st1, G, G.stride(0), bs, Gg, Gg.stride(0), bs, y, dx1, dy, dG, dG.stride(0), dGg, dGg.stride(0), acc, acc[768:], M, N=D, BN=1024, ROWS=R, num_warps=W), M * D * (4 + 2 + 4 + 2 + 2 + 2 + 4 + 2 + 4 + 2)),
    "adaln_a_bwd": (lambda R, W: T._adaln_a_bwd[T.grid(M, R)](dxa, x, st, G, G.stride(0), bs, dx1, dx, dG, dG.stride(0), acc, M, N=D, BN=1024, ROWS=R, num_warps=W), M * D * (2 + 4 + 2 + 4 + 4 + 4)),
    "gate_o": (lambda R, W: T._gate_o[T.grid(M, R)](O, qkvg, og, M, ROWS=R, num_warps=W), M * D * (4 + 2 + 2)),
    "gate_o_bwd": (lambda R, W: T._gate_o_bwd[T.grid(M, R)](dog, O, qkvg, dob, dd, dqkvg, L, M, ROWS=R, num_warps=W), M * D * (2 + 4 + 2 + 2 + 2)),
    "qknorm": (lambda R, W: T._qknorm[T.grid(M, R)](qkvg, w48, w48, qn, kn, vc, rqk, M, 1e-6, 1e-6, ROWS=R, num_warps=W), M * D * (6 + 6)),
    "qknorm_bwd": (lambda R, W: T._qknorm_bwd[T.grid(M, R)](DQ, DK, DV, qkvg, rqk, w48, w48, dqkvg, acc, acc[128:], M, ROWS=R, num_warps=W), M * D * (12 + 4 + 6)),
}
R2 = L * L
pair, pst, wp, wb = f32(R2, 128), torch.rand(R2, 2, device=dev) + 0.5, f32(128), f32(16, 128)
bias_hm, DB, dpair = b16(16, R2), f32(16, R2), f32(R2, 128)
if a.only_pair:
    K = {}
for RP in (32, 64, 128):
    K[f"pair_fwd R{RP}"] = (lambda R, W, RP=RP: T._pair_bias_fwd[(triton.cdiv(R2, RP),)](pair, wp, wb, bias_hm, pst, R2, 1e-5, ROWS=RP, num_warps=W), R2 * (512 + 32 + 8))
    ch = triton.cdiv(R2, RP); npb = min(ch, 592); nit = triton.cdiv(ch, npb)
    K[f"pair_bwd R{RP}"] = (lambda R, W, RP=RP, nit=nit, ch=ch: T._pair_bias_bwd[(triton.cdiv(ch, nit),)](DB, pair, pst, wp, wb, dpair, acc, R2, nit, ROWS=RP, num_warps=W), R2 * (64 + 512 + 8 + 512))
for name, (fn, byt) in K.items():
    res = []
    for R in ((1,) if name.startswith("pair") else (1, 2, 4, 8, 16)):
        for W in (2, 4, 8, 16):
            try:
                us = t_us(lambda: fn(R, W))
                res.append((us, R, W))
            except Exception:  # noqa: BLE001 -- too many registers / shared memory for this config
                pass
    res.sort()
    cur = [r for r in res if (r[1], r[2]) in ((8, 4), (4, 4))]
    print(f"{name:16s} best {res[0][0]:7.1f} us (ROWS {res[0][1]}, warps {res[0][2]}) {byt / res[0][0] / 1e3:6.0f} GB/s   "
          f"current {', '.join(f'R{r[1]}w{r[2]} {r[0]:.1f}' for r in cur)}", flush=True)
