"""sm_100a inference core (tdit/b200_core.py) against the Triton gated2 core and an fp32 torch reference, then both timed (graph replay).
usage: run_b200.sh test_b200_core.py [--length 384] [--samples 5]"""
import argparse
import torch
from tdit.attn import attention_gated_in_place2, bias_descriptor
from tdit.b200_core import InfCore

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--samples", type=int, default=5)
p.add_argument("--blocks", type=int, default=2)
a = p.parse_args()
S, L, H, DH, NB = a.samples, a.length, 16, 48, a.blocks
D, M = H * DH, a.samples * a.length
torch.manual_seed(0)
dev = "cuda"
# logits in exp2 units, as the step produces them (q pre-scaled by sm_scale log2 e, bias by log2 e)
qkvg0 = torch.randn(M, 4 * D, device=dev)
qkvg0[:, :D] *= DH ** -0.5 * 1.4426950408889634
qkvg0 = qkvg0.bfloat16()
bias = (torch.randn(NB * H, L, L, device=dev) * 1.4426950408889634).bfloat16()
blk = NB - 1


def ref_fp32(qkvg):
    q, k, v, g = (qkvg[:, i * D:(i + 1) * D].float().view(S, L, H, DH).transpose(1, 2) for i in range(4))
    s = q @ k.transpose(-1, -2) + bias[blk * H:(blk + 1) * H].float()[None]
    o = torch.softmax(s * 0.6931471805599453, dim=-1) @ v                # exp2 units -> natural
    o = o * torch.sigmoid(g)
    return o.transpose(1, 2).reshape(M, D)


ref = ref_fp32(qkvg0)
t = qkvg0.clone()
q4, k4, v4, g4 = (t.view(S, L, 4 * D)[..., i * D:(i + 1) * D].unflatten(-1, (H, DH)) for i in range(4))
bd = bias_descriptor(bias)
attention_gated_in_place2(q4, k4, v4, g4, bd, blk)
tri = t[:, :D].float()
core = InfCore()
u = qkvg0.clone()
core(u, bias, blk, S, H)
torch.cuda.synchronize()
ours = u[:, :D].float()


def rel(x, y):
    return ((x - y).norm() / y.norm()).item()


print(f"S={S} L={L}: b200 vs fp32 {rel(ours, ref):.2e}   triton vs fp32 {rel(tri, ref):.2e}   b200 vs triton {rel(ours, tri):.2e}   "
      f"max|b200-fp32| {(ours - ref).abs().max().item():.3e}   k|v|g untouched {torch.equal(u[:, D:], qkvg0[:, D:])}")


def gtime(fn, reps=20, rounds=5):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / reps)
    return sorted(ts)[len(ts) // 2]


print(f"   triton gated2 {gtime(lambda: attention_gated_in_place2(q4, k4, v4, g4, bd, blk)):.1f} us   "
      f"b200 sm_100a {gtime(lambda: core(u, bias, blk, S, H)):.1f} us")
