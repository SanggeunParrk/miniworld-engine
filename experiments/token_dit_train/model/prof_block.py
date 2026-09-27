"""P0 for the token DiT backward: where does one v1 token DiT block spend its training time?

Exactly the phase-2 regime: team_gm DiffusionTransformer (engine AugmentedAttentionPairBias + ConditionedTransition),
d 768 / cond 384 / pair 128 / 16 heads, qk_norm on, per-block gradient checkpointing (the forward is recomputed in
the backward), fp32 params and activations (Fabric's default 32-true) with torch matmul precision "medium", A = 48
augments. Reports per-block fwd and fwd+bwd time (CUDA events, eager) and the per-kernel breakdown of one fwd+bwd.

  python prof_block.py --length 384 [--blocks 4] [--dtype fp32|bf16]
"""
import argparse, collections, statistics, sys
import torch
from torch.profiler import profile, ProfilerActivity
from team_gm.modules import DiffusionTransformer
from team_gm.modules.exceptions import ImplementationType

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--blocks", type=int, default=4)
p.add_argument("--augment", type=int, default=48)
p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
p.add_argument("--no-ckpt", action="store_true")
p.add_argument("--hoist", action="store_true")
p.add_argument("--core", default="model")
p.add_argument("--adaln", choices=("auto", "cublas", "fused"), default="auto")
a = p.parse_args()
torch.set_float32_matmul_precision("medium")
from miniworld_engine.kernels.adaln.triton import training as _adaln_tr
_adaln_tr.set_forward_mode(a.adaln)
dev, L, A, NB = "cuda", a.length, a.augment, a.blocks
dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
torch.manual_seed(0)
cfg = DiffusionTransformer.Config(d_single=768, d_cond=384, d_pair=128, n_head=16, n_block=NB, use_qk_norm=True,
                                  n_checkpoint_segments=None if a.no_ckpt else NB,
                                  implementation=ImplementationType.MINIWORLD_ENGINE, hoist_pair_bias=a.hoist, attention_core_dtype=a.core)
m = DiffusionTransformer(cfg).to(dev).to(dt)
with torch.no_grad():                                               # non-zero everywhere (the zero-init outs hide work)
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[1] ** -0.5 * 0.5)
        elif prm.numel() > 1: prm.add_(torch.randn_like(prm) * 0.1)
single = torch.randn(A, 1, L, 768, device=dev, dtype=dt, requires_grad=True)
cond = torch.randn(A, 1, L, 384, device=dev, dtype=dt, requires_grad=True)
pair = torch.randn(1, L, L, 128, device=dev, dtype=dt, requires_grad=True)


def step(bwd=True):
    out = m(single, cond, pair, None)
    if bwd:
        out.float().square().mean().backward()
    return out


def timed(fn, reps=5):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)


with torch.no_grad():
    t_f = timed(lambda: step(False))
t_fb = timed(lambda: step(True))
print(f"token DiT block, L={L} A={A} {a.dtype} (matmul 'medium'), qk_norm, ckpt={'off' if a.no_ckpt else 'per block'}, hoist={int(a.hoist)}, adaln fwd={a.adaln}, core={a.core}", flush=True)
print(f"  per block: fwd {t_f / NB:8.1f} us   fwd+bwd {t_fb / NB:8.1f} us   (x24 = {t_fb * 24 / NB / 1e3:6.1f} ms)", flush=True)

torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    step(True)
    torch.cuda.synchronize()
per = collections.defaultdict(lambda: [0.0, 0])
for ev in prof.key_averages():
    t = getattr(ev, "self_device_time_total", 0)
    if t > 0:
        per[ev.key][0] += t / NB; per[ev.key][1] += ev.count
tot = sum(v[0] for v in per.values())
print(f"  kernel time {tot:8.1f} us/block; top kernels (us/block, calls/step):", flush=True)
for k, (t, c) in sorted(per.items(), key=lambda x: -x[1][0])[:30]:
    print(f"  {t:8.1f}  {100 * t / tot:5.1f}%  x{c:<4d} {k[:110]}", flush=True)
