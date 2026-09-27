"""Token DiT TRAINING regime, one block: forward + backward latency (CUDA events, eager) for each implementation.

engine modules.dit.DiTBlock (AugmentedAttentionPairBias + ConditionedTransition), d 768 / cond 384 / pair 128 / 16 heads,
qk_norm on, A = 48 augments sharing one pair, fp32 parameters. Variants:
  pytorch        ImplementationType.PYTORCH, fp32 activations (torch "medium" matmul precision, as the H100 training runs)
  engine         ImplementationType.MINIWORLD, fp32
  engine-bf16    ImplementationType.MINIWORLD with compute_dtype=bf16 (the attention core in bf16)
  engine-b200    engine-bf16 with the core on the sm_100a augattn_sm100 kernels (attn_fwd2 + attn_dqb + attn_dkv)
  tdit-b200      tdit/train_b200.py: the fused training block (bf16 GEMM operands, fp32 residual, sm_100a core)
Gradient check: every variant's input / parameter grads against the pytorch fp32 run (rel error).

  run_b200.sh train_block.py --length 384 [--augment 48] [--variants pytorch engine engine-bf16]
"""
import argparse
import statistics
import torch
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--augment", type=int, default=48)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--variants", nargs="+", default=["pytorch", "engine", "engine-bf16"])
p.add_argument("--gradreport", action="store_true", help="per-tensor grad rel error of every variant vs the first")
p.add_argument("--profile", action="store_true", help="per-kernel breakdown of one forward+backward of the last variant")
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")     # the H100 training runs regime (prof_block.py)
dev, L, A = "cuda", a.length, a.augment
torch.manual_seed(0)


def make(impl):
    torch.manual_seed(0)
    m = DiTBlock(768, 384, 128, 16, 2, use_qk_norm=True, implementation=impl).to(dev)
    with torch.no_grad():                                           # non-zero everywhere (zero-init outputs would hide work)
        for prm in m.parameters():
            if prm.ndim == 2:
                prm.normal_(std=prm.shape[1] ** -0.5 * 0.5)
            elif prm.numel() > 1:
                prm.add_(torch.randn_like(prm) * 0.1)
    return m


g = torch.Generator(device=dev).manual_seed(1)
single0 = torch.randn(A, 1, L, 768, device=dev, generator=g)
cond0 = torch.randn(A, 1, L, 384, device=dev, generator=g)
pair0 = torch.randn(1, L, L, 128, device=dev, generator=g)
dout = torch.randn(A, 1, L, 768, device=dev, generator=g)


_ORIG_CORE = None


def use_b200_core(on):
    """Route the bf16 attention core of AugmentedAttentionPairBias to augattn_sm100 (no mask, 16 x 48, B == 1, L % 128 == 0)."""
    global _ORIG_CORE
    import sys
    from pathlib import Path
    from miniworld_engine.modules.augmented_attention.module import AugmentedAttentionPairBias as AAP
    if _ORIG_CORE is None:
        _ORIG_CORE = AAP._kernel_attention_pair_bias
    if not on:
        AAP._kernel_attention_pair_bias = _ORIG_CORE
        return
    aug = str(Path(__file__).resolve().parents[1] / "augattn_sm100")
    if aug not in sys.path:
        sys.path.insert(0, aug)
    import os
    cwd = os.getcwd(); os.chdir(aug)
    from attn_op import augattn, kernels
    kernels()                                                       # load the cubins (paths relative to augattn_sm100)
    os.chdir(cwd)

    def core(self, query, key, value, bias, mask=None, compute_dtype=None):
        A, B, L, h, d = query.shape
        if compute_dtype is torch.bfloat16 and mask is None and B == 1 and (h, d) == (16, 48) and L % 128 == 0:
            b = bias[0].permute(2, 0, 1).to(torch.bfloat16).contiguous()        # [1, L, L, 16] -> head-major [16, L, L]
            return augattn(*(t.to(torch.bfloat16).contiguous() for t in (query, key, value)), b)
        return _ORIG_CORE(self, query, key, value, bias, mask, compute_dtype)
    AAP._kernel_attention_pair_bias = core


def run_variant(name):
    use_b200_core(name == "engine-b200")
    impl = ImplementationType.PYTORCH if name == "pytorch" else ImplementationType.MINIWORLD
    if name == "tdit-b200":
        from tdit.train_b200 import block_forward
    kw = {"compute_dtype": torch.bfloat16} if name in ("engine-bf16", "engine-b200") else {}
    m = make(impl)
    single, cond, pair = (t.clone().requires_grad_(True) for t in (single0, cond0, pair0))

    def step():
        for t in (single, cond, pair):
            t.grad = None
        m.zero_grad(set_to_none=True)
        out = block_forward(m, single, cond, pair) if name == "tdit-b200" else m(single, cond, pair, None, **kw)
        out.float().backward(dout)
        return out

    def fwd():
        with torch.no_grad():
            return block_forward(m, single, cond, pair) if name == "tdit-b200" else m(single, cond, pair, None, **kw)

    def timed(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts)

    out = step()
    grads = {"out": out.detach().float(), "single": single.grad.float(), "cond": cond.grad.float(), "pair": pair.grad.float()}
    grads.update({k: v.grad.float() for k, v in m.named_parameters() if v.grad is not None})
    t_train, t_fwd = timed(step), timed(fwd)
    if a.profile and name == a.variants[-1]:
        import collections
        from torch.profiler import profile, ProfilerActivity
        step(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step(); torch.cuda.synchronize()
        agg = collections.defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            if e.device_type.name == "CUDA":
                agg[e.name][0] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
                agg[e.name][1] += 1
        tot = sum(v[0] for v in agg.values())
        print(f"  --- {name}: kernel time {tot:.0f} us over {sum(v[1] for v in agg.values())} launches")
        for k, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]:
            print(f"    {t:8.1f} us  x{n:<3d} {k[:110]}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return t_fwd, t_train, grads, peak


ref = None
print(f"[token DiT training] one block, A={A}, L={L}, qk_norm, fp32 params")
for name in a.variants:
    t_fwd, t_train, gr, peak = run_variant(name)
    if ref is None:
        ref = gr
    errs = {k: ((gr[k] - ref[k]).norm() / ref[k].norm().clamp_min(1e-30)).item() for k in ref if k in gr}
    worst = max(errs.values())
    if a.gradreport and name != a.variants[0]:
        print("   ", "  ".join(f"{k}:{v:.1e}" for k, v in sorted(errs.items(), key=lambda kv: -kv[1])[:12]))
    print(f"  {name:14s} forward {t_fwd:9.1f} us   forward+backward {t_train:9.1f} us   peak {peak:5.2f} GB   "
          f"max grad rel vs {a.variants[0]} {worst:.2e}", flush=True)
