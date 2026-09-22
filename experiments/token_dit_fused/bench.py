"""Token DiT stack (24 x AF3 Alg. 23 blocks), one sampling step, S samples: accuracy against fp32 and CUDA-graph time.

  engine MINIWORLD           24 x miniworld_engine.modules.dit.DiTBlock, bf16, as the engine runs it today
  Anthropic composed         the same weights through Anthropic's parts, the way their kits assemble a block
                             (torch GEMMs, q|k|v|g fused, ln_proj.pair_bias, apb_views with the gate, dit_fast row kernels,
                             fp32 residual, conditioning shared by the samples); pair bias computed per block per step
  Anthropic composed, hoisted   the same with every block's pair bias computed once per sample, as their samplers do
  fused (this package)       FusedTokenDiT.step, pair bias hoisted by FusedTokenDiT.hoist

The one-time per-sample pair-bias cost of the two hoisting rows is timed separately.

  python bench.py --length 384 [--blocks 24] [--samples 5] [--save results/L384.json]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--blocks", type=int, default=24)
p.add_argument("--samples", type=int, default=5)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--no-anthropic", action="store_true")
p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
               help="activation / weight dtype of every fast path; fp32 runs its GEMMs in TF32, MiniWorld's v1 diffusion recipe")
p.add_argument("--save", default="")
a = p.parse_args()
L, S, NB, dev = a.length, a.samples, a.blocks, "cuda"
bf = torch.float32 if a.dtype == "fp32" else torch.bfloat16      # the path dtype (named bf for history: it is fp32 with --dtype fp32)
DS, DC, DP, H, D = 768, 384, 128, 16, 48
torch.backends.cuda.matmul.allow_tf32 = False

from miniworld_engine.modules.dit import DiTBlock                 # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType  # noqa: E402
from tdit import FusedTokenDiT                                      # noqa: E402


def _graph(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g.replay


def time_us(fn, graph=True):
    run = _graph(fn) if graph else fn
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    out = []
    for _ in range(a.rounds):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(a.reps):
            run()
        en.record()
        torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / a.reps)
    return statistics.median(out)


def rel(got, want):
    got, want = got.float(), want.float()
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


# ------------------------------------------------------------------ one set of weights, three implementations
torch.manual_seed(0)
ref_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                 for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in ref_blocks.parameters():                  # to_out / squeeze / to_bias are zero-init: give every weight a value
        if prm.ndim == 2:
            prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1:
            prm.add_(torch.randn_like(prm) * 0.1)
    for blk in ref_blocks:                                # keep a 24-deep residual stream bounded, as trained weights do
        blk.attention.to_out.weight.mul_(0.25)
        blk.transition.squeeze.weight.mul_(0.25)
eng_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.MINIWORLD)
                                 for _ in range(NB)).to(dev)
eng_blocks.load_state_dict(ref_blocks.state_dict())
eng_blocks = eng_blocks.to(bf).eval()
bf_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                for _ in range(NB)).to(dev)
bf_blocks.load_state_dict(ref_blocks.state_dict())
bf_blocks = bf_blocks.to(bf).eval()                        # the source of the bf16 weights every fast path sees

single = torch.randn(S, 1, L, DS, device=dev)
cond1 = torch.randn(1, 1, L, DC, device=dev)
cond = cond1.expand(S, 1, L, DC).contiguous()             # at a sampling step every sample shares the conditioning
pair = torch.randn(1, L, L, DP, device=dev)
s_bf, c_bf, z_bf = single.to(bf), cond.to(bf), pair.to(bf)

with torch.no_grad():
    x = single
    for blk in ref_blocks:
        x = blk(x, cond, pair)
    ref = x
if a.dtype == "fp32":
    torch.backends.cuda.matmul.allow_tf32 = True                   # the reference above is IEEE fp32; the fast paths run TF32
rec = {"length": L, "samples": S, "blocks": NB, "dtype": a.dtype, "device": torch.cuda.get_device_name(), "step_us": {},
       "rel_rms": {}, "once_per_sample_us": {}}


def report(name, us, out=None):
    rec["step_us"][name] = us
    e = "" if out is None else f"  rel_rms vs fp32 {rel(out, ref):.2e}"
    if out is not None:
        rec["rel_rms"][name] = rel(out, ref)
    print(f"  {name:<34s} {us:10.1f} us/step  {us / NB:7.1f} us/block{e}", flush=True)


print(f"\n[token DiT] {NB} blocks, S={S}, L={L}, {a.dtype}{' (TF32 GEMMs, IEEE reference)' if a.dtype == 'fp32' else ''}, one sampling step", flush=True)

with torch.no_grad():
    def engine():
        x = s_bf
        for blk in eng_blocks:
            x = blk(x, c_bf, z_bf)
        return x
    out = engine().clone()
    report("engine MINIWORLD", time_us(engine), out)

    fused = FusedTokenDiT(bf_blocks, dtype=bf)
    bias = fused.hoist(z_bf)
    out = fused.step(s_bf, c_bf, bias).clone()
    report("fused v7 (v6 + per-shape GEMM choice)", time_us(lambda: fused.step(s_bf, c_bf, bias)), out)
    print("    GEMM choices:", {k: v for k, v in fused._mm_cfg.items()}, flush=True)
    if fused.gated_gemm:
        fused.gated_gemm = False
        out = fused.step(s_bf, c_bf, bias).clone()
        report("fused v5 (cuBLAS expand + swiglu_rows)", time_us(lambda: fused.step(s_bf, c_bf, bias)), out)
        fused.gated_gemm = True
    fused.core = "gated"
    out = fused.step(s_bf, c_bf, bias).clone()
    report("fused v4 (v1 core, pre-scaled)", time_us(lambda: fused.step(s_bf, c_bf, bias)), out)
    fused.core = "gated2"
if a.dtype == "bf16":
  with torch.no_grad():
    unscaled = FusedTokenDiT(bf_blocks, prescale=False, core="gated")
    bias_u = unscaled.hoist(z_bf)
    out = unscaled.step(s_bf, c_bf, bias_u).clone()
    report("fused v3, logits not pre-scaled", time_us(lambda: unscaled.step(s_bf, c_bf, bias_u)), out)
    unscaled.core = "engine"
    out = unscaled.step(s_bf, c_bf, bias_u).clone()
    report("fused v2, engine attention core", time_us(lambda: unscaled.step(s_bf, c_bf, bias_u)), out)
    del unscaled, bias_u
    v1 = FusedTokenDiT(bf_blocks, prescale=False, core="gated")
    bias_v1 = v1.hoist(z_bf)
    out = v1.step_v1(s_bf, c_bf, bias_v1).clone()
    report("fused v1 (AdaLN in Triton GEMM)", time_us(lambda: v1.step_v1(s_bf, c_bf, bias_v1)), out)
    del v1, bias_v1
with torch.no_grad():
    rec["once_per_sample_us"]["fused hoist (all blocks' pair bias)"] = time_us(lambda: fused.hoist(z_bf))

# ------------------------------------------------------------------ Anthropic composition over the same bf16 weights
if not a.no_anthropic:
    from opt_core.kernels import ln_proj
    from opt_core.kernels.apb.fpf_apb.apb_triton import apb_views
    from opt_core.kernels.apb.ditfast import kernels as dk
    W = lambda t: t.detach().to(bf).contiguous()             # noqa: E731
    Wf = lambda t: t.detach().to(bf).float().contiguous()    # noqa: E731
    packs = []
    for blk in bf_blocks:
        at, tr = blk.attention, blk.transition
        packs.append(dict(
            lnc_a=Wf(at.ada_ln_in.ln_cond.weight), ws_a=W(at.ada_ln_in.to_scale.weight), bs_a=W(at.ada_ln_in.to_scale.bias),
            wb_a=W(at.ada_ln_in.to_bias.weight),
            wqkvg=torch.cat([W(at.to_query.weight), W(at.to_key.weight), W(at.to_value.weight), W(at.to_gate.weight)], 0),
            bqkvg=torch.cat([W(at.to_query.bias), torch.zeros(3 * DS, device=dev, dtype=bf)]),
            wo=W(at.to_out.weight), wsc_a=W(at.to_scale.weight), bsc_a=W(at.to_scale.bias),
            lnc_t=Wf(tr.ada_ln_in.ln_cond.weight), ws_t=W(tr.ada_ln_in.to_scale.weight), bs_t=W(tr.ada_ln_in.to_scale.bias),
            wb_t=W(tr.ada_ln_in.to_bias.weight), wab=torch.cat([W(tr.expand_a.weight), W(tr.expand_b.weight)], 0),
            wsq=W(tr.squeeze.weight), wsc_t=W(tr.to_scale.weight), bsc_t=W(tr.to_scale.bias),
            packed=ln_proj.pack_pair_bias_weights(Wf(at.ln_pair.weight), None, W(at.to_bias.weight), 1e-5, dev)))
    M = S * L
    res = torch.empty(M, DS, device=dev)
    c = cond1.reshape(-1, DC).to(bf)

    def anth_hoist():
        return [ln_proj.pair_bias(z_bf, pk["packed"], out_layout="bhij", out_dtype=bf)[0] for pk in packs]

    def anthropic(biases=None):
        res.copy_(s_bf.reshape(M, DS))
        for b, pk in enumerate(packs):
            cn = F.layer_norm(c.float(), (DC,), pk["lnc_a"], None, 1e-5).to(bf)
            xa = dk.adaln(res, F.linear(cn, pk["ws_a"], pk["bs_a"]), F.linear(cn, pk["wb_a"]), bf)
            qkvg = F.linear(xa, pk["wqkvg"], pk["bqkvg"]).view(S, L, 4 * DS)
            q, k, v, g = (qkvg[:, :, i * DS:(i + 1) * DS].unflatten(2, (H, D)) for i in range(4))
            bias_b = biases[b] if biases is not None else ln_proj.pair_bias(z_bf, pk["packed"], out_layout="bhij", out_dtype=bf)[0]
            o = apb_views(q, k, v, bias_b, g, scale=D ** -0.5)
            o2 = F.linear(o.reshape(M, DS), pk["wo"])
            gl_a = F.linear(c, pk["wsc_a"], pk["bsc_a"])
            cn_t = F.layer_norm(c.float(), (DC,), pk["lnc_t"], None, 1e-5).to(bf)
            xt = dk.resgate_adaln(gl_a, o2, res, F.linear(cn_t, pk["ws_t"], pk["bs_t"]), F.linear(cn_t, pk["wb_t"]), bf)
            t = F.linear(dk.swiglu(F.linear(xt, pk["wab"]), bf), pk["wsq"])
            dk.resgate(F.linear(c, pk["wsc_t"], pk["bsc_t"]), t, res, out=res)
        return res.view(S, 1, L, DS)

    with torch.no_grad():
        out = anthropic().clone()
        report("Anthropic composed", time_us(anthropic), out)
        biases = anth_hoist()
        out = anthropic(biases).clone()
        report("Anthropic composed, hoisted", time_us(lambda: anthropic(biases)), out)
        rec["once_per_sample_us"]["Anthropic hoist (24 x ln_proj.pair_bias)"] = time_us(anth_hoist)

for k, v in rec["once_per_sample_us"].items():
    print(f"  once per sample: {k:<44s} {v:9.1f} us", flush=True)
if a.save:
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    Path(a.save).write_text(json.dumps(rec, indent=1) + "\n")
    print("saved", a.save, flush=True)
