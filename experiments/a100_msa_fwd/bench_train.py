"""Training (forward + backward) of the A100 MSA kernels against fp32 autograd of the same module on the same weights.

  python bench_train.py --op opm --length 384 768 [--out records/x.json] [-D NAME=VAL ...] [--no-time]

Gradients: every input (msa, residual) and every parameter, each as rel-RMS vs fp32; the bf16 PyTorch module's own error for scale.
Timing: forward + backward (with a fixed random dz) captured in one CUDA graph, median of 7 x 50 replays.
"""
import argparse
import collections
import copy
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import msa_a100 as MA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.msa_pair_weighted_averaging import MSAPairWeightedAveraging  # noqa: E402
from miniworld_engine.modules.outer_product import OuterProductMean  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--op", nargs="+", default=["opm"])
p.add_argument("--length", nargs="+", type=int, default=[384, 768])
p.add_argument("--msa-depth", type=int, default=1024)
p.add_argument("--out", default=None)
p.add_argument("--define", "-D", action="append", default=[])
p.add_argument("--no-time", action="store_true")
p.add_argument("--repeat", type=int, default=1, help="extra steps before the check (for ncu -s)")
p.add_argument("--mask-frac", type=float, default=.1, help="PWA: fraction of masked keys")
p.add_argument("--mask-mode", default="rand", choices=["rand", "none", "all"])
a = p.parse_args()
extra = [f"-D{d}" for d in a.define]
ext = MA.build(extra=extra)
S = a.msa_depth
record = dict(gpu=torch.cuda.get_device_name(), host=os.uname().nodename, extra=extra, rows=[])


def init_(m):
    torch.manual_seed(1234)
    with torch.no_grad():
        for n, t in m.named_parameters():
            if t.ndim >= 2:
                t.normal_(std=t.shape[-1] ** -.5)
            elif "weight" in n:
                t.copy_(1 + .1 * torch.randn_like(t))
            else:
                t.normal_(std=.05)
    return m


def rel(x, ref):
    """rel-RMS vs the fp32 reference; for a reference that is analytically 0 (e.g. LN_z beta under a softmax: its row sums of dlogit
    vanish) the absolute RMS is returned as a negative number, printed as 'abs'."""
    d = x.float() - ref.float()
    rr = float(ref.float().square().mean().sqrt())
    dd = float(d.square().mean().sqrt())
    return -dd if rr < 1e-5 else dd / rr


def graph_ms(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(5):
        g.replay()
    en.record()
    en.synchronize()
    warm = min(2000, max(10, int(300 / max(st.elapsed_time(en) / 5, 1e-3))))
    for _ in range(warm):
        g.replay()
    rounds = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(20):
            g.replay()
        en.record()
        en.synchronize()
        rounds.append(st.elapsed_time(en) / 20)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
    kt = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kt[e.name[:60]] += e.device_time_total / 5
    return statistics.median(rounds), rounds, dict(sorted(kt.items(), key=lambda kv: -kv[1]))


def case_opm(L):
    mod = init_(OuterProductMean(64, 128, 32, implementation=I.PYTORCH).cuda().bfloat16())
    torch.manual_seed(90323)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    msa, mask, pair = torch.randn(1, S, L, 64, **kw), torch.rand(1, S, L, device="cuda") > .1, torch.randn(1, L, L, 128, **kw)
    dz = torch.randn(1, L, L, 128, **kw)
    names = dict(ln_w="ln_msa.weight", ln_b="ln_msa.bias", w_left="to_left.weight", w_right="to_right.weight", w_out="to_out.weight",
                 b_out="to_out.bias")

    def autograd_grads(m, dtype):
        x = msa.to(dtype).requires_grad_(True)
        r = pair.to(dtype).requires_grad_(True)
        y = m(x, mask, residual=r)
        prm = dict(m.named_parameters())
        gs = torch.autograd.grad(y, [x, r] + [prm[v] for v in names.values()], dz.to(dtype))
        return dict(zip(["msa", "residual", *names], gs))
    ref = autograd_grads(copy.deepcopy(mod).float(), torch.float32)
    bf = autograd_grads(mod, torch.bfloat16)
    pk = MA.pack_opm_train(mod)
    bufs = {}

    def step():
        MA.opm_train_forward(ext, msa, mask, pk, pair, bufs)
        return MA.opm_backward(ext, dz, msa, mask, pk, bufs)
    return step, ref, bf


def case_pwa(L, bwd="ref"):
    mod = init_(MSAPairWeightedAveraging(64, 128, 8, 32, implementation=I.PYTORCH, p_drop=.15).cuda().bfloat16())
    torch.manual_seed(90323)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    msa, pair = torch.randn(1, S, L, 64, **kw), torch.randn(1, L, L, 128, **kw)
    mask = {"rand": torch.rand(1, L, device="cuda") > a.mask_frac, "none": None,
            "all": torch.zeros(1, L, dtype=torch.bool, device="cuda")}[a.mask_mode]
    dres = torch.randn(1, S, L, 64, **kw)
    keep = (torch.rand(L, 64, device="cuda") > mod.drop_msa.p_drop).to(torch.bfloat16)
    scale = 1.0 / (1.0 - mod.drop_msa.p_drop)
    names = dict(ln_msa_w="ln_msa.weight", ln_msa_b="ln_msa.bias", w_value="to_value.weight", w_gate="to_gate.weight",
                 ln_pair_w="ln_pair.weight", ln_pair_b="ln_pair.bias", w_bias="to_bias.weight", w_out="to_out.weight")

    def autograd_grads(m, dtype):
        m = copy.deepcopy(m).to(dtype).eval()          # dropout off in the module; the same keep-mask applied by hand
        x = msa.to(dtype).requires_grad_(True)
        z = pair.to(dtype).requires_grad_(True)
        upd = m(x, z, mask) - x
        yv = x + upd * (keep.to(dtype) * scale)[None, None]
        prm = dict(m.named_parameters())
        gs = torch.autograd.grad(yv, [x, z] + [prm[v] for v in names.values()], dres.to(dtype))
        return dict(zip(["msa", "pair", *names], gs))
    ref = autograd_grads(mod, torch.float32)
    bf = autograd_grads(mod, torch.bfloat16)
    pk = MA.pack_pwa_train(mod)
    bufs = {}
    bw = {"ref": MA.pwa_backward_ref, "k": lambda *z: MA.pwa_backward(ext, *z)}[bwd]

    def step():
        MA.pwa_train_forward(ext, msa, pair, mask, pk, bufs, keep)
        return bw(dres, msa, pair, mask, pk, bufs)
    return step, ref, bf


for op in a.op:
    for L in a.length:
        step, ref, bfm = {"opm": case_opm, "pwa_ref": lambda L: case_pwa(L, "ref"), "pwa": lambda L: case_pwa(L, "k")}[op](L)
        with torch.no_grad():
            for _ in range(a.repeat - 1):
                step()
            g = step()
            torch.cuda.synchronize()
            errs = {k: rel(g[k], ref[k]) for k in ref}
            errs_bf = {k: rel(bfm[k], ref[k]) for k in ref}
        row = dict(op=op, L=L, S=S, rel=errs, bf16_module_rel=errs_bf, finite=all(bool(torch.isfinite(v).all()) for v in g.values()))
        fm = lambda e: f"abs {-e:.1e}" if e < 0 else f"{e:.2e}"  # noqa: E731
        msg = f"{op} L{L} grads rel (bf16 module): " + "  ".join(f"{k} {fm(errs[k])} ({fm(errs_bf[k])})" for k in ref)
        if not a.no_time:
            with torch.no_grad():
                ms, rounds, kt = graph_ms(step)
            row.update(us=ms * 1000, rounds_ms=rounds, kernels_us=kt)
            msg = f"{op} L{L}: fwd+bwd {ms*1000:9.1f} us\n  " + msg + "\n    " + "  ".join(f"{k[:38]} {v:.0f}" for k, v in list(kt.items())[:9])
        print(msg, flush=True)
        record["rows"].append(row)
        del step, ref, bfm, g
        torch.cuda.empty_cache()
if a.out:
    Path(a.out).write_text(json.dumps(record, indent=1))
