"""Triton OPM / PWA vs the PyTorch module (static torch.compile, same run) and fp32 autograd (accuracy).

  python bench.py --op opm pwa --length 384 768 [--out records/x.json] [--no-time]

Inference: output rel-RMS vs the fp32 module. Training: every input and parameter gradient vs fp32 autograd (bf16 module's error for
scale). Timing: each arm captured in one CUDA graph, median of 7 x 20 replays; PyTorch = torch.compile(fullgraph, dynamic=False) of the
module (the a100_anthropic_baseline protocol), measured in the same process.
"""
import argparse
import copy
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
if os.environ.get("MSA_TRITON_PRECOMPILE") == "1":
    # the engine's build-time hook: every autotune round is compiled in parallel (spawned workers re-import the kernel module, so
    # this directory must be on PYTHONPATH -- env.sh adds it), then timed serially as usual
    from miniworld_engine.autotune import capture
    capture.install()
import opm_triton as OT  # noqa: E402
import pwa_triton as PT  # noqa: E402
from miniworld_engine.modules.msa_pair_weighted_averaging import MSAPairWeightedAveraging  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.outer_product import OuterProductMean  # noqa: E402

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
    d = x.float() - ref.float()
    rr = float(ref.float().square().mean().sqrt())
    dd = float(d.square().mean().sqrt())
    return -dd if rr < 1e-5 else dd / rr


def fm(e):
    return f"abs {-e:.1e}" if e < 0 else f"{e:.2e}"


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
    for _ in range(min(2000, max(10, int(300 / max(st.elapsed_time(en) / 5, 1e-3))))):
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
    del g
    return statistics.median(rounds)


def run(op, L):
    kw = dict(device="cuda", dtype=torch.bfloat16)
    if op == "opm":
        mod = init_(OuterProductMean(64, 128, 32, implementation=I.PYTORCH).cuda().bfloat16())
        torch.manual_seed(90323)
        x, mask, r = torch.randn(1, S, L, 64, **kw), torch.rand(1, S, L, device="cuda") > .1, torch.randn(1, L, L, 128, **kw)
        dz = torch.randn(1, L, L, 128, **kw)
        inputs = [x, r]
        fwd_ref = lambda m, xx, rr: m(xx, mask, residual=rr)  # noqa: E731
        fwd_tri = lambda xx, rr: OT.outer_product_mean(mod, xx, mask, rr)  # noqa: E731
    elif op == "pwa":
        # accuracy in eval mode (no dropout: the module's RNG stream cannot be matched); timing in train mode (p = 0.15 on both sides)
        mod = init_(MSAPairWeightedAveraging(64, 128, 8, 32, implementation=I.PYTORCH, p_drop=.15).cuda().bfloat16()).eval()
        torch.manual_seed(90323)
        x, zz, mask = torch.randn(1, S, L, 64, **kw), torch.randn(1, L, L, 128, **kw), torch.rand(1, L, device="cuda") > .1
        dz = torch.randn(1, S, L, 64, **kw)
        inputs = [x, zz]
        fwd_ref = lambda m, xx, pp: m(xx, pp, mask)  # noqa: E731
        fwd_tri = lambda xx, pp: PT.pair_weighted_averaging(mod, xx, pp, mask)  # noqa: E731
    else:
        raise ValueError(op)
    if a.engine:                      # the engine's own module dispatch (implementation="miniworld") in place of this directory's copy
        def fwd_tri(*xs, _ref=fwd_ref):
            mod.implementation = I.MINIWORLD
            try:
                return _ref(mod, *xs)
            finally:
                mod.implementation = I.PYTORCH
    prm = [p_ for p_ in mod.parameters()]
    names = [n for n, _ in mod.named_parameters()]

    def grads(fn, dtype_inputs, params):
        xs = [t.detach().to(dtype_inputs).requires_grad_(True) for t in inputs]
        y = fn(*xs)
        gs = torch.autograd.grad(y, xs + params, dz.to(y.dtype))
        return y, gs
    ref32 = copy.deepcopy(mod).float().eval()
    y32, g32 = grads(lambda *xs: fwd_ref(ref32, *xs), torch.float32, list(ref32.parameters()))
    ybf, gbf = grads(lambda *xs: fwd_ref(mod, *xs), torch.bfloat16, prm)
    ytr, gtr = grads(fwd_tri, torch.bfloat16, prm)
    labels = ["in%d" % k for k in range(len(inputs))] + names
    row = dict(op=op, L=L, fwd_rel=rel(ytr, y32), fwd_rel_bf16_module=rel(ybf, y32),
               grad_rel={k: rel(t, r_) for k, t, r_ in zip(labels, gtr, g32)},
               grad_rel_bf16_module={k: rel(t, r_) for k, t, r_ in zip(labels, gbf, g32)})
    print(f"{op} L{L} fwd rel {row['fwd_rel']:.2e} (bf16 module {row['fwd_rel_bf16_module']:.2e})\n  grads: "
          + "  ".join(f"{k} {fm(row['grad_rel'][k])} ({fm(row['grad_rel_bf16_module'][k])})" for k in labels), flush=True)
    del y32, g32, ref32, ybf, gbf, ytr, gtr
    if not a.no_time:
        comp = torch.compile(mod, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        with torch.no_grad():
            row["infer_triton_ms"] = graph_ms(lambda: fwd_tri(*inputs))
            row["infer_pytorch_ms"] = graph_ms(lambda: fwd_ref(comp, *inputs))
        xs = [t.detach().requires_grad_(True) for t in inputs]
        mod.train()

        def tr_step(fn):
            def step():
                y = fn(*xs)
                return torch.autograd.grad(y, xs + prm, dz)
            return step
        row["train_triton_ms"] = graph_ms(tr_step(fwd_tri))
        torch.compiler.reset()
        comp = torch.compile(mod, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        row["train_pytorch_ms"] = graph_ms(tr_step(lambda *z: fwd_ref(comp, *z)))
        print(f"  inference: triton {row['infer_triton_ms']:.3f} ms  pytorch-compile {row['infer_pytorch_ms']:.3f} ms  "
              f"-> {row['infer_pytorch_ms'] / row['infer_triton_ms']:.2f}x\n"
              f"  training : triton {row['train_triton_ms']:.3f} ms  pytorch-compile {row['train_pytorch_ms']:.3f} ms  "
              f"-> {row['train_pytorch_ms'] / row['train_triton_ms']:.2f}x", flush=True)
        mod.eval()
    record["rows"].append(row)
    torch.cuda.empty_cache()


# guarded: the precompile pool SPAWNS workers, which re-import this file as __mp_main__ and must not run the benchmark
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", nargs="+", default=["opm"])
    p.add_argument("--length", nargs="+", type=int, default=[384, 768])
    p.add_argument("--msa-depth", type=int, default=1024)
    p.add_argument("--out", default=None)
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--engine", action="store_true", help="time the engine module (implementation=miniworld) as the Triton arm")
    a = p.parse_args()
    S = a.msa_depth
    record = dict(gpu=torch.cuda.get_device_name(), host=os.uname().nodename, rows=[])
    for op in a.op:
        for L in a.length:
            run(op, L)
    if a.out:
        Path(a.out).write_text(json.dumps(record, indent=1))
    if os.environ.get("MSA_TRITON_PRECOMPILE") == "1":
        capture.shutdown_precompile()           # close the pool before interpreter teardown (else Pool.__del__ warns)
