"""Per-kernel profile of one token DiT case: CUDA-graph replay of the default route and of engine-fp8 (torch profiler, CUDA activities).

    python td_profile.py --module TokenDiTProtenix --length 256 --samples 5 [--static-weights] [--dpair 128]
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--module", default="TokenDiTProtenix")
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--dpair", type=int, default=128)
    p.add_argument("--static-weights", action="store_true")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    import torch
    from paper_common import cfg as cfgmod
    from miniworld_engine import fp8
    from miniworld_engine.kernels import _capture
    from miniworld_engine.measure import config
    from miniworld_engine.modules.primitives import unpadded_masks
    from torch.profiler import ProfilerActivity, profile

    if a.static_weights:
        _capture.set_static_weights(True)
    cfg = cfgmod.load(str(Path(__file__).resolve().parent / "cfg_tokendit.yaml"), argparse.Namespace(module=[a.module], lengths=str(a.length), dtypes=["master"], modes=["eval"]))
    case = next(c for c in cfgmod.limit(cfg, 0) if c.samples == a.samples and f"d_pair={a.dpair};" in config.describe(c, cfg)["label"] + ";" and c.dtype == "master")
    dev = torch.device("cuda")
    base = config.build(case, device=dev)
    with unpadded_masks():
        inputs = config.inputs(case, base, device=dev)
    ac = config.autocast(case)

    def mk(module):
        fwd = module._forward_cuda

        def step():
            with torch.no_grad(), torch.autocast("cuda", dtype=ac) if ac is not None else contextlib.nullcontext():
                return fwd(**inputs)

        return step

    torch.backends.cuda.matmul.allow_tf32 = True
    for name, backend in (("default", "cuda"), ("fp8", "engine-fp8")):
        m = copy.deepcopy(base).set_backend(backend).eval()
        step = mk(m)
        fp8.reset_route_stats()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            step()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
        agg = collections.defaultdict(lambda: [0, 0.0])
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                agg[e.name][0] += 1
                agg[e.name][1] += e.device_time
        tot = sum(v[1] for v in agg.values()) / 10
        n = sum(v[0] for v in agg.values()) / 10
        print(f"\n=== {name} {a.module} L{a.length} S{a.samples} dpair{a.dpair} static={a.static_weights}: {n:.0f} kernels/replay, kernel sum {tot:.1f} us; stats {fp8.route_stats()['token_dit']}")
        for k, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
            print(f"{t / 10:9.2f} us  x{c / 10:5.1f}  {k[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
