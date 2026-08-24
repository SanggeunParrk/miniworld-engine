"""Is layernorm_fwd_cuda's answer at the driver's ragged shape a function of its inputs only?

Two runs under the same env disagreed: the warm run passed, the run under compute-sanitizer gave
rel rstd=7.05e-01 y=4.46e-01, while memcheck itself reported 0 errors -- consistent with an
overrun small enough to stay inside cudaMalloc's alignment padding (a 125-element bf16 row is 250
bytes, a 3-element overrun 6 bytes).

An earlier version of this probe hardcoded rows2d(4096, d) and found the kernel deterministic --
but the driver's ragged row count is _M = 16381, not 4096, so it was testing a shape the failure
never came from. This calls the REGISTERED CHECKER, so the shapes are exactly the ones that failed.

A kernel reading only its operands must return bit-identical output on every repeat, however the
allocator is dirtied in between.
"""
from __future__ import annotations

import hashlib
import sys

sys.path.insert(0, "src")

import torch


def run_once(dirty: bool, seed: int = 1234):
    if dirty:
        scratch = torch.empty(1 << 24, device="cuda", dtype=torch.float32).uniform_(-3.0, 3.0)
        del scratch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from miniworld_engine.kernels.checks.layernorm import layernorm_fwd_cuda as chk
    got = chk()
    pairs = got if isinstance(got, dict) else {"out": got}
    rels, h = {}, hashlib.sha256()
    for name in sorted(pairs):
        a, e = (t.float() for t in pairs[name])
        rels[name] = ((a - e).abs().max() / (e.abs().max() or 1.0)).item()
        h.update(name.encode())
        h.update(a.contiguous().cpu().numpy().tobytes())
    return rels, h.hexdigest()[:16]


def main() -> int:
    from miniworld_engine.kernels.drivers import SHAPE_MODE
    from miniworld_engine.kernels.drivers.layernorm_linear import _D, _M
    print(f"device={torch.cuda.get_device_name()}  mode={SHAPE_MODE}  _M={_M} _D={_D}")
    shas, allrels = [], []
    for i in range(5):
        rels, sha = run_once(dirty=bool(i % 2))
        shas.append(sha); allrels.append(rels)
        print("  run {} dirty={}  sha={}  {}".format(
            i, i % 2, sha, "  ".join(f"{k}={v:.3e}" for k, v in sorted(rels.items()))))
    print(f"\n  distinct output hashes over 5 runs: {len(set(shas))}")
    print(f"  bit-identical every run: {len(set(shas)) == 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
