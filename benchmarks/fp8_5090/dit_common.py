"""Shared harness of the RTX 5090 token DiT / atom DiT FP8 benchmarks (``token_dit_ab.py``, ``atom_dit_ab.py``, ``dit_profile.py``).

Blocks come from the measurers' registry rows (``miniworld_engine.measure.config``): eval, ``master`` (fp32 parameters, bf16 activations), the
module's first eval / master row (Token DiT: ``d_pair = 128`` where the class has it), with the length / samples replaced and the conditioning
shared by the samples (the ``*_shared_cond`` stream: FoldForge's samplers call the blocks that way)."""

from __future__ import annotations

import copy
import dataclasses
import statistics
import tempfile

import torch

YAML = (
    "modules: [{mods}]\nbackends: [pytorch, cuda]\nexternal: []\nrows: []\ndtypes: []\nmodes: []\nlengths: min\ndepths: min\nsamples: all\n"
    "device: cuda\nseed: 0\nfixed: {{bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}}\n"
)


def case(module: str, length: int, samples: int = 5, shared: bool = True, pick: str | None = None):
    """The registry case of ``module`` (eval, master), ``length`` / ``samples`` replaced. ``pick``: a substring the case label must hold (e.g.
    ``key_window=slide``); without it the case with no ``key_window`` / ``mask_form`` option is preferred, then ``d_pair=128``."""
    from miniworld_engine.measure import config as C

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(YAML.format(mods=module))
    cfg = C.load(f.name)

    def rank(lab):
        return (("key_window" in lab or "mask_form" in lab) and pick is None, "d_pair=128" not in lab)

    best = None
    for c in C.cases(cfg):
        if c.module != module or c.mode != "eval" or c.dtype != "master" or c.stream.endswith("_fp32") or c.stream.endswith("_fp32_cond"):
            continue
        lab = C.describe(c, cfg)["label"]
        if pick is not None and pick not in lab:
            continue
        if best is None or rank(lab) < rank(best[1]):
            best = (c, lab)
    if best is None:
        raise LookupError(module)
    c = best[0]
    stream = c.stream
    if shared and not stream.endswith("_shared_cond") and stream in ("token_single", "atom_single"):
        stream = stream + "_shared_cond"
    return dataclasses.replace(c, length=length, samples=samples, stream=stream)


class Rig:
    """One block + its inputs + a callable step on the ``cuda`` backend (eval, no grad); ``gold()`` = the module in float64 on PyTorch."""

    def __init__(self, c):
        from miniworld_engine.measure import config as C
        from miniworld_engine.measure import speed as S

        dev = torch.device("cuda")
        self.m = copy.deepcopy(C.build(c, device=dev))
        with S.unpadded_masks():
            self.inp = C.inputs(c, self.m, device=dev)
        self.case, self.C, self.S = c, C, S
        self.step = S.make_step(self.m, self.inp, backend="cuda", train=False, autocast=C.autocast(c), seed=0, compile_mode=None)
        self.xk = next(k for k, v in self.inp.items() if torch.is_tensor(v) and v.is_floating_point() and v.ndim == 4)

    def gold(self, scale: float = 1.0):
        m64 = copy.deepcopy(self.m).double()
        inp = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in self.inp.items()}
        inp[self.xk] = inp[self.xk] * scale
        st = self.S.make_step(m64, inp, backend="pytorch", train=False, autocast=None, seed=0, compile_mode=None)
        with torch.no_grad():
            y = st()
        y = (y if torch.is_tensor(y) else y[0]).double()
        return y, inp[self.xk].double()

    def run(self, scale: float = 1.0):
        x = self.inp[self.xk]
        keep = x.clone()
        x.mul_(scale)
        try:
            with torch.no_grad():
                o = self.step()
            torch.cuda.synchronize()
            return (o if torch.is_tensor(o) else o[0]).float().clone()
        finally:
            x.copy_(keep)


def err(y, gold, x):
    """(update rms, update max, judge, rms_rel): rms(y - y64) / rms(y64 - x), max|y - y64| / max|y64 - x|, max|y - y64| / max|y64|, ||y - y64|| / ||y64||."""
    d = y.double() - gold
    u = gold - x
    return (
        float(d.pow(2).mean().sqrt() / u.pow(2).mean().sqrt()),
        float(d.abs().max() / u.abs().max()),
        float(d.abs().max() / gold.abs().max()),
        float(d.pow(2).sum().sqrt() / gold.pow(2).sum().sqrt()),
    )


_FLUSH: list = []


def graph(f, n: int = 3, flush: bool = True):
    """A CUDA graph of ``n`` calls of ``f`` (each after a 128 MB L2 flush when ``flush``) and one of the flushes alone, captured under
    ``static_weights()`` + ``static_inputs()`` (as FoldForge's sampler graphs: weight forms and hoisted tables are made by the eager warm-up,
    a replay reads them). Returns (graph, flush graph, the static-tables handle: keep it while the graph is replayed)."""
    from miniworld_engine.kernels import _capture

    if not _FLUSH:
        _FLUSH.append(torch.empty(128 << 20, dtype=torch.uint8, device="cuda"))
    fl = _FLUSH[0]
    with torch.no_grad(), _capture.static_weights(), _capture.static_inputs() as handle:
        f()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                if flush:
                    fl.zero_()
                f()
        gf = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gf):
            for _ in range(n):
                if flush:
                    fl.zero_()
    return g, gf, handle


def time_graph(g, gf, n: int = 3, reps: int = 7, *_keep) -> float:
    """us per call: median of ``reps`` replays of ``g`` minus that of the flush-only graph, over ``n``."""

    def t(gr):
        gr.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    return (t(g) - t(gf)) / n * 1e3


def ab(graphs: dict, rounds: int = 6, n: int = 3) -> dict:
    """Alternating A-B over ``rounds`` (order reversed every round); per label the median of its per-round times (us)."""
    out: dict = {k: [] for k in graphs}
    keys = list(graphs)
    for r in range(rounds):
        for k in keys if r % 2 == 0 else keys[::-1]:
            out[k].append(time_graph(*graphs[k][:2], n=n))
    return {k: statistics.median(v) for k, v in out.items()}
