"""Bias-only token DiT pair-bias logit modes (``miniworld_engine.bias_logits``): the speed / memory comparison and the cross-version bit-identity dump.

    python benchmarks/bias_logits_modes.py bench [--lengths 384 768 2048 3072] [--configs 384:128 768:256] [--out FILE.jsonl]
    python benchmarks/bias_logits_modes.py dump  FILE.pt      # fp32-mode outputs of the judge's eval cells (runs on 3.0.6 too: no switch is touched)
    python benchmarks/bias_logits_modes.py compare A.pt B.pt  # max abs difference per cell (the fp32 mode must be 0 against 3.0.6)

``bench``: fp32 against bf16 mode on the same process, interleaved A-B-A-B, each timing a CUDA-graph replay with a 256 MB L2 flush before it
(the engine's speed measurer's rule), median of the replays. Two things are timed per (cond:pair config, L):

* ``pbuild`` -- the logits + softmax that make P of one block, the part the mode changes. H100: ``FusedBiasOnlyDiT.hoist`` (a sampler makes P once per
  pair, the block's step reads it); A100 / RTX 5090: ``pair_bias_generic`` + the row softmax (the sm_80 block makes them at every call).
* ``block`` -- the whole block forward with one sample (A = 1; the logits are a fixed cost, independent of A). Every replay rebuilds P (the P cache is
  keyed per call, and a capture makes its P inside the graph), so it contains ``pbuild``; its eager peak only shows the activations (the cached P is held
  across the two eager calls), the memory column of ``pbuild`` is the one to read.
Peak memory: ``torch.cuda.max_memory_allocated`` above the memory held before the call (pair, weights, activations), eager, of the same call.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections.abc import Callable, Mapping
from typing import Any

import torch

DEV = "cuda"
EPS = 1e-5


def _card() -> str:
    cc = torch.cuda.get_device_capability()
    return {(9, 0): "h100_sm90", (8, 0): "a100_sm80", (12, 0): "5090_sm120", (10, 0): "b200_sm100"}.get(cc, f"sm{cc[0]}{cc[1]}")


def _block(dc: int, dp: int, dtype: str = "bfloat16", backend: str = "cuda", wide: bool = False):
    from miniworld_engine.measure import config as C
    from miniworld_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly

    m = TokenDiTMiniWorldBiasOnly(d_single=768, d_cond=dc, d_pair=dp, n_head=16, n=2).to(DEV).float()
    g = torch.Generator(device=DEV).manual_seed(0)
    C.randomize(m, g)
    if wide:  # the huge-logit regime (|logit| ~ 1e3)
        with torch.no_grad():
            m.attention.to_bias.weight.mul_(45.0)
    out: Any = C.cast(m, dtype).eval().requires_grad_(False)
    return out.set_backend(backend)


def _inputs(m: Any, length: int, samples: int, dtype: str, seed: int = 1):
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    inp = m.make_inputs(
        length=length, samples=samples, depth=None, dtype=act, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed)
    )
    mask = torch.ones(1, length, dtype=torch.bool, device=DEV)
    mask[:, length - 24 :] = False
    inp["mask"] = mask
    return inp


# ------------------------------------------------------------------------------------------------------------------------------- timing
def _graph(fn: Callable[[], object]):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _ab(fns: Mapping[str, Callable[[], object]], rounds: int = 5, inner: int = 8) -> dict[str, float]:
    """Interleaved A-B-A-B of the graphs of ``fns`` (name -> callable): median microseconds of one replay after an L2 flush."""
    graphs = {k: _graph(f) for k, f in fns.items()}
    flush = torch.empty(256 << 20, dtype=torch.int8, device=DEV)
    samples: dict[str, list[float]] = {k: [] for k in fns}
    order = list(fns)
    for r in range(rounds):
        for k in order if r % 2 == 0 else order[::-1]:
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(inner)]
            for s, e in ev:
                flush.zero_()
                s.record()
                graphs[k].replay()
                e.record()
            torch.cuda.synchronize()
            samples[k] += [s.elapsed_time(e) * 1e3 for s, e in ev]
    del graphs
    return {k: statistics.median(v) for k, v in samples.items()}


def _peak(fn) -> float:
    """MiB above the memory held before the call, eager."""
    fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    del out
    return (torch.cuda.max_memory_allocated() - base) / 2**20


# ------------------------------------------------------------------------------------------------------------------------------- bench
def _pbuild_fns(m, inp, length: int):
    """name -> callable making P of one block in the mode (the part the mode changes)."""
    card = _card()
    pair, mask = inp["pair"], inp["mask"]
    if card in ("h100_sm90", "b200_sm100"):
        from miniworld_engine.kernels.integrations import bias_only_dit as bo
        from miniworld_engine.modules.token_dit import _kernel_view

        view = _kernel_view(m)
        weights = [view.get_parameter(n) for n in bo.WEIGHTS]
        runner = bo._runner(inp["single"], weights)
        pc = pair.contiguous()
        return {mode: (lambda mode=mode: runner.hoist(pc, mask, mode == "bf16")) for mode in ("fp32", "bf16")}
    from miniworld_engine.kernels.integrations import bias_only_dit_sm80 as sm

    at = m.attention
    wcm = (at.to_bias.weight.float() * at.ln_pair.weight.float()[None, :]).t().contiguous()
    pr = pair.contiguous()
    f32 = torch.float32

    def make(mode: str):
        def run():
            bias = sm._sm80().pair_bias_generic(pr[0], wcm, None, length, EPS, oscale=1.0, out_dtype=None if mode == "bf16" else f32)
            P, _ = sm._softmax(bias, mask, False, torch.bfloat16)
            return P

        return run

    return {mode: make(mode) for mode in ("fp32", "bf16")}


def _block_fns(m, inp):
    def make(mode: str):
        def run():
            m.set_pair_bias_logits(mode)
            return m(**inp)

        return run

    return {mode: make(mode) for mode in ("fp32", "bf16")}


def bench(args) -> None:
    card = _card()
    rows = []
    for cfg in args.configs:
        dc, dp = (int(x) for x in cfg.split(":"))
        for length in args.lengths:
            if length > 3072 and card != "h100_sm90":
                continue
            gc.collect()
            torch.cuda.empty_cache()
            try:
                m = _block(dc, dp)
                inp = _inputs(m, length, 1, "bfloat16")
                with torch.no_grad():
                    why = m.unsupported(**inp)
                    if why is not None:
                        print(f"{card} dc{dc} dp{dp} L{length}: skipped ({why})", flush=True)
                        continue
                    for what, fns in (("pbuild", _pbuild_fns(m, inp, length)), ("block", _block_fns(m, inp))):
                        t = _ab(fns)
                        mem = {k: _peak(f) for k, f in fns.items()}
                        rows.extend(
                            {
                                "card": card,
                                "cond": dc,
                                "pair": dp,
                                "L": length,
                                "what": what,
                                "mode": mode,
                                "us": round(t[mode], 1),
                                "peak_mib": round(mem[mode], 1),
                            }
                            for mode in ("fp32", "bf16")
                        )
                        print(
                            f"{card} cond{dc}/pair{dp} L{length:5d} {what:6s}: fp32 {t['fp32']:9.1f} us {mem['fp32']:8.0f} MiB | bf16 {t['bf16']:9.1f} us "
                            f"{mem['bf16']:8.0f} MiB | x{t['fp32'] / t['bf16']:.2f} time, x{mem['fp32'] / max(mem['bf16'], 1e-9):.2f} mem",
                            flush=True,
                        )
                    m.set_pair_bias_logits(None)
            except torch.OutOfMemoryError as e:
                print(f"{card} cond{dc}/pair{dp} L{length}: out of memory ({str(e)[:80]})", flush=True)
            finally:
                from miniworld_engine.kernels.integrations import bias_only_dit

                bias_only_dit._RUNNERS.clear()
                gc.collect()
                torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


# ------------------------------------------------------------------------------------------------------------------------------- dump / compare
#: the judge's eval cells of TokenDiTMiniWorldBiasOnly (requirements_module.csv): (cond, pair) x dtype float32 | master, L 128 .. 768, 5 samples
def dump(path: str) -> None:
    out: dict[str, torch.Tensor] = {}
    backends = ["cuda"] + (["pytorch", "triton"] if torch.cuda.is_available() else [])
    for dc, dp in ((384, 128), (768, 128), (384, 256), (768, 256)):
        for dtype in ("float32", "master", "bfloat16"):
            for length in (128, 256, 384, 768):
                for backend in backends:
                    if backend != "cuda" and (length != 256 or dc != 384):
                        continue
                    m = _block(dc, dp, "float32" if dtype == "master" else dtype, "pytorch")
                    if dtype == "master":
                        from miniworld_engine.measure import config as C

                        C.cast(m, "master")
                    m.set_backend(backend)
                    inp = _inputs(m, length, 5, "float32" if dtype == "float32" else "bfloat16")
                    key = f"{backend}/{dtype}/c{dc}p{dp}/L{length}"
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master"):
                        why = m.unsupported(**inp) if backend != "pytorch" else None
                        if why is not None:
                            out[key] = torch.tensor([float("nan")])
                            print(key, "unsupported:", why[:80], flush=True)
                            continue
                        for rep in (0, 1):  # a second call: the P cache serves it (must equal the first)
                            y = m(**inp).float().cpu()
                            if rep == 0:
                                out[key] = y
                            else:
                                assert torch.equal(out[key], y), f"{key}: second call differs"
                    print(key, "ok", flush=True)
                    del m
                gc.collect()
    # the hoisted bias path and one training cell
    from miniworld_engine.modules.token_dit import pair_bias_all

    m = _block(384, 128, "bfloat16", "cuda")
    inp = _inputs(m, 256, 5, "bfloat16")
    with torch.no_grad():
        try:
            b = pair_bias_all([m], inp["pair"])[0]
            out["hoist/bias"] = b.float().cpu()
            out["hoist/y"] = m(inp["single"], inp["cond"], None, inp["mask"], bias=b).float().cpu()
        except Exception as e:
            print("hoist cell:", type(e).__name__, str(e)[:80])
    m = _block(384, 128, "bfloat16", "cuda")
    single, cond, pair = (
        t.detach().clone().requires_grad_(i != 1) for i, t in enumerate((inp["single"][:2], inp["cond"][:2], inp["pair"]))
    )
    try:
        with torch.enable_grad():
            y = m.requires_grad_(True)(single, cond, pair, inp["mask"])
            (y.float() ** 2).sum().backward()
        out["train/y"], out["train/dpair"] = y.detach().float().cpu(), pair.grad.float().cpu()
    except Exception as e:
        print("train cell:", type(e).__name__, str(e)[:80])
    torch.save(out, path)
    print("dumped", len(out), "cells to", path)


def compare(a: str, b: str) -> int:
    x, y = torch.load(a), torch.load(b)
    bad = 0
    for k in sorted(set(x) | set(y)):
        if k not in x or k not in y:
            print(f"{k}: only in {'A' if k in x else 'B'}")
            bad += 1
            continue
        both_nan = bool(torch.isnan(x[k]).all()) and bool(torch.isnan(y[k]).all())
        d = 0.0 if both_nan else (x[k] - y[k]).abs().max().item()
        bad += d != 0.0
        if d != 0.0:
            print(f"{k}: max abs diff {d:.3e}")
    print(f"{len(x)} cells, {bad} differ")
    return int(bad > 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--lengths", type=int, nargs="+", default=[384, 768, 2048, 3072])
    b.add_argument("--configs", nargs="+", default=["384:128", "768:256"])
    b.add_argument("--out")
    d = sub.add_parser("dump")
    d.add_argument("path")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "bench":
        bench(args)
        return 0
    if args.cmd == "dump":
        dump(args.path)
        return 0
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
