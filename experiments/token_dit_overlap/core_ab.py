"""Kernel-level A/B of CUDA-core build variants in one process, interleaved, with bench.us (do_bench, L2 evicted).

  python core_ab.py --variants "QDEP=1 NODANGLE=0" "" "REGFIX=1"
"""
import argparse, os, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import us                                                  # noqa: E402
from tdit import cuda_core as CC                                      # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--variants", nargs="+", default=["QDEP=1 NODANGLE=0", ""])
p.add_argument("--lengths", type=int, nargs="+", default=[384, 768])
p.add_argument("--rounds", type=int, default=5)
a = p.parse_args()
S, H, DS, NB, dev, bf = 5, 16, 768, 24, "cuda", torch.bfloat16


def build(defs):
    os.environ["ATTN_DEFS"] = defs
    CC._ext.cache_clear()
    ext = CC._ext()
    CC._ext.cache_clear()
    return ext


EXT = {v: build(v) for v in a.variants}
os.environ["ATTN_DEFS"] = ""
for L in a.lengths:
    torch.manual_seed(0)
    src = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
    bias = (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
    qkvg = src.clone()
    t = {v: [] for v in a.variants}
    for _ in range(a.rounds):
        for v in a.variants:
            t[v].append(us(lambda e=EXT[v]: e.attn_core(qkvg, bias, 3, S, H, None)))
    base = statistics.median(t[a.variants[0]])
    print(f"L{L}", flush=True)
    for v in a.variants:
        m = statistics.median(t[v])
        print(f"  [{v or 'default'}]{'':<{max(0, 26 - len(v or 'default'))}} {m:7.2f} us  ({m - base:+.2f})   "
              f"runs {' '.join(f'{x:.1f}' for x in t[v])}", flush=True)
