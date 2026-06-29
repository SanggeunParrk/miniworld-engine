"""Benchmarks for the bias-only triangle attention modules (single-dir + bidir).

Methodology (matches the trimul module benchmarks):
  inference : pytorch = torch.compile ; ours = manual CUDA-graph (deployment regime)
  training  : both torch.compile (params require grad, exact fwd+bwd)
ms / layer, B=1, bf16, H100. d_pair sweep {128, 256, 512}; bias-only
(use_self_attention=False). "ours" = the TRITON implementation (repo layernorm_kernel
+ fused_gate_out/split + inference LN+proj concat + per-GPU dispatch).

Emits parseable sections; capture under benchmarks/artifacts/, render reports
with benchmarks/runners/plot_bench.py.
Run via srun on a compute node (see CLAUDE.md).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import triton

from miniworld_kernels.modules.exceptions import ImplementationType as IT
from miniworld_kernels.modules.triangle_attention import (
    BidirectionalTriangleAttention,
    TriangleAttention,
)

DEVICE = torch.device("cuda")


def _bench(fn, warmup=20, rep=100):
    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    except Exception as e:  # noqa: BLE001
        print(f"   bench fail: {type(e).__name__}: {str(e)[:70]}", flush=True)
        return float("nan")


def bench_compiled(model, args):
    fn = torch.compile(model)
    try:
        for _ in range(8):
            fn(*args)
        torch.cuda.synchronize()
        return _bench(lambda: fn(*args))
    except Exception as e:  # noqa: BLE001
        print(f"   compile fail: {type(e).__name__}: {str(e)[:70]}", flush=True)
        return float("nan")


def bench_cudagraph(model, args):
    try:
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    model(*args)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                model(*args)
        return _bench(g.replay)
    except Exception as e:  # noqa: BLE001
        print(f"   cudagraph fail: {type(e).__name__}: {str(e)[:70]}", flush=True)
        return float("nan")


def bench_train_compiled(model, pair, mask, dy):
    fn = torch.compile(model)
    p = pair.clone().requires_grad_(True)

    def step():
        model.zero_grad(set_to_none=True)
        fn(p, mask).backward(dy)

    try:
        for _ in range(8):
            step()
        torch.cuda.synchronize()
        return _bench(step)
    except Exception as e:  # noqa: BLE001
        print(f"   train fail: {type(e).__name__}: {str(e)[:70]}", flush=True)
        return float("nan")


def make(kind, d, nh, impl, dtype):
    if kind == "single":  # one (starting) bias-only triangle attention
        m = TriangleAttention(d, nh, starting=True, use_self_attention=False, implementation=impl)
    else:
        m = BidirectionalTriangleAttention(d, nh, implementation=impl)
    with torch.no_grad():
        m.to_out.weight.normal_(0, 0.02)
        m.to_gate.weight.normal_(0, 0.02)
    return m.to(DEVICE).to(dtype)


def run(kind, dims, dtype, B, nh):
    title = "single-dir" if kind == "single" else "bidirectional"
    print(f"### {title} bias-only triangle attention  B={B} H={nh} dtype={dtype}")
    print(f"# device={torch.cuda.get_device_name()}  inference=pt:compile/ours:cudagraph  train=compile")
    # Emit the benchmarks/runners/plot_bench.py format (M=L^2, d_in=d_out=d_pair; backend lines
    # `pytorch`/`triton` with fwd=inference, fwd+bwd=training) so the shared plotter
    # renders the standard speedup/latency charts + tables.
    print(f"host={torch.cuda.get_device_name()} torch={torch.__version__} dtype={dtype}")
    print("implementations=pytorch,triton  inference: pytorch=torch.compile / "
          "triton=CUDA-graph ; train: both torch.compile  use_self_attention=False")
    for d, Ls in dims:
        for L in Ls:
            pair = torch.randn(B, L, L, d, device=DEVICE, dtype=dtype)
            mask = torch.rand(B, L, device=DEVICE) > 0.2
            dy = torch.randn_like(pair)
            pt = make(kind, d, nh, IT.PYTORCH, dtype)
            ours = make(kind, d, nh, IT.TRITON, dtype)
            ours.load_state_dict(pt.state_dict())
            ipt = bench_compiled(pt, (pair, mask))
            iou = bench_cudagraph(ours, (pair, mask))
            tpt = bench_train_compiled(pt, pair, mask, dy)
            tou = bench_train_compiled(ours, pair, mask, dy)
            print(f"\n=== M={L * L}  d_in={d}  d_out={d} ===")
            print(f"# L={L} D={d}")
            print(f"pytorch fwd={ipt:.3f} ms fwd+bwd={tpt:.3f} ms")
            print(f"triton fwd={iou:.3f} ms fwd+bwd={tou:.3f} ms", flush=True)
            del pair, mask, dy, pt, ours
            torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    kind = sys.argv[1] if len(sys.argv) > 1 else "single"
    dims = [(128, [256, 384, 512, 768, 1024]),
            (256, [256, 384, 512, 768]),
            (512, [256, 384, 512])]
    run(kind, dims, torch.bfloat16, B=1, nh=4)
