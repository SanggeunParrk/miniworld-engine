"""nsys capture-replay driver for BIDIRECTIONAL trimul (BidirV6TriMul), mirroring the single-dir
diagnostics/nsys_trimul_capture_replay.py. Captures fwd (inference) or fwd+bwd (training) in a
manual CUDA graph and replays under a cudaProfilerApi range so nsys aggregates only the replays.
Run with: nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while ROOT.name != "miniworld-kernels" and ROOT.parent != ROOT:
    ROOT = ROOT.parent
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn
from lightning import Fabric

from benchmarks.runners.bench import capture_cudagraph
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import BidirV6TriMul
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)


class MultiBidir(nn.Module):
    def __init__(self, *, d_pair: int, n_layers: int) -> None:
        super().__init__()
        torch.manual_seed(0)
        layers = [BidirV6TriMul(BidirectionalTriangleMultiplication(d_pair).to(torch.bfloat16))
                  for _ in range(n_layers)]
        self.layers = nn.ModuleList(layers)

    def forward(self, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            pair = layer(pair, mask)
        return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inference", "training"], required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-pair", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--reps", type=int, default=25)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    fabric = Fabric(precision="bf16-mixed", accelerator="cuda", devices=1)
    fabric.launch()

    model = MultiBidir(d_pair=args.d_pair, n_layers=args.n_layers).cuda()
    torch.manual_seed(1)
    pair = torch.randn(1, args.seq_len, args.seq_len, args.d_pair,
                       device="cuda", dtype=torch.bfloat16)
    pair.requires_grad_(args.mode == "training")
    dy = torch.randn_like(pair)
    mask = torch.ones(1, args.seq_len, device="cuda", dtype=torch.bool)

    def inference_step() -> torch.Tensor:
        return model(pair, mask)

    def training_step() -> None:
        fabric.backward(inference_step(), dy)

    step = inference_step if args.mode == "inference" else training_step
    params = [p for p in model.parameters() if p.requires_grad]
    graph = capture_cudagraph(step, params, is_train=args.mode == "training")

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.reps):
        graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"captured {args.reps} replays: bidir mode={args.mode} "
          f"L={args.seq_len} d_pair={args.d_pair}", flush=True)


if __name__ == "__main__":
    main()
