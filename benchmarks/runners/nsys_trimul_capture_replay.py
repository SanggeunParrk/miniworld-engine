from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while ROOT.name != "miniworld-kernels" and ROOT.parent != ROOT:
    ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from lightning import Fabric

from benchmarks.runners.bench import (
    MiniWorldTriangleMultiplicationInference,
    MiniWorldTriangleMultiplicationTraining,
    capture_cudagraph,
)
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication


class MultiMiniWorldTriangleMultiplication(nn.Module):
    def __init__(self, *, d_pair: int, n_layers: int, mode: str) -> None:
        super().__init__()
        torch.manual_seed(0)
        layers = []
        layer_cls = (
            MiniWorldTriangleMultiplicationInference
            if mode == "inference"
            else MiniWorldTriangleMultiplicationTraining
        )
        for _ in range(n_layers):
            base = TriangleMultiplication(d_pair)
            for linear in (
                base.to_left,
                base.to_left_gate,
                base.to_right,
                base.to_right_gate,
                base.to_gate,
                base.to_out,
            ):
                nn.init.normal_(linear.weight, std=d_pair**-0.5)
            layers.append(layer_cls(base))
        self.layers = nn.ModuleList(layers)

    def forward(self, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            pair = layer(pair, mask)
        return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inference", "training"], required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--d-pair", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--mask-prob", type=float, default=0.0)
    parser.add_argument("--reps", type=int, default=25)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    fabric = Fabric(precision="bf16-mixed", accelerator="cuda", devices=1)
    fabric.launch()

    model = MultiMiniWorldTriangleMultiplication(
        d_pair=args.d_pair,
        n_layers=args.n_layers,
        mode=args.mode,
    ).cuda()

    torch.manual_seed(1)
    pair = torch.randn(
        1,
        args.seq_len,
        args.seq_len,
        args.d_pair,
        device="cuda",
        dtype=torch.bfloat16,
    )
    pair.requires_grad_(args.mode == "training")
    dy = torch.randn_like(pair)
    mask = torch.rand(1, args.seq_len, device="cuda") > args.mask_prob

    def inference_step() -> torch.Tensor:
        return model(pair, mask)

    def training_step() -> None:
        y = inference_step()
        fabric.backward(y, dy)

    step = inference_step if args.mode == "inference" else training_step
    params = [p for p in model.parameters() if p.requires_grad]
    graph = capture_cudagraph(step, params, is_train=args.mode == "training")

    # Keep all graph-capture setup and warmup outside the profiler range. The nsys run should
    # use --capture-range=cudaProfilerApi so only these replays are aggregated.
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.reps):
        graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(
        f"captured {args.reps} graph replays: mode={args.mode} "
        f"L={args.seq_len} d_pair={args.d_pair}",
        flush=True,
    )


if __name__ == "__main__":
    main()
