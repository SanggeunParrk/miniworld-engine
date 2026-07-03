"""nsys capture-replay driver for the Transition module (miniworld impl), mirroring
nsys_trimul_capture_replay.py. Captures fwd (inference) or fwd+bwd (training) in a manual CUDA
graph and replays under a cudaProfilerApi range so nsys aggregates only the replays.

The bench maps the "miniworld" transition implementation to ImplementationType.CUEQUIVARIANCE,
which dispatches to kernels.transition.{triton,cute}.fused by d_pair. Run with:
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node ...
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
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.transition import Transition


class MultiTransition(nn.Module):
    def __init__(self, *, d_pair: int, n_layers: int, impl: ImplementationType) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList(
            [Transition(d_pair, implementation=impl).to(torch.bfloat16) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inference", "training"], required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--d-pair", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--reps", type=int, default=25)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    fabric = Fabric(precision="bf16-mixed", accelerator="cuda", devices=1)
    fabric.launch()

    # "miniworld" transition == CUEQUIVARIANCE impl (dispatches triton/cute fused by d_pair)
    model = MultiTransition(d_pair=args.d_pair, n_layers=args.n_layers,
                            impl=ImplementationType.CUEQUIVARIANCE).cuda()
    torch.manual_seed(1)
    x = torch.randn(1, args.seq_len, args.seq_len, args.d_pair,
                    device="cuda", dtype=torch.bfloat16)
    x.requires_grad_(args.mode == "training")
    dy = torch.randn_like(x)

    def inference_step() -> torch.Tensor:
        return model(x)

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
    print(f"captured {args.reps} replays: transition mode={args.mode} "
          f"L={args.seq_len} d_pair={args.d_pair}", flush=True)


if __name__ == "__main__":
    main()
