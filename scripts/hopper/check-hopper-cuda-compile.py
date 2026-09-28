"""CPU-node-only nvcc compilation; does not allocate or launch a CUDA tensor."""
from miniworld_engine.autotune.hopper_cuda_config import candidates
from miniworld_engine.kernels.transition.cuda import _ext

for kind in ("b2b", "expand_gate", "gatebwd"):
    widths = (128, 256) if kind == "b2b" else (128, 256, 512)
    for width in widths:
        config = candidates(kind, width)[0]
        print(f"compile {kind} K={width} {config}", flush=True)
        _ext(kind, width, config)
        print("PASS", flush=True)
