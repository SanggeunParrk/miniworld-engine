"""Compile the Hopper fused LayerNormLinear on a CPU compute node; launch nothing.

Run with CUTE_DSL_ARCH=sm_90a in the repository's pinned environment.
This checks CuTe lowering/compilation, not numerical results or GPU synchronization.
"""
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import cutlass

from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    _compile_fused,
)


def main():
    # Cover independent branches without rerunning identical compilations.
    cases = [
        ("default", "k", (128, 128, 1, 1, True), False),
        ("gate", "k", (128, 128, 1, 1, True), True),
        ("m-major", "m", (128, 128, 1, 1, True), False),
        ("m-major-gate", "m", (128, 128, 1, 1, True), True),
        ("non-pingpong", "k", (64, 128, 1, 1, False), False),
        ("non-pingpong-gate", "k", (64, 128, 1, 1, False), True),
    ]
    for name, a_major, cfg, gate in cases:
        # Bypass quack's persistent cache: this audit must lower the current source.
        _compile_fused.__wrapped__(
            cutlass.BFloat16, cutlass.BFloat16, cutlass.BFloat16,
            a_major, "k", "n", cutlass.Float32, (9, 0), cfg,
            cutlass.BFloat16 if gate else None, "n" if gate else None,
        )
        print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
