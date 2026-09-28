"""Compile the schedules adjacent to H100 build faults without a CUDA device.

This is codegen evidence only. Execute the same candidates under Compute Sanitizer
on an allocated H100 before claiming that a device memory fault is repaired.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from miniworld_engine.kernels.adaln.triton.inference import _adaln_gemm_gate_kernel
    from miniworld_engine.kernels.augmented_attention.triton.main import _attn_bwd

    cases = [
        ("attention", "augmented_attention-*-corebfloat16-dims2-L128-*.json",
         "augmented_attention_bwd_split_triton", _attn_bwd.fn,
         {"BLOCK_M1": 32, "BLOCK_M2": 64}),
        ("adaln", "op-adaln_gemm_gate_triton-bfloat16-token-L5120-D768-H384-*.json",
         "adaln_gemm_gate_triton", _adaln_gemm_gate_kernel.fn,
         {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 1}),
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, pattern, op, fn, cfg in cases:
        paths = list(args.shards.glob(pattern))
        assert len(paths) == 1, paths
        slot = json.loads(paths[0].read_text())[op]
        measurement = next(iter(next(iter(slot["measurements"].values())).values()))
        values = measurement["workload"]["arguments"] | cfg
        signature, constants, attrs = {}, {}, {}
        for param in fn.params:
            value = values[param.name]
            if param.is_constexpr:
                signature[param.name] = "constexpr"
                constants[param.name] = value
            elif isinstance(value, dict):
                dtype = {"torch.bfloat16": "*bf16", "torch.float32": "*fp32",
                         "torch.bool": "*i1"}[value["dtype"]]
                signature[param.name] = dtype
                attrs[(param.num,)] = [["tt.divisibility", 16]]
            elif isinstance(value, int):
                if value == 1:
                    signature[param.name] = "constexpr"
                    constants[param.name] = value
                else:
                    signature[param.name] = "i32" if -(2**31) <= value < 2**31 else "i64"
                    if value % 16 == 0:
                        attrs[(param.num,)] = [["tt.divisibility", 16]]
            else:
                signature[param.name] = "fp32"
        for warps in (2, 4):
            for stages in (1, 2):
                source = ASTSource(fn, signature, constexprs=constants, attrs=attrs)
                kernel = compile(source, target=GPUTarget("cuda", 90, 32),
                                 options={"num_warps": warps, "num_stages": stages})
                stem = f"{name}-w{warps}-s{stages}"
                for ext in ("ptx", "ttgir"):
                    (args.out / f"{stem}.{ext}").write_text(kernel.asm[ext])
                print(stem, "shared", kernel.metadata.shared,
                      "wgmma", kernel.asm["ptx"].count("wgmma.mma_async"), flush=True)


if __name__ == "__main__":
    main()
