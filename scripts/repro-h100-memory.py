"""Replay one H100 fault-adjacent schedule, with a fresh context per invocation.

Run under compute-sanitizer on an allocated H100. Inputs retain the failed build's
shapes, strides and dtypes; values are regenerated and outputs checked against torch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("attention", "adaln"))
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--shards", type=Path, default=Path(".scratch/h100-build/shards"))
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (9, 0), "An allocated H100 is required"
    torch.manual_seed(6950)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.family == "attention":
        from miniworld_engine.kernels.augmented_attention.triton.main import _attn_bwd
        fn = _attn_bwd.fn
        op = "augmented_attention_bwd_split_triton"
        pattern = "augmented_attention-*-corebfloat16-dims2-L128-*.json"
        cfg = {"BLOCK_M1": 32, "BLOCK_M2": 64}
    else:
        from miniworld_engine.kernels.adaln.triton.inference import (
            _adaln_gemm_gate_kernel,
        )
        fn = _adaln_gemm_gate_kernel.fn
        op = "adaln_gemm_gate_triton"
        pattern = "op-adaln_gemm_gate_triton-bfloat16-token-L5120-D768-H384-*.json"
        cfg = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 1}
    paths = list(args.shards.glob(pattern))
    assert len(paths) == 1, paths
    slot = json.loads(paths[0].read_text())[op]
    measurement = next(iter(next(iter(slot["measurements"].values())).values()))
    values = measurement["workload"]["arguments"] | cfg
    for name, value in list(values.items()):
        if isinstance(value, dict):
            dtype = getattr(torch, value["dtype"].removeprefix("torch."))
            tensor = torch.empty_strided(value["shape"], value["stride"], dtype=dtype, device="cuda")
            if dtype == torch.bool:
                tensor.fill_(True)
            else:
                tensor.normal_(std=0.125)
            values[name] = tensor
    if args.family == "attention":
        q, k, v, do = (values[n].float().transpose(2, 3) for n in ("Q", "K", "V", "DO"))
        scale = values["sm_scale"]
        logits = q @ k.transpose(-1, -2) * scale + values["Bias"].float()[None]
        p = logits.softmax(-1)
        dp = do @ v.transpose(-1, -2)
        delta = (p * dp).sum(-1)
        values["M"].copy_(logits.logsumexp(-1) * 1.44269504)
        values["D"].copy_(delta)
        ds = p * (dp - delta[..., None])
        rounded_ds = ds.to(values["Q"].dtype).float()
        expected = {
            "DQ": (rounded_ds @ k * scale).transpose(2, 3),
            "DK": (rounded_ds.transpose(-1, -2) @ q * scale).transpose(2, 3),
            "DV": (p.to(values["V"].dtype).float().transpose(-1, -2) @ do).transpose(2, 3),
            "DBias": ds,
        }
        for name in expected:
            values[name].zero_()
        grid = (triton.cdiv(values["N_CTX"], cfg["BLOCK_M2"]), values["A"], values["B"] * values["H"])
    else:
        c, sw, bw, x, sb = (values[n].float() for n in ("CondN", "SW", "BW", "X", "SB"))
        expected = {"Y": (c @ sw + sb).sigmoid() *
                    (x * values["Rstd"][:, None] - values["C1"][:, None]) + c @ bw}
        grid = (triton.cdiv(values["M"], cfg["BLOCK_M"]) * triton.cdiv(values["NX"], cfg["BLOCK_N"]),)
    torch.cuda.synchronize()
    print("LAUNCH", op, cfg, "warps", args.warps, "stages", args.stages, "grid", grid, flush=True)
    kernel = fn[grid](**values, num_warps=args.warps, num_stages=args.stages)
    torch.cuda.synchronize()
    print("LAUNCH OK shared=", kernel.metadata.shared, flush=True)
    for name, reference in expected.items():
        result = values[name].sum(0) if name == "DQ" else values[name]
        torch.testing.assert_close(result.float(), reference, atol=0.003, rtol=0.025)
    print("PASS numerical reference", flush=True)


if __name__ == "__main__":
    main()
