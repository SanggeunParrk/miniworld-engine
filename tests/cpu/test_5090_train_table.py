"""The 5090_sm120 launch-config tables hold the fp32 training keys of an augment chunk of 8 / 12 / 24 samples x 384 tokens (width 384, 2048 / 4096 /
8192 row buckets). Without them the Transition expand kernel ran its default config (131072 B of shared memory) and raised ``OutOfResources`` against
the card's 101376 B; a stored config was timed on the card (a config that does not fit is recorded failed by the pretuner and never stored)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

KERNELS = Path(__file__).resolve().parents[2] / "src" / "miniworld_engine" / "kernels"

KEYS = {
    ("transition", "transition_expand_swiglu_triton"): [
        "float32|K=384,M=2048,ND=768|HALF=0,STORE_AB=1",
        "float32|K=384,M=4096,ND=768|HALF=0,STORE_AB=0",  # the sampler conditioning (20 samples x 384 rows, no grad)
        "float32|K=384,M=8192,ND=768|HALF=0,STORE_AB=0",
        "float32|K=384,M=1024,ND=768|HALF=0,STORE_AB=1",
        "float16+float32|K=384,M=4096,ND=768|HALF=1,STORE_AB=1",
        "float16+float32|K=384,M=8192,ND=768|HALF=1,STORE_AB=1",
    ],
    ("transition", "transition_bwd_squeeze_swiglu_triton"): [
        "float32|K=384,M=2048,ND=768|HALF=0,SAVED=1",
        "float16+float32|K=384,M=4096,ND=768|HALF=1,SAVED=1",
        "float16+float32|K=384,M=8192,ND=768|HALF=1,SAVED=1",
    ],
    ("transition", "transition_squeeze_residual_triton"): ["float32|K=768,M=4096,N=384", "float32|K=768,M=8192,N=384"],
    ("layernorm", "layernorm_fwd_saveact_triton"): ["float32|M=4096,N=384|HAS_ROWSCALE=0", "float32|M=8192,N=384|HAS_ROWSCALE=0"],
    ("layernorm", "layernorm_bwd_atomic_triton"): [
        f"float32|M={m},N={n}|HAS_ROWSCALE=0" for m in (1024, 2048, 4096, 8192) for n in (384, 768)
    ],
}


@pytest.mark.parametrize(("family", "op"), sorted(KEYS))
def test_chunk_keys_present(family, op):
    table = json.loads((KERNELS / family / "configs" / "5090_sm120.json").read_text())["ops"][op]
    for key in KEYS[family, op]:
        assert table.get(key), key
        assert all({"BLOCK_M1", "num_warps", "num_stages"} <= {*c["kwargs"], *c} for c in table[key]), key
