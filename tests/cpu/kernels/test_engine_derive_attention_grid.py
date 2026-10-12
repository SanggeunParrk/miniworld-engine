"""The cache planner must record attention backward without launching a GPU kernel."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("chunked", [False, True])
def test_attention_backward_derivation_runs_grid_callback(chunked):
    # compile_wrap is read at import time. A subprocess keeps recorder monkey
    # patches and opaque-op registration isolated from normal kernel tests.
    code = """
import torch
from miniworld_engine.autotune import derive
from miniworld_engine.kernels.augmented_attention.triton import main
sink = []
derive.install_recorder(sink)
main._CHUNK_WORKSPACE_BYTES = 1
a, b, length, heads, dim = 3, 2, 259, 4, 32
q = torch.empty(a, b, length, heads, dim, device="meta", dtype=torch.bfloat16)
bias = torch.empty(b, heads, length, length, device="meta", dtype=q.dtype)
mask = torch.ones(a, b, length, device="meta", dtype=torch.bool)
m = torch.empty(a, b, heads, length, device="meta", dtype=torch.float32)
fn = main._aa_bwd_chunked if CHUNKED else main._aa_bwd
dq, dk, dv, dbias = fn(q, q, q, q, bias, mask, q, m, 0)
assert dq.shape == q.shape and dk.shape == q.shape and dv.shape == q.shape
assert dbias.shape == ((b, heads, length, length) if CHUNKED else (a, b, heads, length, length))
assert len(sink) == (7 if CHUNKED else 3), sink
assert all(not op.startswith("<") and not bucket.startswith("<") for op, dtype, bucket in sink), sink
assert all(dtype for op, dtype, bucket in sink)
print(sink)
""".replace("CHUNKED", repr(chunked))
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MINIWORLD_COMPILE_WRAP": "disable"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
