"""RTX 5090 (``5090_sm120``): the fp32 TRAINING Transition / LayerNorm Triton kernels at the rows an augment chunk of 8 / 12 / 24 samples x 384 tokens
reaches (2048 / 4096 / 8192 launch buckets, width 384) used to find no table entry: the kernel default config (a 128 x 64 fp32 tile, three stages,
131072 B of shared memory) then raised ``OutOfResources`` against the card's 101376 B. The tables now hold entries for those keys; this test
runs the step with TF32 off (the diffusion head's fp32 setting, IEEE dots) and asks for no launch-config miss, no error, and the error against a
float64 gold inside the engine standard ``max(err_pytorch, 8e-3)``."""

from __future__ import annotations

import copy
import warnings

import pytest
import torch

from miniworld_engine.kernels import _launch_config
from miniworld_engine.measure import config as C
from miniworld_engine.modules.norm import LayerNorm
from miniworld_engine.modules.transition import Transition

pytestmark = [pytest.mark.gpu, pytest.mark.rtx5090]

#: the 5090 table keys that were missing (transition, layernorm families)
CHUNKS = (8, 12, 24)


def _run(module: torch.nn.Module, x: torch.Tensor, up: torch.Tensor, backend: str) -> dict[str, torch.Tensor]:
    module.set_backend(backend)
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    y = module(x)
    y.backward(up.to(y.dtype))
    grads = {n: p.grad for n, p in module.named_parameters() if p.grad is not None}
    return {"y": y.detach(), "dx": x.grad, **grads}


def _err(a: torch.Tensor, gold: torch.Tensor) -> float:
    return float((a.double() - gold).abs().max() / gold.abs().max().clamp_min(1e-30))


@pytest.fixture(autouse=True)
def _ieee_fp32():
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    _launch_config._WARNED.clear()
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


@pytest.mark.parametrize("chunk", CHUNKS)
@pytest.mark.parametrize(
    ("make", "width"),
    [(lambda w: Transition(w, 2), 384), (lambda w: LayerNorm(w), 384), (lambda w: LayerNorm(w), 768)],
    ids=["transition384", "layernorm384", "layernorm768"],
)
def test_train_step_fits_shared_memory_and_hits_the_table(chunk, make, width):
    assert _launch_config.gpu_key() == "5090_sm120"
    torch.manual_seed(0)
    module = make(width).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    x, up = (torch.randn(chunk, 384, width, device="cuda") for _ in range(2))
    gold_module = copy.deepcopy(module).double()
    gold = _run(gold_module, x.double(), up.double(), "pytorch")
    reference = _run(module, x, up, "pytorch")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(module, x, up, "triton")  # OutOfResources here before the table entries
    misses = [str(w.message) for w in caught if "launch_config" in str(w.message)]
    assert not misses, misses
    for name, value in out.items():
        tol = max(_err(reference[name], gold[name]), 8e-3)
        assert torch.isfinite(value).all(), name
        assert _err(value, gold[name]) <= tol, name


def test_table_keys_for_the_chunk_rows_exist():
    """The keys themselves (a table edit that drops one fails here without running a kernel)."""
    from miniworld_engine.kernels.transition.triton import main

    keys = main.transition_fwd_kernel._entries("5090_sm120")
    flags = {(dt, flag) for (dt, _, flag) in keys}
    assert ("float32", (("HALF", 0), ("STORE_AB", 1))) in flags
    assert ("float16+float32", (("HALF", 1), ("STORE_AB", 1))) in flags
