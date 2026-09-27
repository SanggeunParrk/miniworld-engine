"""Inference integration contracts; no new backward implementation."""
import pytest
import torch
from contextlib import contextmanager

from miniworld_engine.integrations import anthropic as A
from miniworld_engine.modules import Transition, TriangleMultiplication, TriangleAttention
from miniworld_engine.modules.dispatch import resolve, KernelBackend
from miniworld_engine import settings


@contextmanager
def override(**kw):
    saved=settings.configure(**kw)
    try:
        yield
    finally:
        settings.configure(**vars(saved))


@pytest.mark.parametrize("op", ["transition", "triangle_multiplication", "triangle_attention"])
def test_explicit_dispatch(op):
    with override(engine_backend="auto"):
        assert resolve(op, "anthropic") == KernelBackend.ANTHROPIC


def test_no_silent_unsupported_backend():
    with override(engine_backend="auto"):
        with pytest.raises(ValueError, match="no Anthropic module adapter"):
            resolve("adaptive_layernorm", "anthropic")
    with override(engine_backend="triton"):
        with pytest.raises(ValueError, match="conflicts"):
            resolve("transition", "anthropic")


@pytest.mark.parametrize("cls", [Transition, TriangleMultiplication, TriangleAttention])
def test_grad_rejected_before_import_or_launch(cls):
    with override(engine_backend="auto"):
        module = cls(128, implementation="anthropic").eval()
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        module(torch.randn(1, 2, 2, 128))


def test_transition_weight_cache_invalidates(monkeypatch):
    with override(engine_backend="auto"):
        m=Transition(128, implementation="anthropic").eval()
    packs=[]
    def pack(**kw):
        packs.append(kw)
        return kw
    monkeypatch.setattr(A, 'pack_transition', pack)
    monkeypatch.setattr(A, 'transition', lambda x,w,**kw:(x, 'test'))
    x=torch.randn(1,2,2,128)
    with torch.no_grad():
        m(x);m(x)
        assert len(packs)==1
        m.expand_a.weight.add_(.1)
        m(x)
        assert len(packs)==2
        m.load_state_dict(m.state_dict())
        m(x)
        assert len(packs)==3


def test_trimul_dropout_refused_before_source_load():
    with override(engine_backend="auto"):
        m=TriangleMultiplication(128,implementation="anthropic").train()
    with torch.no_grad(),pytest.raises(RuntimeError,match="eval"):
        m(torch.randn(1,2,2,128))


def test_block_qk_norm_refused_before_source_load():
    with override(engine_backend="auto"):
        m=TriangleAttention(128, implementation="anthropic", use_qk_norm=True,
                            anthropic_row="block:triattn_native").eval()
    with torch.no_grad(), pytest.raises(ValueError, match="Q/K RMSNorm"):
        m(torch.randn(1,2,2,128))


def test_unknown_operation_rejected():
    with pytest.raises(ValueError, match="Unknown upstream operation"):
        A.operation("unregistered")
