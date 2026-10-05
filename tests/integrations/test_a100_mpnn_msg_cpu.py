"""The gates of the A100 MPNN message-side CUDA paths (integrations/mpnn_msg_sm80.py) without a GPU: nothing is served off a CUDA device, ``MINIWORLD_MPNN_MSG_SM80=0`` and a forced Triton engine
backend switch the paths off, and the Triton / PyTorch paths keep serving a CPU call."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import mpnn_msg_sm80 as sm80
from miniworld_engine.kernels.mpnn_message import (
    message_hidden_reduce,
    message_hidden_reduce_pytorch,
)
from miniworld_engine.kernels.mpnn_relative_position import (
    relative_position_embed,
    relative_position_embed_pytorch,
)


def test_nothing_is_served_off_a_cuda_device():
    p = torch.zeros(2, 48, 128, dtype=torch.bfloat16)
    assert not sm80.serves_message(p, "auto")
    assert not sm80.serves_node(p, "triton_compute")
    assert not sm80.serves_relpos(torch.zeros(10, dtype=torch.long), torch.zeros(66, 16), torch.zeros(16))


def test_the_switch_and_the_engine_backend_turn_the_paths_off(monkeypatch):
    monkeypatch.delenv("MINIWORLD_MPNN_MSG_SM80", raising=False)
    assert sm80._enabled() == (settings.current().engine_backend != "triton")
    monkeypatch.setenv("MINIWORLD_MPNN_MSG_SM80", "0")
    assert not sm80._enabled()
    monkeypatch.delenv("MINIWORLD_MPNN_MSG_SM80")
    previous = settings.configure(engine_backend="triton")
    try:
        assert not sm80._enabled()
    finally:
        settings.configure(engine_backend=previous.engine_backend)


def test_the_message_and_relative_position_references_serve_a_cpu_call():
    g = torch.Generator().manual_seed(0)
    p = torch.randn(3, 48, 128, generator=g)
    w = torch.randn(128, 128, generator=g) / 128**0.5
    b = torch.randn(128, generator=g) * 0.1
    mask = (torch.rand(3, 48, generator=g) > 0.2).float()
    out = message_hidden_reduce(p, w, b, mask, 48, backend="pytorch")
    torch.testing.assert_close(out, message_hidden_reduce_pytorch(p, w, b, mask, 48))
    bucket = torch.randint(0, 66, (5, 7), generator=g)
    table, bias = torch.randn(66, 16, generator=g), torch.randn(16, generator=g)
    torch.testing.assert_close(relative_position_embed(bucket, table, bias, backend="off"), relative_position_embed_pytorch(bucket, table, bias))


@pytest.mark.parametrize("value", ["1", ""])
def test_any_other_switch_value_leaves_the_path_on(monkeypatch, value):
    monkeypatch.setenv("MINIWORLD_MPNN_MSG_SM80", value)
    assert sm80._enabled() == (settings.current().engine_backend != "triton")
