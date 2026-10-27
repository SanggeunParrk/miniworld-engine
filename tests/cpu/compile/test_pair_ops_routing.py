"""Where the three pair ops (`pair_embed_geom` / `pair_embed_ca`, `pair_embed_adaln`, `token_pair_init`) run on each card (`kernels/_pair_route.py`, `docs/dispatch.md`):

CUDA where a hand-written kernel serves the card (H100; for the token-pair initialisation also B200), the portable kernels on a card the engine holds a record for, the
PyTorch reference with one `UnrecordedCardWarning` on any other card, and never the portable kernels on H100 / B200 unless the internal A/B switch is on. No GPU is
needed: the card is faked (`tests/fake_card.py`) and the manifests are written into a temp directory."""
from __future__ import annotations

import warnings

import pytest
import torch
from fake_card import fake_card, write_manifest
from routing_helpers import clean_routing_unrecorded as _clean  # noqa: F401
from settings_helpers import (
    forced_triton as portable_switch,  # the internal A/B switch: the one place the tests name it
)

from miniworld_engine.autotune import records
from miniworld_engine.kernels import _pair_route as R
from miniworld_engine.modules import dispatch

OPS = ("pair_embed_geom", "pair_embed_ca", "pair_embed_adaln", "token_pair_init")
GEOM_CAPS = ((9, 0),)
TPI_CAPS = ((9, 0), (10, 0))
A6000 = ("NVIDIA RTX A6000", (8, 6))


def _grad_weight():
    return torch.zeros(4, requires_grad=True)


def test_the_kernels_of_each_op_by_mode():
    got = {op: {mode: sorted(records.triton_kernels(op, "bf16", mode)) for mode in ("inference", "training")} for op in OPS}
    assert got["pair_embed_geom"]["inference"] == ["pair_embed_geom_bins_triton", "pair_embed_geom_fwd_gather_triton"]
    assert got["pair_embed_ca"]["inference"] == ["pair_embed_ca_bins_triton", "pair_embed_geom_fwd_gather_triton"]
    for op in ("pair_embed_geom", "pair_embed_ca"):
        assert {"pair_embed_geom_dw_triton", "pair_embed_geom_bwd_reduce_triton", "pair_embed_geom_bwd_ln_triton"} <= set(got[op]["training"])
    assert got["pair_embed_adaln"]["inference"] == ["pair_embed_adaln_fwd_triton"]
    assert {"pair_embed_adaln_bwd_triton", "pair_embed_adaln_dw_triton", "pair_embed_adaln_bwd_reduce_triton"} <= set(got["pair_embed_adaln"]["training"])
    assert got["token_pair_init"]["inference"] == ["token_pair_init_fwd_triton"]
    assert {"token_pair_init_bwd_dx_triton", "token_pair_init_dw_triton", "token_pair_init_dw_reduce_triton"} <= set(got["token_pair_init"]["training"])


@pytest.mark.parametrize("op", OPS)
def test_h100_runs_cuda_and_b200_only_where_a_cuda_kernel_exists(monkeypatch, op):
    caps = TPI_CAPS if op == "token_pair_init" else GEOM_CAPS
    dev = fake_card(monkeypatch, "NVIDIA H100 80GB HBM3", (9, 0))
    assert R.route(op, dev, _grad_weight(), cuda_caps=caps) == R.CUDA
    dev = fake_card(monkeypatch, "NVIDIA B200", (10, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # B200 without a CUDA kernel of the op: the reference, silently
        got = R.route(op, dev, _grad_weight(), cuda_caps=caps)
    assert got == (R.CUDA if op == "token_pair_init" else R.REFERENCE)
    assert (R.restricted_reason(dev, caps) is None) == (op == "token_pair_init")


@pytest.mark.parametrize("op", OPS)
def test_the_ab_switch_puts_every_card_on_the_portable_kernels(monkeypatch, op):
    for name, cap in (("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0)), A6000):
        dev = fake_card(monkeypatch, name, cap)
        with portable_switch():
            assert R.route(op, dev, _grad_weight()) == R.PORTABLE
            assert R.restricted_reason(dev, GEOM_CAPS) is None
        assert R.cuda_served(dev, GEOM_CAPS) == (cap == (9, 0))


@pytest.mark.parametrize("op", OPS)
def test_an_unrecorded_card_warns_once_and_runs_the_reference(monkeypatch, op, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    with records.use_manifest_dir(tmp_path):                                  # no manifest at all
        with pytest.warns(dispatch.UnrecordedCardWarning, match=op):
            assert R.route(op, dev, cuda_caps=GEOM_CAPS) == R.REFERENCE
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                    # once per (op, card, mode)
            assert R.route(op, dev, cuda_caps=GEOM_CAPS) == R.REFERENCE


@pytest.mark.parametrize("op", OPS)
def test_a_recorded_card_runs_the_portable_kernels(monkeypatch, op, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    write_manifest(tmp_path, *A6000, OPS)
    with records.use_manifest_dir(tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert R.route(op, dev, _grad_weight(), cuda_caps=GEOM_CAPS) == R.PORTABLE
            assert R.route(op, dev, cuda_caps=GEOM_CAPS) == R.PORTABLE


@pytest.mark.parametrize("op", OPS)
def test_a_card_recorded_for_inference_only_runs_the_reference_for_a_call_with_a_backward(monkeypatch, op, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    write_manifest(tmp_path, *A6000, OPS, training=False)
    with records.use_manifest_dir(tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert R.route(op, dev, torch.zeros(4), cuda_caps=GEOM_CAPS) == R.PORTABLE      # no gradient needed
        with pytest.warns(dispatch.UnrecordedCardWarning, match="training"):
            assert R.route(op, dev, _grad_weight(), cuda_caps=GEOM_CAPS) == R.REFERENCE
        with torch.no_grad():
            assert R.route(op, dev, _grad_weight(), cuda_caps=GEOM_CAPS) == R.PORTABLE      # grad mode off: inference


def test_one_failed_kernel_blocks_only_the_ops_that_launch_it(monkeypatch, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    write_manifest(tmp_path, *A6000, OPS, skip=frozenset({"pair_embed_adaln_fwd_triton"}))
    with records.use_manifest_dir(tmp_path):
        assert R.route("pair_embed_geom", dev, cuda_caps=GEOM_CAPS) == R.PORTABLE
        with pytest.warns(dispatch.UnrecordedCardWarning, match="pair_embed_adaln_fwd_triton"):
            assert R.route("pair_embed_adaln", dev, cuda_caps=GEOM_CAPS) == R.REFERENCE
        assert R.route("token_pair_init", dev, cuda_caps=GEOM_CAPS) == R.PORTABLE


def test_a_tensor_that_is_not_on_a_card_is_the_reference(monkeypatch):
    fake_card(monkeypatch, "NVIDIA H100 80GB HBM3", (9, 0))
    with portable_switch():
        assert R.route("pair_embed_geom", torch.device("cpu")) == R.REFERENCE


def test_the_entry_points_fall_back_to_the_reference_on_cpu_tensors():
    """`pair_embed_geom` & co. on CPU tensors (no CUDA device anywhere) are the PyTorch reference, bf16 out for the two pair builders."""
    from miniworld_engine.kernels.pair_embed_adaln import (
        pair_embed_adaln,
        pair_embed_adaln_reference,
    )
    from miniworld_engine.kernels.pair_embed_geom import (
        pair_embed_ca,
        pair_embed_ca_reference,
        pair_embed_geom,
        pair_embed_geom_reference,
    )
    from miniworld_engine.kernels.token_pair_init import (
        token_pair_init,
        token_pair_init_reference,
    )

    g = torch.Generator().manual_seed(0)
    B, n = 2, 12
    ca = torch.cumsum(torch.randn(B, n, 3, generator=g) * 0.4, 1)
    bb = torch.stack([ca + 0.1, ca, ca - 0.1, ca + 0.2], 2)
    mask = torch.ones(B, n, dtype=torch.bool)
    mask[1, 9:] = False
    idx = torch.arange(1, n + 1).expand(B, n)
    chains = torch.zeros(B, n, dtype=torch.long)
    w = torch.randn(256, 317, generator=g) * 0.3
    out = pair_embed_geom(w, bb, mask, mask, mask, idx, chains)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, pair_embed_geom_reference(w, bb, mask, mask, mask, idx, chains).bfloat16())
    wc = torch.randn(256, 157, generator=g) * 0.3
    assert torch.equal(pair_embed_ca(wc, ca, mask), pair_embed_ca_reference(wc, ca, mask).bfloat16())
    gamma, beta = torch.sigmoid(torch.randn(B, 256, generator=g)), torch.randn(B, 256, generator=g) * 0.1
    wa, lw, lb = torch.randn(256, 187, generator=g) * 0.3, torch.ones(256), torch.zeros(256)
    got = pair_embed_adaln(wa, lw, lb, gamma, beta, ca, ca, idx, mask)
    assert torch.equal(got, pair_embed_adaln_reference(wa, lw, lb, gamma, beta, ca, ca, idx, mask).bfloat16())
    ids = [torch.randint(0, 3, (B, n), generator=g) for _ in range(5)]
    left, right = torch.randn(B, n, 128, generator=g), torch.randn(B, n, 128, generator=g)
    w_rel, w_bond = torch.randn(128, 139, generator=g), torch.randn(128, 2, generator=g)
    bond = torch.rand(B, n, n, generator=g) < 0.1
    assert torch.equal(token_pair_init(left, right, w_rel, w_bond, *ids, bond), token_pair_init_reference(left, right, w_rel, w_bond, *ids, bond))
