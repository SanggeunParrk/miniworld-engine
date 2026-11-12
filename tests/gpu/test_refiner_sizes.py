"""FoldForge OpenDDE's structural-token refiner (5I28 n = 12 / 16: S = 2976 / 3968 tokens) at d_pair 384: TriangleAttention (12 heads) and
AttentionPairBias (8 heads, the refiner's extra_pair_bias) on the ``cuda`` / ``triton`` backends.

The float64 / float32 references do not fit at S (the accuracy measurer records those cases as skipped), so the refiner sizes are checked by a
symmetry against the float32 reference at P = S / R: tiling the inputs R x R over the token axes leaves every P x P block of the output equal
to the output at P (each query sees its P keys R times with the same logits: the softmax weights divide by R and the R copies sum back).
The tolerance is the accuracy measurer's GREEN bound at P: ``max(err_pytorch, 8e-3)`` (the PyTorch reference at the case dtype against the
float32 gold).

The row-blocked paths (the A100 wide inference ``sm80_wide.forward_rows`` and the module's ``_attention_rows`` for the Triton core) are also
compared with the whole-pair calls at a small length, the thresholds lowered.
"""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import AttentionPairBias, TriangleAttention
from miniworld_engine.modules.attention import EXTRA_BIAS_STREAM

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

FIXED = 8e-3  # the measurer's fixed tolerance for bf16 / master
SIZES = [(186, 16), (248, 16)]  # (P, R): S = 2976 (n = 12), 3968 (n = 16)


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _err(x: torch.Tensor, gold: torch.Tensor) -> float:
    return float((x.float() - gold).abs().max() / gold.abs().max())


def _triattn(starting: bool) -> TriangleAttention:
    m = TriangleAttention(384, 12, starting=starting).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    return m


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize(("P", "R"), SIZES)
def test_triangle_attention_refiner_sizes_tiled(backend: str, starting: bool, P: int, R: int) -> None:
    base = _triattn(starting)
    gen = torch.Generator(device="cuda").manual_seed(1)
    small = base.make_inputs(length=P, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    small["mask"][:, P - 5 :] = False  # a masked tail, repeated in every copy
    with torch.no_grad():
        gold = copy.deepcopy(base).float()(small["pair"].float(), small["mask"]).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            err_pt = _err(copy.deepcopy(base)(**small), gold)
        kern = copy.deepcopy(base).set_backend(backend)
        pair = small["pair"].repeat(1, R, R, 1).contiguous()
        mask = small["mask"].repeat(1, R)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert kern.unsupported(pair, mask) is None
            big = kern(pair, mask)
        del pair
    tol = max(err_pt, FIXED)
    for i in (0, R // 2, R - 1):
        for j in (0, R - 1):
            err = _err(big[:, i * P : (i + 1) * P, j * P : (j + 1) * P], gold)
            assert err <= tol, f"block ({i}, {j}): {err:.3e} > tol {tol:.3e} (pytorch at P {err_pt:.3e})"


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize(("P", "R"), SIZES)
def test_attention_pair_bias_refiner_sizes_tiled(backend: str, P: int, R: int) -> None:
    base = AttentionPairBias(384, 384, n_head=8).cuda().float().eval()
    C.randomize(base, torch.Generator(device="cuda").manual_seed(0))
    base.set_master_weights()
    gen = torch.Generator(device="cuda").manual_seed(1)
    small = base.make_inputs(length=P, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen, stream=EXTRA_BIAS_STREAM)
    assert "extra_pair_bias" in small
    small["mask"][:, P - 5 :] = False
    extra = small["extra_pair_bias"][0, 0]  # [P, P], expanded over the heads as FoldForge hands it
    with torch.no_grad():
        gold = copy.deepcopy(base).float()(**{k: v.float() if v.is_floating_point() else v for k, v in small.items()}).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            err_pt = _err(copy.deepcopy(base)(**small), gold)
        kern = copy.deepcopy(base).set_backend(backend)
        big_in = {
            "single": small["single"].repeat(1, R, 1).contiguous(),
            "pair": small["pair"].repeat(1, R, R, 1).contiguous(),
            "mask": small["mask"].repeat(1, R),
            "extra_pair_bias": extra.repeat(R, R)[None, None].expand(1, base.n_head, P * R, P * R),
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert kern.unsupported(**big_in) is None
            big = kern(**big_in)
        del big_in
    tol = max(err_pt, FIXED)
    for i in (0, R // 2, R - 1):
        err = _err(big[:, i * P : (i + 1) * P], gold)
        assert err <= tol, f"block {i}: {err:.3e} > tol {tol:.3e} (pytorch at P {err_pt:.3e})"


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize("starting", [True, False])
def test_triangle_attention_triton_rows_match_whole(starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_attention_rows`` (blocks of the attended frame's rows, the Triton core on each block against the whole bias) against the
    whole-pair composition: the same statements per row."""
    base = _triattn(starting)
    kern = copy.deepcopy(base).set_backend("triton")
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = base.make_inputs(length=384, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    inputs["mask"][:, -9:] = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert not kern._rows_blocked(inputs["pair"])
        whole = kern(**inputs)
        monkeypatch.setattr(type(kern), "ROWS_BLOCKED_BYTES", 0)
        monkeypatch.setattr(type(kern), "ROWS_BLOCK_BYTES", 100 * 384 * 384 * 2)  # 100-row blocks: a partial last block
        assert kern._rows_blocked(inputs["pair"])
        rows = kern(**inputs)
    torch.testing.assert_close(rows.float(), whole.float(), rtol=0, atol=2e-2)
    assert _err(rows, whole.float()) < 4e-3


@pytest.mark.skipif(_card() != (8, 0), reason="A100 (the sm_80 wide path)")
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("masked", [True, False])
def test_triangle_attention_sm80_rows_match_whole(starting: bool, masked: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """``sm80_wide.forward_rows`` against the whole-pair ``sm80_wide.forward`` at L 512 (blocks of 130 rows: a partial last block), and
    deterministic."""
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    base = _triattn(starting)
    kern = copy.deepcopy(base).set_backend("cuda")
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = base.make_inputs(length=512, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    if masked:
        inputs["mask"][:, -40:] = False
    else:
        inputs["mask"] = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert kern.unsupported(**inputs) is None
        whole = kern(**inputs)
        monkeypatch.setattr(sm80_wide, "ROWS_BLOCKED_BYTES", 0)
        monkeypatch.setattr(sm80_wide, "ROWS_BLOCK_BYTES", 130 * 512 * 1568 * 2)
        rows = kern(**inputs)
        again = kern(**inputs)
    assert torch.equal(rows, again)
    torch.testing.assert_close(rows.float(), whole.float(), rtol=0, atol=2e-2)
    assert _err(rows, whole.float()) < 4e-3


@pytest.mark.skipif(_card() != (8, 0), reason="A100 (the sm_80 wide path pads L)")
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("length", [300, 200])
def test_triangle_attention_sm80_padded_length(starting: bool, length: int) -> None:
    """L not a multiple of 128 (OpenDDE n = 12: 2976): the wide inference pads L, against the float32 reference at the measurer's bound."""
    base = _triattn(starting)
    kern = copy.deepcopy(base).set_backend("cuda")
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = base.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    inputs["mask"][:, -11:] = False
    with torch.no_grad():
        gold = copy.deepcopy(base).float()(inputs["pair"].float(), inputs["mask"]).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            err_pt = _err(copy.deepcopy(base)(**inputs), gold)
            assert kern.unsupported(**inputs) is None
            out = kern(**inputs)
    assert _err(out, gold) <= max(err_pt, FIXED)


def _transition_rows(rows: int, step: int) -> list[tuple[int, int]]:
    """Row ranges checked at a refiner size: the first rows, both sides of every block boundary of a split call, the last rows."""
    span = 32768
    out = [(0, span)]
    out += [(r - span, r + span) for r in range(step, rows, step)]
    out.append((rows - span, rows))
    return out


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("S", [2976, 3968])
def test_transition_refiner_sizes_rows(backend: str, S: int) -> None:
    """Transition d_hidden 384 n 2 (master) at the refiner's pair rows (S^2: past one sm_80 launch's rows, the call is split): the module is
    row-wise, so the float32 reference on the first / last rows and on both sides of each block boundary checks the whole call, at the measurer's
    GREEN bound there. On A100 the cuda rows also equal (bitwise) one unsplit call on the same rows."""
    from miniworld_engine.kernels.transition.cuda import fused_fwd_sm80, fused_wide_sm80
    from miniworld_engine.modules import Transition

    base = Transition(384, n=2).cuda().float().eval()
    C.randomize(base, torch.Generator(device="cuda").manual_seed(0))
    base.set_master_weights()
    kern = copy.deepcopy(base).set_backend(backend)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = base.make_inputs(length=S, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)["x"]
    rows = S * S
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert kern.unsupported(x) is None
            big = kern(x).reshape(rows, 384)
        flat = x.reshape(rows, 384)
        del x
        sm80 = _card() == (8, 0)
        limit = fused_fwd_sm80.max_rows(flat.device) if sm80 else rows
        nblk = -(-rows // limit)
        per = -(-rows // nblk)
        step = min(-(-per // 256) * 256, limit)  # transition_sm80's balanced blocks
        ref32 = copy.deepcopy(base).float()
        ref_pt = copy.deepcopy(base)
        for lo, hi in _transition_rows(rows, step):
            sub = flat[lo:hi]
            gold = ref32(sub.float()).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                err_pt = _err(ref_pt(sub), gold)
            err = _err(big[lo:hi], gold)
            tol = max(err_pt, FIXED)
            assert err <= tol, f"rows [{lo}, {hi}): {err:.3e} > tol {tol:.3e} (pytorch {err_pt:.3e})"
            if sm80 and backend == "cuda":
                p = kern
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    whole = fused_wide_sm80.transition_wide_sm80(
                        sub.contiguous(), p.ln_in.weight, p.ln_in.bias, p.expand_a.weight, p.expand_b.weight, p.squeeze.weight, p.ln_in.eps
                    )
                assert torch.equal(big[lo:hi], whole), f"rows [{lo}, {hi}): the split call differs from one unsplit call"
