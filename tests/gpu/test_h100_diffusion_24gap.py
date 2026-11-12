"""H100 regressions vs engine 2.4.4 on the diffusion side (2026-11-12), the served pieces:

* ``pair_bias_all`` on the ``cuda`` backend is the fused hoist on H100 (pair 128 / 256, bf16 / fp32): each block's bias a contiguous
  head-major [1, H, L, L] slice (the bias-only block copies nothing per call), the PyTorch fold's values to rounding, the training path's
  gradients reaching the pair and every block's ``to_bias`` / ``ln_pair`` through it;
* ``PoolAtoms``' wide kernel at SUB 4 / 8 (by the CTA count) gives the same bits.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]

H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
BF = torch.bfloat16


def _blocks(d_pair: int, n: int = 3):
    from miniworld_engine.measure import config as C
    from miniworld_engine.modules import TokenDiT

    out = []
    for i in range(n):
        b = TokenDiT("miniworld_bias_only", 768, 384, d_pair, 16, 2).cuda().float()
        C.randomize(b, torch.Generator(device="cuda").manual_seed(i))
        out.append(b.set_backend("cuda"))
    return out


@pytest.mark.skipif(not H100, reason="H100 (sm_90)")
@pytest.mark.parametrize(("d_pair", "dtype"), [(128, BF), (256, BF), (128, torch.float32)])
def test_hoist_is_fused_head_major(d_pair: int, dtype: torch.dtype) -> None:
    from miniworld_engine.kernels.integrations import bias_only_dit_hoist
    from miniworld_engine.modules.token_dit import pair_bias_all

    blocks = [b.eval().requires_grad_(False) for b in _blocks(d_pair)]
    L = 256
    pair = torch.randn(1, L, L, d_pair, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9)).to(dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=BF, enabled=dtype is BF):
        assert bias_only_dit_hoist.serves("miniworld", [b.attention for b in blocks], pair)
        fused = pair_bias_all(blocks, pair)
        for b in blocks:
            b.set_backend("pytorch")
        fold = pair_bias_all(blocks, pair)
    for a, c in zip(fused, fold, strict=True):
        assert tuple(a.shape) == (1, 16, L, L) and a.is_contiguous()
        rel = float((a.float() - c.float()).norm() / c.float().norm())
        assert rel < (1e-2 if dtype is BF else 2e-3), rel  # bf16 rounding / TF32 GEMM vs the fold


@pytest.mark.skipif(not H100, reason="H100 (sm_90)")
def test_hoist_training_gradients() -> None:
    """bf16 training through the fused hoist and the H100 training block on the bias vs the PyTorch block on the pair."""
    from miniworld_engine.modules.token_dit import pair_bias_all

    blocks = [b.train() for b in _blocks(128)]
    L = 128
    g = torch.Generator(device="cuda").manual_seed(5)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g).to(BF)
    single = torch.randn(4, 1, L, 768, device="cuda", generator=g).to(BF)
    cond = torch.randn(4, 1, L, 384, device="cuda", generator=g).to(BF)
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")

    def run(hoisted: bool):
        for b in blocks:
            b.zero_grad(set_to_none=True)
            b.set_backend("cuda" if hoisted else "pytorch")
        p = pair.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=BF):
            x = single
            if hoisted:
                for b, bias in zip(blocks, pair_bias_all(blocks, p), strict=True):
                    x = b(x, cond, None, mask, bias=bias)
            else:
                for b in blocks:
                    x = b(x, cond, p, mask)
        x.float().square().mean().backward()
        return [p.grad] + [t.grad for b in blocks for t in (b.attention.to_bias.weight, b.attention.ln_pair.weight)]

    got, ref = run(True), run(False)
    for a, c in zip(got, ref, strict=True):
        assert a is not None and c is not None
        rel = float((a.float() - c.float()).norm() / c.float().norm())
        assert rel < 3e-2, rel


@pytest.mark.skipif(not H100, reason="H100 (sm_90)")
@pytest.mark.parametrize(("samples", "atoms"), [(1, 1024), (48, 4096)])
def test_pool_atoms_sub_choice_same_bits(samples: int, atoms: int) -> None:
    """Below 1024 CTAs the wide kernel runs SUB 8, from 1024 on SUB 4; both sum the atoms in the same order: a call on each side of the
    threshold equals the PyTorch reference to bf16 rounding, and the two SUBs agree bitwise on the rows they share."""
    from miniworld_engine.modules import sampler_ops as so

    g = torch.Generator(device="cuda").manual_seed(3)
    T, rows = atoms // 8, atoms + 64
    x = torch.randn(samples, rows, 128, device="cuda", generator=g).to(BF)
    idx = torch.randint(0, rows, (T, 24), device="cuda", generator=g)
    am = torch.rand(T, 24, device="cuda", generator=g) > 0.3
    gm = am | (torch.rand(T, 24, device="cuda", generator=g) > 0.9)
    m = so.PoolAtoms().cuda().set_backend("cuda")
    y = m(x, idx, gm, am)
    ref = m.set_backend("pytorch")(x, idx, gm, am)
    torch.testing.assert_close(y.float(), ref.float(), rtol=1e-2, atol=1e-2)
    if samples * T >= 1024:  # the first sample alone runs SUB 8 (T < 1024 CTAs): the same bits as its row of the SUB 4 call
        m.set_backend("cuda")
        assert torch.equal(m(x[:1].contiguous(), idx, gm, am)[0], y[0])
