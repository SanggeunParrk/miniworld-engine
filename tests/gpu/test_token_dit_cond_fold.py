"""The H100 bf16 token DiT step's folded conditioning and its SwiGLU tiles (``kernels/conditioned_transition/cuda``).

- The conditioning folded into ONE GEMM of the raw c (the LayerNorm of c applied where the AdaLN tables are read) matches the fp32
  reference block as closely as the LayerNorm-then-two-GEMMs step did, for a conditioning shared by the samples (stride 0) and one per
  sample, at d_cond 384 and 768, sequential and parallel blocks, the up-gate transition (Boltz-2).
- The conditioning GEMM runs on a side stream from the previous block's start (``_cond_gemm``): a captured 12-block stack replayed after the
  conditioning was refilled in place gives the eager output of the new conditioning, bitwise -- also under ``static_inputs()`` (FoldForge's
  denoiser graph makes the step's conditioning inside the graph: nothing made from it may outlive a replay).
- The bf16 SwiGLU GEMM's 128 x 128 tiles (``ht`` 64) and its up-gate form against an fp32 reference.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]

S = 5


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _block(model: str, dc: int, dp: int):
    from miniworld_engine.modules import TokenDiT

    torch.manual_seed(0)
    b = TokenDiT(model, 768, dc, dp, 16, 2)
    for p in b.parameters():
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.3, std=0.2)
    return b.float().cuda().eval().requires_grad_(False)


def _inputs(L: int, dc: int, dp: int, shared: bool, seed: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, 1, L, 768, device="cuda", generator=g).bfloat16()
    c0 = (torch.randn(1 if shared else S, 1, L, dc, device="cuda", generator=g) * 2 + 0.7).bfloat16()
    cond = c0.expand(S, 1, L, dc) if shared else c0
    pair = torch.randn(1, L, L, dp, device="cuda", generator=g).bfloat16()
    mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
    mask[0, -7:] = False
    return x, cond, pair, mask


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize(
    ("model", "dc", "dp", "bound"),
    [("esmfold2", 768, 256, 5e-3), ("boltz2", 768, 128, 6e-3), ("protenix", 384, 128, 4e-3), ("rosettafold3", 384, 128, 4e-3)],
)
@pytest.mark.parametrize("shared", [True, False])
def test_folded_conditioning_matches_reference(model, dc, dp, bound, shared):
    b = _block(model, dc, dp)
    x, cond, pair, mask = _inputs(128, dc, dp, shared)
    with torch.no_grad():
        b.set_backend("cuda")
        out = b(x, cond, pair, mask).float()
        b.set_backend("pytorch")
        ref = b(x.float(), cond.float(), pair.float(), mask).float()
    rel = float((out - ref).norm() / ref.norm())
    # measured 2026-11-12 (H100): esmfold2 3.6e-3, boltz2 4.1e-3, protenix / rosettafold3 3.1e-3 (the unfolded step: 3.9e-3 / 4.9e-3 /
    # 3.2e-3 / 3.2e-3)
    assert rel < bound, rel


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize(("model", "dc", "dp"), [("esmfold2", 768, 256), ("boltz2", 768, 128)])
def test_pipelined_conditioning_replay_sees_new_conditioning(model, dc, dp):
    from miniworld_engine.capture import static_inputs, static_weights

    blocks = [_block(model, dc, dp) for _ in range(4)]
    for b in blocks:
        b.set_backend("cuda")
    L = 128
    x, cond, pair, mask = _inputs(L, dc, dp, shared=True)
    c_buf = cond[:1].clone()  # the step's conditioning buffer (refilled in place between replays)
    x_buf = x.clone()

    def stack():
        h = x_buf
        cnd = c_buf.expand(S, 1, L, dc)
        for b in blocks:
            h = b(h, cnd, pair, mask)
        return h

    with torch.no_grad(), static_weights(), static_inputs():
        stack()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                stack()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = stack()
        for seed in (2, 3):
            _, c_new, _, _ = _inputs(L, dc, dp, shared=True, seed=seed)
            c_buf.copy_(c_new[:1])
            x_buf.copy_(x * (seed - 1))
            graph.replay()
            eager = stack()
            torch.cuda.synchronize()
            assert torch.equal(gout, eager), float((gout.float() - eager.float()).abs().max())


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize("M", [640, 1280, 1920])
@pytest.mark.parametrize("up", [False, True])
def test_swiglu_gemm_tiles(M, up):
    from miniworld_engine.kernels.conditioned_transition import cuda as C

    op = C._token_swiglu(torch.cuda.current_device())
    K, H = 768, 1536
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn((3 if up else 2) * H, K, device="cuda", generator=g) * 0.05).bfloat16()
    x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
    parts = [x.float() @ w[i * H : (i + 1) * H].float().t() for i in range(3 if up else 2)]
    ref = torch.nn.functional.silu(parts[0]) * parts[1] * (parts[2] if up else 1.0)
    if up:
        wp, ht = C._bf16_wab64(w.contiguous(), up=True), 64
    else:  # the token pack's per-128 layout, then the 64 re-tiling
        w128 = torch.stack([w[:H].view(H // 128, 128, K), w[H:].view(H // 128, 128, K)], 1).reshape(2 * H, K).contiguous()
        ht = C._swiglu_ht(M, H, op.nsm, bf16=True)
        wp = w128 if ht == 128 else C._bf16_wab64(w128)
    h = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    op.ext.gemm_swiglu(x, wp, h, op.nsm, ht, up, True)
    rel = float((h.float() - ref).norm() / ref.norm())
    assert rel < 3e-3, rel  # bf16 output rounding: 1.7e-3 measured
