"""The fused H100 token DiT inference over B trunk_seeds x A diffusion_seeds in ONE call: single [A, B, L, d], pair [B, L, L, dp] (each
trunk_seed its own pair), the conditioning shared by the A diffusion_seeds, one key mask expanded over B (grouped-bias attention, ``NBG``).

The grouped call must (a) actually take the fused route (a silent fallback to the composed path is a failure), (b) agree with B
separate B = 1 calls of the same fused route, and (c) hold the repo's yardstick against an fp64 PyTorch run: error no worse than 1.5x
the error of the bf16 PyTorch module."""
import pytest
import torch
from portable_helpers import randomize_half_std as _randomize
from settings_helpers import policy_portable_off as policy  # noqa: F401

from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize(("A", "B", "L", "masked", "qk"), [
    (5, 5, 256, False, False),
    (5, 5, 512, True, False),
    (3, 4, 256, True, True),
    (1, 3, 384, False, False),      # one diffusion_seed per trunk_seed
    (4, 2, 512, True, False),
])
def test_trunk_seed_group_matches_separate_calls_and_the_fp64_yardstick(A, B, L, masked, qk):
    from miniworld_engine.kernels.conditioned_transition import cuda as ct_cuda

    torch.manual_seed(11)
    eng = _randomize(DiTBlock(use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()).eval()
    pt = DiTBlock(use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda().eval()
    pt.load_state_dict(eng.state_dict())
    ref = DiTBlock(use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda().double().eval()
    ref.load_state_dict({k: v.double() for k, v in eng.state_dict().items()})

    bf = torch.bfloat16
    single = torch.randn(A, B, L, 768, device="cuda").to(bf)
    cond = torch.randn(1, B, L, 384, device="cuda").to(bf).expand(A, -1, -1, -1)          # shared by the A diffusion_seeds
    pair = torch.randn(B, L, L, 128, device="cuda").to(bf)
    mask = None
    if masked:
        row = torch.rand(1, L, device="cuda") > 0.15
        mask = row.expand(B, L)                                                         # one key mask, stride 0 over B

    taken = []
    orig = ct_cuda.token_inference
    ct_cuda.token_inference = lambda s, *a, **k: (taken.append(tuple(s.shape)), orig(s, *a, **k))[1]
    try:
        with torch.no_grad():
            grouped = eng(single, cond, pair, mask)
            again = eng(single, cond, pair, mask)
            separate = torch.cat([
                eng(single[:, b:b + 1], cond[:, b:b + 1], pair[b:b + 1], None if mask is None else mask[b:b + 1])
                for b in range(B)], dim=1)
    finally:
        ct_cuda.token_inference = orig
    assert taken[0] == (A, B, L, 768), f"the grouped call did not reach the fused core as one call: {taken}"
    assert all(t[1] == 1 for t in taken[2:]), taken
    assert torch.equal(grouped, again), "the grouped call is not repeatable bit for bit"
    assert _rel(grouped, separate) < 1e-2, _rel(grouped, separate)

    with torch.no_grad():
        r = torch.cat([ref(single[:, b:b + 1].double(), cond[:, b:b + 1].double(), pair[b:b + 1].double(),
                           None if mask is None else mask[b:b + 1]) for b in range(B)], dim=1)
        with torch.autocast("cuda", dtype=bf):
            p = torch.cat([pt(single[:, b:b + 1], cond[:, b:b + 1], pair[b:b + 1],
                              None if mask is None else mask[b:b + 1]) for b in range(B)], dim=1)
    assert _rel(grouped, r) <= 1.5 * _rel(p, r), (_rel(grouped, r), _rel(p, r))
