"""B200 atom-block weight-gradient and column-sum kernels (``sm100_atom.atom_dw``, ``colsum32``): the grouped
split-K weight-gradient kernel gives every problem's A^T B (bf16 operands, or fp32 operands on tf32 tensor cores) to the destination of each
128-row block of A -- strided column views, several blocks per problem, row counts that are not a multiple of the stage -- and the column
sums match fp64."""

import pytest
import torch

from miniworld_engine.integrations import local_dit
from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]


def _problems(R, dt, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, device="cuda", generator=g).to(dt)

    big768, big512, x128, x128b, x256 = rnd(R, 768), rnd(R, 512), rnd(R, 128), rnd(R, 128), rnd(R, 256)
    return [
        (x128, x256, 1),                          # [128, 256]
        (big512, x128, 4),                        # [512, 128] -> 4 destination blocks
        (big768[:, 0:256], x128b, 2),             # strided column views
        (big768[:, 512:768], x128, 2),
        (big512[:, 384:512], x128, 1),
        (big512[:, 128:384], x128b, 2),
    ]


@pytest.mark.parametrize("R", [128, 20480, 61440 + 128, 196608])
@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "tf32"])
def test_atom_dw_matches_fp64(R, f32):
    assert local_dit._kernels32_ready(torch.cuda.current_device()) if f32 else local_dit._kernels_ready(torch.cuda.current_device())
    dt = torch.float32 if f32 else torch.bfloat16
    problems, refs = [], []
    for A, B, mb in _problems(R, dt):
        dests = [torch.empty(128, B.shape[1], device="cuda", dtype=torch.float32 if f32 else torch.bfloat16) for _ in range(mb)]
        problems.append((A, B, dests))
        refs.append(A.double().t() @ B.double())
    rows.atom_dw(torch.cuda.current_device(), f32)(problems, R)
    torch.cuda.synchronize()
    tol = 3e-3 if f32 else 1e-2                   # tf32 operands (2^-11); bf16 destinations (2^-9)
    for (A, B, dests), ref in zip(problems, refs):
        got = torch.cat([d.double() for d in dests], 0)
        assert got.shape == ref.shape
        assert float((got - ref).norm() / ref.norm()) < tol, (tuple(A.shape), tuple(B.shape))


@pytest.mark.parametrize("R", [8, 1000, 20480, 196608])
def test_atom_dw_is_deterministic(R):
    R = ((R + 127) // 128) * 128
    assert local_dit._kernels32_ready(torch.cuda.current_device())
    problems = [(A, B, [torch.empty(128, B.shape[1], device="cuda") for _ in range(mb)]) for A, B, mb in _problems(R, torch.float32)]
    dw = rows.atom_dw(torch.cuda.current_device(), True)
    dw(problems, R)
    first = [torch.cat(d, 0).clone() for _, _, d in problems]
    dw(problems, R)
    for (_, _, d), f in zip(problems, first):
        assert torch.equal(torch.cat(d, 0), f)


@pytest.mark.parametrize("M", [1, 7, 4096, 196608 + 5])
def test_column_sums(M):
    assert local_dit._kernels32_ready(torch.cuda.current_device())
    g = torch.Generator(device="cuda").manual_seed(M)
    wide = torch.randn(M, 384, device="cuda", generator=g)
    x32 = wide[:, 128:256]                         # a strided view: unit column stride, row pitch 384
    ref = x32.double().sum(0)
    assert torch.allclose(rows.colsum32(x32).double(), ref, rtol=1e-4, atol=1e-3 * max(1.0, M ** 0.5))
