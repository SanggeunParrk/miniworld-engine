"""3D RoPE angles are fp32 by contract (``build_3d_rope``'s cos / sin feed a rotation whose angle precision matters): they must not
change under an autocast region. An ``einsum`` outer product does -- autocast runs it as a bf16 matmul, so the UID angles (ids up to
thousands times a frequency) and the spatial ones came out in bf16 and the rotation was wrong."""

import pytest
import torch

from miniworld_engine.modules.swa_atom_attention.module import build_3d_rope

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_build_3d_rope_is_independent_of_autocast(device):
    torch.manual_seed(0)
    pos = torch.randn(2, 512, 3, device=device) * 30.0
    uid = torch.randint(0, 3000, (2, 512), device=device)
    cos0, sin0 = build_3d_rope(pos, uid, 32)
    with torch.autocast(device, dtype=torch.bfloat16):
        cos1, sin1 = build_3d_rope(pos, uid, 32)
    assert cos1.dtype == sin1.dtype == torch.float32
    assert torch.equal(cos1, cos0) and torch.equal(sin1, sin0)


@pytest.mark.parametrize("device", DEVICES)
def test_build_3d_rope_matches_the_fp32_outer_product(device):
    pos = torch.randn(1, 64, 3, device=device) * 10.0
    uid = torch.arange(64, device=device)[None]
    cos, sin = build_3d_rope(pos, uid, 32)
    spatial = 1.0 / (20.0 ** (torch.arange(2, dtype=torch.float32, device=device) / 2))
    uidf = 1.0 / (10000.0 ** (torch.arange(10, dtype=torch.float32, device=device) / 10))
    want = torch.cat([(pos[..., None] * spatial).reshape(1, 64, 6), uid.float()[..., None] * uidf, torch.zeros(1, 64, 0, device=device)], -1)
    assert torch.allclose(cos[..., :16], want.cos(), atol=1e-6) and torch.allclose(sin[..., :16], want.sin(), atol=1e-6)
