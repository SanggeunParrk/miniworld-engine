"""H100: the anthropic TriangleAttention backend runs the kits' in-model attention core (the release's ``triattn_native`` / ``cuda_sm90a``
rows, word fast) and the block with its residual folded, not the k2b Triton step-aside. Needs the release with a prebuilt of
triattn_native for this stack (third_party/anthropic/build_local.sh); skipped where the release or the prebuilt is absent."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 (sm_90) only"
)


def _have_prebuilt() -> bool:
    from miniworld_engine.modules import _anthropic

    if _anthropic.missing("opt_core.kernels.triattn") is not None:
        return False
    tn = _anthropic.load("opt_core.kernels.triattn.triattn_native")
    try:
        return tn.stack_key() in tn.stacks_built()
    except Exception:  # noqa: BLE001 -- any import failure: the payload is not servable here
        return False


@pytest.mark.parametrize("length", [384, 768])
def test_triattn_anthropic_serves_the_kit_core(length):
    if not _have_prebuilt():
        pytest.skip("the release's triattn_native has no prebuilt for this stack")
    from miniworld_engine.modules import _anthropic
    from miniworld_engine.modules.triangle import TriangleAttention

    torch.manual_seed(0)
    m = TriangleAttention(d_pair=128, n_head=4, starting=True).cuda().eval()
    m.set_backend("anthropic")
    pair = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    PF = _anthropic.load("opt_core.attn.pair_fused")
    PF.SERVED_CORES.clear()
    with torch.no_grad():
        assert m.unsupported(pair) is None
        out = m._forward_anthropic(pair)
        ref = m._forward_pytorch(pair)
    served = ",".join(PF.SERVED_CORES)
    want = "cuda_sm90a" if length <= 400 else "triattn_native"
    assert f"tier:fast={want}" in served, served
    assert out.data_ptr() != pair.data_ptr()  # the folded block updates a copy, never the caller's pair
    assert (out.float() - ref.float()).abs().mean().item() < 2e-2  # bf16 roundings only
