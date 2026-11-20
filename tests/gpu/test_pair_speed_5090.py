"""RTX 5090 (sm_120a), s5090/pair-speed (and s5090/four: the RF3 D 128 split front): the D 256 triangle-attention front without constant columns (K = 256, the LayerNorm shift in the GEMM bias),
the persistent TokenPairInit forward and the Chai-1 outer-product products at S >= 1024 keep the old outputs (the shift's one bf16 rounding aside)."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TokenPairInit, TriangleAttentionAF3, TriangleAttentionChai1

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs an RTX 5090 (sm_120)"),
]


def _run(module, inputs, backend: str) -> torch.Tensor:
    m = copy.deepcopy(module).set_backend(backend).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m(**inputs).float()


@pytest.mark.parametrize("cls", [TriangleAttentionAF3, TriangleAttentionChai1])
@pytest.mark.parametrize("length", [128, 256])
def test_bare_front_matches_padded_front(cls, length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    module = cls(d_pair=256, n_head=8 if cls is TriangleAttentionAF3 else 4, starting=True, p_drop=0.0, master_weights=True).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    inputs = module.make_inputs(
        length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_BARE", "1")
        bare = _run(module, inputs, "cuda")
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_BARE", "0")
        padded = _run(module, inputs, "cuda")
    ref = _run(module, inputs, "pytorch")
    scale = ref.abs().max().item()
    # the bare form differs from the padded one by the shift's single bf16 rounding; both stay inside the accuracy standard against the reference
    assert (bare - padded).abs().max().item() <= 8e-3 * scale
    assert (bare - ref).abs().max().item() <= max((padded - ref).abs().max().item(), 8e-3 * scale)


@pytest.mark.parametrize("length", [128, 160, 170, 171, 256, 384])  # B L <= 170 SMs: one-row CTAs; beyond: the persistent grid
def test_token_pair_init_persistent_forward_matches_reference(length: int) -> None:
    module = TokenPairInit().cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    inputs = module.make_inputs(
        length=length, samples=1, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )
    with torch.no_grad():
        ours = copy.deepcopy(module).set_backend("cuda")(**inputs)
        ref = copy.deepcopy(module).set_backend("pytorch")(**inputs)
    assert (ours - ref).abs().max().item() <= 1e-5


@pytest.mark.parametrize("length", [128, 256])
def test_rf3_gate_bias_folded_into_the_front_shift(length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from miniworld_engine.modules import TriangleAttentionRF3

    module = TriangleAttentionRF3(d_pair=64, d_hidden=256, n_head=4, starting=True, p_drop=0.0, master_weights=True).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    inputs = module.make_inputs(
        length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_BARE", "auto")
        folded = _run(module, inputs, "cuda")
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_BARE", "0")
        epilogue = _run(module, inputs, "cuda")
    ref = _run(module, inputs, "pytorch")
    scale = ref.abs().max().item()
    assert (folded - epilogue).abs().max().item() <= 8e-3 * scale
    assert (folded - ref).abs().max().item() <= max((epilogue - ref).abs().max().item(), 8e-3 * scale)


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("length", [128, 384])
def test_rf3_d128_front_split_into_two_gemms(length: int, starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """s5090/four: inference at D 128 with four 32-channel heads runs q | k | v | g (N 512) and the bias heads (N 8) as two GEMMs on ``ln_rows``' padded rows
    (K = 128): within the bf16 rounding of the folded shift of the single K = 136 / N = 520 GEMM, and no further from the reference than it."""
    from miniworld_engine.modules import TriangleAttentionRF3

    module = TriangleAttentionRF3(d_pair=128, n_head=4, starting=starting, p_drop=0.0, master_weights=True).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    inputs = module.make_inputs(
        length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_SPLIT", "auto")
        split = _run(module, inputs, "cuda")
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_SPLIT", "0")
        single = _run(module, inputs, "cuda")
    ref = _run(module, inputs, "pytorch")
    scale = ref.abs().max().item()
    assert (split - single).abs().max().item() <= 8e-3 * scale
    assert (split - ref).abs().max().item() <= max((single - ref).abs().max().item(), 8e-3 * scale)
