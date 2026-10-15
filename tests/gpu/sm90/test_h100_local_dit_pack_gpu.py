"""The one-launch weight pack of the H100 atom block (sm90_atom/pack90.cu) equals the torch expression of it bit for bit."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"),
]

SHAPES = {
    "g1": (128,), "sc1_w": (128, 128), "sc1_b": (128,), "bi1_w": (128, 128), "wq": (128, 128), "bq": (128,), "wk": (128, 128),
    "wv": (128, 128), "ln_pair": (16,), "wb": (4, 16), "wg": (128, 128), "wo": (128, 128), "so_w": (128, 128), "so_b": (128,),
    "g2": (128,), "sc2_w": (128, 128), "sc2_b": (128,), "bi2_w": (128, 128), "wa": (256, 128), "wb2": (256, 128), "wd": (128, 256),
    "st_w": (128, 128), "st_b": (128,), "gkv": (128,), "kv_sc_w": (128, 128), "kv_sc_b": (128,), "kv_bi_w": (128, 128),
}


@pytest.mark.parametrize("tf32", [False, True], ids=["bf16", "tf32"])
@pytest.mark.parametrize("pdt", [torch.float32, torch.bfloat16], ids=["fp32params", "bf16params"])
def test_kernel_pack_matches_torch_pack(tf32, pdt):
    from miniworld_engine.kernels.augmented_attention.cuda import sm90_atom as k

    torch.manual_seed(0)
    ws = [torch.randn(*SHAPES[n], device="cuda").to(pdt) for n in k._I]
    ref_f, ref_b = k._pack_torch(ws, tf32), k._pack_bwd_torch(ws, tf32)
    fwd, bwd = k._pack_all(ws, tf32)
    for name, ref in {**ref_f, **ref_b}.items():
        if name == "dtype":
            continue
        got = fwd.get(name, bwd.get(name))
        assert got.shape == ref.shape and got.dtype == ref.dtype, (name, got.shape, ref.shape, got.dtype, ref.dtype)
        assert torch.equal(got, ref), name
