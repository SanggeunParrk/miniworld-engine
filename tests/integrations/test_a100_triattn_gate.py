"""The geometry gate of the A100 TriangleAttention path (``kernels/triangle_attention/cuda/sm80_wide.py``): which (d_pair, d_hidden, n_head) it serves and with which engine.
Pure Python, runs without a GPU."""

import torch

from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide as w


def test_geometry_gate(monkeypatch):
    monkeypatch.delenv("MINIWORLD_TRIATTN_SM80_ROWS", raising=False)
    assert w.cfg_for(64, 64, 4) == w.Cfg(64, 4, 16, True)                    # a 16-channel head is padded to 32 by the packed weights
    assert w.cfg_for(64, 128, 4).hd == 32
    assert w.cfg_for(64, 128, 4).fused
    assert w.cfg_for(64, 64, 2).fused
    assert w.cfg_for(128, 128, 4).fused
    assert w.cfg_for(256, 256, 8) == w.Cfg(256, 8, 32, False)                # the row kernels + cuBLAS
    assert w.cfg_for(384, 384, 12) == w.Cfg(384, 12, 32, False)
    assert w.cfg_for(512, 512, 16) == w.Cfg(512, 16, 32, False)
    assert w.cfg_for(100, 100, 4) is None                                     # d_pair not a multiple of 64
    assert w.cfg_for(64, 96, 4) is None                                       # head dim 24
    assert w.cfg_for(64, 64, 8) is None                                       # head dim 8
    assert w.cfg_for(64, 64, 17) is None                                      # too many heads
    assert w.cfg_for(64, 66, 4) is None                                       # hidden not divisible by the heads
    assert w.cfg_for(576, 576, 18) is None                                    # d_pair above 512
    c = w.cfg_for(256, 256, 8)
    assert (c.dhp, c.hpad, c.n) == (256, 8, 1032)
    assert w.cfg_for(384, 384, 12).hpad == 16
    assert w.cfg_for(64, 64, 2).hpad == 8
    assert w.cfg_for(64, 64, 4).native                                        # 16-channel heads at d_pair 64: kept as they are (no padding)
    assert w.cfg_for(64, 64, 4).n == 256
    assert w.cfg_for(64, 64, 4).lhd == 16
    assert not w.cfg_for(128, 64, 4).native                                   # the other 16-channel layouts are padded to 32
    assert w.cfg_for(128, 64, 4).dhp == 128
    assert w.cfg_for(64, 64, 4).dhp == 64
    assert w.cfg_for(64, 64, 4).hd == 16


def test_the_row_kernels_engine_can_be_forced_at_the_narrow_widths(monkeypatch):
    monkeypatch.setenv("MINIWORLD_TRIATTN_SM80_ROWS", "1")
    assert not w.cfg_for(64, 128, 4).fused
    assert not w.cfg_for(128, 128, 4).fused
    assert w.cfg_for(64, 128, 4).n == 4 * 128 + 8


def test_the_gate_rejects_cpu_tensors_and_the_switch(monkeypatch):
    cfg = w.cfg_for(64, 128, 4)
    pair = torch.zeros(1, 128, 128, 64, dtype=torch.bfloat16)
    ws = [torch.zeros(128, 64, dtype=torch.bfloat16)] * 4 + [torch.zeros(4, 64, dtype=torch.bfloat16)]
    wo, ln = torch.zeros(64, 128, dtype=torch.bfloat16), torch.zeros(64)
    assert not w.supports(pair, cfg, ws, wo, ln, ln)                          # not CUDA tensors
    assert not w.supports(pair, None, ws, wo, ln, ln)
    monkeypatch.setenv("MINIWORLD_TRIATTN_SM80", "0")
    assert not w.enabled()


# ------------------------------------------------------------------------------------------------------- the projected-attention leaf (sm80_projected.py)
def test_the_projected_attention_gate_rejects_cpu_tensors_and_the_switch(monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected as sp

    q = torch.zeros(128, 1, 4, 128, 32, dtype=torch.bfloat16)
    bias = torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16)
    assert not sp.serves(q, q, q, bias, None)                                  # not CUDA tensors: the op keeps its Triton path
    monkeypatch.setenv("MINIWORLD_TRIATTN_SM80", "0")
    assert not sp.enabled()
    monkeypatch.delenv("MINIWORLD_TRIATTN_SM80")
    assert sp.enabled()


def test_the_projected_attention_operands_are_read_where_they_are_when_the_core_can():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected as sp

    head_major = torch.zeros(128, 1, 4, 128, 32, dtype=torch.bfloat16)
    assert sp._readable(head_major) is head_major
    assert sp._lays(head_major) == (32, 4 * 128 * 32, 4 * 128 * 32, 128 * 32)
    token_major = torch.zeros(128, 1, 128, 4, 32, dtype=torch.bfloat16).permute(0, 1, 3, 2, 4)   # head-major shape, token-major memory
    assert sp._readable(token_major) is token_major
    assert sp._lays(token_major) == (4 * 32, 128 * 4 * 32, 128 * 4 * 32, 32)
    wide = torch.zeros(128, 1, 4, 128, 40, dtype=torch.bfloat16)
    sliced = wide[..., :32]
    assert sp._readable(sliced) is sliced                                      # row strides of 40 elements are granule multiples: read in place
    assert sp._aligned(sliced) is sliced
    shifted = wide[..., 4:36]
    assert sp._readable(shifted) is shifted                                    # the strides are fine ...
    assert sp._aligned(shifted).is_contiguous()                                # ... but an offset of 4 elements is not 16-byte aligned: the ops copy it
    assert sp._readable(head_major.transpose(3, 4)).is_contiguous()            # the channel stride is not 1: copied


def test_the_key_mask_becomes_the_cores_per_row_bytes():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected as sp

    like = torch.zeros(1, dtype=torch.bfloat16)
    assert sp._mask_rows(None, like).numel() == 0
    mask = torch.rand(8, 2, 8) > 0.5                                           # [A, B, L]
    got = sp._mask_rows(mask, like)
    assert got.dtype is torch.uint8
    assert got.shape == (2, 8, 8)                                              # [B, A, L]
    assert torch.equal(got.bool(), mask.transpose(0, 1))


def test_the_bias_gradient_partials_of_a_training_step_are_cubic_in_the_length():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_core2 as c2

    assert c2.dbp_bytes(384, 4, 1) == (384 // c2.DQ_ROWS) * 4 * 384 * 384 * 2
    assert c2.dbp_bytes(768, 12, 1) < 3 << 30                                  # the registry's largest shape: 2.7 GB
    assert c2.dbp_bytes(1536, 12, 1) > 16 << 30                                # beyond the leaf's limit the Triton path serves the gradient


def test_the_partials_guard_of_the_training_step():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_core2 as c2

    assert c2.dbp_fits(768, 12, 1)
    assert not c2.dbp_fits(1536, 12, 1)
