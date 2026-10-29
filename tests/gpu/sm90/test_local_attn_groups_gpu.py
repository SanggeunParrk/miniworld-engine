"""The H100 atom window attention forward (``sm100_atom_local.attn_fwd_mma``, kernel ``h_attn_fwd``) over G distinct pairs in ONE launch:
the A samples laid out [A / G][G] (sample j * G + group reads the group's bias, one trunk_seed per group) must equal G separate one-bias
launches, output and LSE, bit for bit; G = 1 is the unchanged one-bias kernel."""
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


@pytest.mark.parametrize(("n", "per_group", "groups"), [
    (1024, 5, 1),
    (1024, 5, 5),
    (1024, 5, 3),
    (1088, 7, 4),        # N a multiple of 64 only; samples per group not a multiple of the chunk
    (512, 1, 5),         # one diffusion_seed per trunk_seed
    (1024, 32, 2),       # several chunks per group
])
def test_grouped_bias_launch_equals_separate_launches(n, per_group, groups):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    torch.manual_seed(5)
    nwin = (n + 31) // 32
    s = per_group * groups
    dev = "cuda"
    q, k, v = (torch.randn(s, n, 128, device=dev).to(torch.bfloat16) for _ in range(3))
    gamma = torch.randn(16, device=dev).abs() + 0.5
    wb = torch.randn(4, 16, device=dev) * 0.3
    kmask = torch.rand(n, device=dev) > 0.1
    biases = [
        local.pair_bias_mma(torch.randn(nwin, 32, 128, 16, device=dev).to(torch.bfloat16), gamma, wb, kmask, n)
        for _ in range(groups)
    ]
    stacked = torch.stack(biases, 0).contiguous()                                    # [G, 4, nwin, 32, 128]

    o, lse = local.attn_fwd_mma(q, k, v, stacked if groups > 1 else biases[0], groups=groups)
    torch.cuda.synchronize()
    for g in range(groups):
        sel = slice(g, None, groups)                                                # samples j * G + g
        o_ref, lse_ref = local.attn_fwd_mma(q[sel].contiguous(), k[sel].contiguous(), v[sel].contiguous(), biases[g])
        assert torch.equal(o[sel], o_ref), f"output differs for group {g}"
        assert torch.equal(lse[sel], lse_ref), f"LSE differs for group {g}"
