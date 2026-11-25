"""The H100 ``parallel_residual_sm90`` build reads with ``ld.global.nc`` (``PR_HINT=2``, kernels/parallel_residual/cuda/__init__.py ``_FLAGS``)
where the previous build used streaming hints (``PR_HINT=0``). A load hint moves no bit: every output of the new build is bitwise the
streaming-hint build's (forward 1 .. 5 branches, backward with and without leaf copies; bf16 / fp16 / fp32; whole-vector, tail and unaligned sizes)."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="an sm_90 device"),
]


@pytest.fixture(scope="module")
def exts():
    from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    from foldjet_engine.kernels.parallel_residual import cuda as prc

    ensure_cuda_home()
    old = load_extension(
        name="parallel_residual_sm90_hint0_ref",
        sources=[str(prc._dir / "parallel_residual.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a"), "-DPR_HINT=0"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
    return prc.ext("sm90"), old


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("n", [1, 7, 4096 + 5, 3 * 1024 * 1024 + 3, 1 << 24])
def test_forward_bitwise(exts, dtype, n):
    new, old = exts
    g = torch.Generator(device="cuda").manual_seed(n)
    for nb in range(1, 6):
        x = torch.randn(n, device="cuda", dtype=dtype, generator=g)
        ys = [x + 0.1 * torch.randn(n, device="cuda", dtype=dtype, generator=g) for _ in range(nb)]
        assert torch.equal(new.forward(x, ys), old.forward(x, ys)), (dtype, n, nb)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("copies", [0, 1, 3])
def test_backward_bitwise(exts, dtype, copies):
    new, old = exts
    g = torch.randn(1 << 20, device="cuda", dtype=dtype)
    for a, b in zip(new.backward(g, 3, copies), old.backward(g, 3, copies), strict=True):
        assert torch.equal(a, b)


def test_h100_flag_is_the_read_only_load():
    from foldjet_engine.kernels.parallel_residual import cuda as prc

    assert prc._FLAGS["sm90"] == ("-DPR_HINT=2",)
    assert prc._FLAGS["sm80"] == ("-DPR_HINT=1",)
    assert "sm120" not in prc._FLAGS
