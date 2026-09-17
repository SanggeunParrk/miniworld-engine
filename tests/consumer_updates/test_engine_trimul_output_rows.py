"""Independent FP32 checks for projection-aware output backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="H100 required")


@pytest.mark.parametrize("case", ["normal", "shifted", "zero", "zero_gamma"])
@pytest.mark.parametrize("compiled", [False, True])
def test_output_rows_fp32(case, compiled, record_property):
    from miniworld_engine.kernels.trimul_inproj.cute import output_training as out

    torch.manual_seed(980)
    torch.backends.cuda.matmul.allow_tf32 = False
    m, k, n = 264, 256, 128
    x = torch.randn(k, m, device="cuda").t()
    if case == "shifted":
        x = x * 0.03 + 10
    if case == "zero":
        x.zero_()
    x = x.bfloat16()
    g = (torch.randn(k, device="cuda") * 0.2 + 1).bfloat16()
    b = (torch.randn(k, device="cuda") * 0.3).bfloat16()
    if case == "zero_gamma":
        g[::3] = 0
    w = (torch.randn(n, k, device="cuda") * 0.06).bfloat16()
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    def evaluate(x, g, b, w, dy):
        y, xhat, mean, rstd = out.forward(x, g, b, w, 1e-5)
        return (y,) + out.backward(dy, y, xhat, rstd, g, b, w)

    fn = (
        torch.compile(
            evaluate,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        if compiled
        else evaluate
    )
    result = fn(x, g, b, w, dy)
    ref = [t.float().detach().requires_grad_() for t in (x, g, b, w)]
    yr = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(ref[0], (k,), ref[1], ref[2], 1e-5), ref[3]
    )
    grads = torch.autograd.grad(yr, ref, dy.float())
    for name, a, z in zip(("y", "dx", "dg", "db", "dw"), result, (yr,) + grads):
        assert torch.isfinite(a).all(), name
        if z.norm() == 0:
            assert torch.count_nonzero(a) == 0, name
        else:
            err = (a.float() - z).norm() / z.norm()
            record_property(name + "_relative_l2", float(err))
            assert err < 0.015, (case, name, float(err))
    assert result[1].stride() == x.stride()


def test_gamma_storage_dtype_shares_native_cache_key(monkeypatch):
    from miniworld_engine.autotune import native
    from miniworld_engine.kernels.layernorm_linear.cute.dgrad_ln_rows import (
        dgrad_ln_rows,
    )

    seen = []

    def select(op, *, bucket, candidates, **kwargs):
        seen.append(bucket)
        return candidates[0]

    monkeypatch.setattr(native, "select_config", select)
    dy = torch.randn(264, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    xhat = torch.randn(264, 256, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    stats = [torch.randn(264, device="cuda") for _ in range(3)]
    a = dgrad_ln_rows(dy, w, xhat, gamma, *stats)
    b = dgrad_ln_rows(dy, w, xhat, gamma.float(), *stats)
    assert len(seen) == 2 and seen[0] == seen[1]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# Engine CI selects GPU checks explicitly.
pytestmark = [pytest.mark.gpu, *([pytestmark] if "pytestmark" in globals() else [])]
