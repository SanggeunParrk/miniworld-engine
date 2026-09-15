"""Check real Hopper autograd orchestration using CPU math for GPU primitives.

This verifies saved tensors, mask placement, dropout, residual and gradient slots;
it does not claim to execute or validate the GPU instruction bodies.
"""
import pytest
import torch
import torch.nn.functional as F


def front(x, wl, wlg, wr, wrg, _wg, **kwargs):
    flat = x.reshape(-1, x.shape[-1])
    lp, lg, rp, rg = [flat @ w for w in (wl, wlg, wr, wrg)]
    def planes(t):
        return t.reshape(*x.shape[:-1], -1).permute(0, 3, 1, 2)
    return planes(lp * lg.sigmoid()), planes(rp * rg.sigmoid()), torch.cat((lp, lg, rp, rg), -1)


def front_backward(dl, dr, pre, x, wl, wlg, wr, wrg, *, pair_mask):
    width = wl.shape[-1]
    lp, lg, rp, rg = pre.split(width, -1)
    dl, dr = [v.permute(0, 2, 3, 1).reshape(-1, width) for v in (dl, dr)]
    if pair_mask is not None:
        dl, dr = dl * pair_mask.reshape(-1, 1), dr * pair_mask.reshape(-1, 1)
    sl, sr = lg.sigmoid(), rg.sigmoid()
    grads = (dl * sl, dl * lp * sl * (1 - sl), dr * sr, dr * rp * sr * (1 - sr))
    x2 = x.reshape(-1, x.shape[-1])
    dw = [x2.t() @ g for g in grads]
    return torch.cat(grads, -1).t(), *dw, torch.cat([w.t() for w in (wl, wlg, wr, wrg)])


def te_forward(x, gamma, beta, weight, bias, eps):
    mean = x.mean(-1)
    rstd = (x.var(-1, unbiased=False) + eps).rsqrt()
    act = (x - mean[:, None]) * rstd[:, None] * gamma + beta
    return act @ weight.t(), act, mean, rstd


def te_backward(dy, act, x, mean, rstd, gamma, weight, *, has_bias):
    grad = dy @ weight
    xhat = (x - mean[:, None]) * rstd[:, None]
    weighted = grad * gamma
    dx = (weighted - weighted.mean(-1, keepdim=True)
          - xhat * (weighted * xhat).mean(-1, keepdim=True)) * rstd[:, None]
    return dx, (grad * xhat).sum(0), grad.sum(0), dy.t() @ act, None


def gate_forward(x, proj, wg, residual, dropscale, *, seq_len):
    gate = (x @ wg).sigmoid()
    scale = dropscale.reshape(seq_len, -1)[torch.arange(x.shape[0]) % seq_len]
    return residual.reshape_as(proj) + scale * gate * proj, gate


def gate_backward(dy, proj, gate, dropscale, seq_len):
    scale = dropscale.reshape(seq_len, -1)[torch.arange(dy.shape[0]) % seq_len]
    grad = dy * scale
    return grad * gate, grad * proj * gate * (1 - gate)


@pytest.mark.parametrize("kind", ["out", "in", "bidir"])
@pytest.mark.parametrize("mask_kind", ["none", "holes", "zero", "fractional"])
@pytest.mark.parametrize("length", [3, 5])
@pytest.mark.parametrize("dropout", [False, True])
def test_hopper_trimul_all_gradient_slots_on_cpu(monkeypatch, kind, mask_kind, length, dropout):
    from miniworld_engine.kernels.trimul_inproj.cute import bidir_training as bi
    from miniworld_engine.kernels.trimul_inproj.cute import v6_training_merged as single
    module = bi if kind == "bidir" else single
    monkeypatch.setattr(module._bdll_patch, "apply", lambda: None)
    monkeypatch.setattr(module._gate_mul_patch, "apply", lambda: None)
    monkeypatch.setattr(module, "triton_layernorm", lambda x, w, b, eps: F.layer_norm(x, (x.shape[-1],), w, b, eps))
    monkeypatch.setattr(module, "trimul_inproj_cute_forward", front)
    monkeypatch.setattr(module, "front_bwd_dW", front_backward)
    monkeypatch.setattr(module, "_te_forward", te_forward)
    monkeypatch.setattr(module, "_te_backward", te_backward)
    monkeypatch.setattr(module, "gate_elem_train", gate_forward)
    monkeypatch.setattr(module, "gate_elem_bwd_ew", gate_backward)
    monkeypatch.setattr(bi.dispatch, "bmm", lambda name, a, b: torch.bmm(a, b))
    monkeypatch.setattr(bi.dispatch, "mm", lambda name, a, b: a @ b)
    monkeypatch.setattr(bi.dispatch, "pick", lambda name, key, cs: dict(cs)["cublas"]())
    torch.manual_seed(73)
    d, h = 4, 6 if kind == "bidir" else 4
    shapes = [(1, length, length, d), *[(d, h)] * 4, (d, d), (d, h),
              (d,), (d,), (h,), (h,)]
    actual = [(torch.randn(s, dtype=torch.float64) * .3).requires_grad_() for s in shapes]
    reference = [t.detach().clone().requires_grad_() for t in actual]
    scale = torch.ones(length * length, dtype=torch.float64)
    if mask_kind == "holes":
        scale[::3] = 0
    elif mask_kind == "zero":
        scale.zero_()
    elif mask_kind == "fractional":
        scale = torch.rand_like(scale)
    mask = None if mask_kind == "none" else scale
    dropscale = torch.ones(length, d, dtype=torch.float64)
    if dropout:
        dropscale = (torch.rand_like(dropscale) > .3) / .7
    ax, awl, awlg, awr, awrg, awg, awp, agi, abi, ago, abo = actual
    args = (ax, awl, awlg, awr, awrg, awg, awp, agi, abi, ago, abo, .03, None)
    if kind == "bidir":
        y = bi.bidir_forward(*args, h // 2, row_scale=mask, dropscale=dropscale, eps_out=.07)
    else:
        y = single.v6_forward_merged(*args, direction=kind, row_scale=mask,
                                    dropscale=dropscale, eps_out=.07)
    x, wl, wlg, wr, wrg, wg, wp, gi, bi_, go, bo = reference
    xn = F.layer_norm(x, (d,), gi, bi_, .03)
    left = (xn @ wl) * (xn @ wlg).sigmoid()
    right = (xn @ wr) * (xn @ wrg).sigmoid()
    if mask is not None:
        left, right = left * mask.reshape(1, length, length, 1), right * mask.reshape(1, length, length, 1)
    out = torch.einsum("bikh,bjkh->bijh", left, right)
    incoming = torch.einsum("bkih,bkjh->bijh", left, right)
    tri = (torch.cat((out[..., :h // 2], incoming[..., h // 2:]), -1)
           if kind == "bidir" else out if kind == "out" else incoming)
    expected = x + (F.layer_norm(tri, (h,), go, bo, .07) @ wp.t()) * (xn @ wg).sigmoid() * dropscale
    # Broadcast gradient exercises sum().backward() through the real autograd.Function.
    gradients = torch.autograd.grad(y.sum(), actual)
    expected_gradients = torch.autograd.grad(expected.sum(), reference)
    torch.testing.assert_close(y, expected, rtol=1e-9, atol=1e-10)
    for got, ref in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(got, ref, rtol=1e-8, atol=1e-10)
