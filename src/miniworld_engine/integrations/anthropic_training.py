"""Training baseline using unchanged Anthropic native K1/K3 (Apache-2.0).

Forward: upstream K1 -> cuBLAS -> upstream K3, with fresh training workspaces.
Backward: PyTorch recomputation of the two surrounds + analytic cuBLAS contraction
gradients, using the actual saved native planes. This is a correctness baseline,
not an optimized native backward. First-order gradients only; eager API (Dynamo
graph break). BF16 pair activations, one pair plane per call, SM90 only initially.
"""
from importlib import import_module
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

WEIGHT_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp",
               "ln_out_w", "ln_out_b", "w_o", "w_og")


def native_ops():
    from .anthropic import configure
    configure()
    build = os.environ.get("TRIMUL_NATIVE_BUILD_DIR")
    if not build:
        raise RuntimeError("Anthropic native training requires TRIMUL_NATIVE_BUILD_DIR")
    py = Path(build).resolve().parent / "python"
    if not (py / "trimul_native/ops.py").is_file():
        raise RuntimeError("Rebuilt native payload must retain python/ beside build/")
    loaded = sys.modules.get("trimul_native")
    if loaded is not None and Path(loaded.__file__).resolve().parent != py / "trimul_native":
        raise RuntimeError("trimul_native already loaded from a different source")
    if str(py) not in sys.path:
        sys.path.insert(0, str(py))
    return import_module("trimul_native.ops")


def _linear(x, w):
    # Native WGMMA: BF16 operands, FP32 accumulators, no BF16 GEMM-output round.
    return F.linear(x.float(), w.to(torch.bfloat16).float())


def _norm(x, g, b, eps):
    return F.layer_norm(x.float(), (x.shape[-1],), g.float(), b.float(), eps).bfloat16()


def _front(z, w, mask, eps):
    xn = _norm(z, w[0], w[1], eps)
    a = torch.sigmoid(_linear(xn, w[2])) * _linear(xn, w[3])
    b = torch.sigmoid(_linear(xn, w[4])) * _linear(xn, w[5])
    a, b = (v * mask[..., None] for v in (a, b))
    return tuple(v.bfloat16().permute(2, 0, 1).contiguous() for v in (a, b))


def _back(z, tri, w, eps):
    xn = _norm(z, w[0], w[1], eps)
    norm = _norm(tri.permute(1, 2, 0), w[6], w[7], eps)
    return (_linear(norm, w[8]) * torch.sigmoid(_linear(xn, w[9]))).bfloat16()


def native_forward(z, mask, weights, direction, eps):
    """Launch original kernels with fresh buffers and freshly packed live weights."""
    ops = native_ops()
    n, _, c = z.shape
    h = weights[2].shape[0]
    np = ops.ceil16(n)
    packed = ops.pack_weights(dict(zip(WEIGHT_KEYS, weights)))
    kk = ops.kernels()
    ops._check(z, packed)
    ent = ops.K.lookup(kk.arch, c, h, "b", n,
                       "incoming" if direction == "incoming" else "outgoing", False)
    ab = torch.empty((2 * h, np, np), device=z.device, dtype=z.dtype)
    tri = torch.empty((h, np, np), device=z.device, dtype=z.dtype)
    out = torch.empty_like(z)
    # Incoming-only uses upstream's transposed K1 + NT contraction. Bidirectional
    # shares a natural-layout K1 and one global 2h-channel output normalization.
    kk.k1(z, mask, packed, ab, N=n, Np=np, cz=c, ch=h, cfg=ent["k1"],
          lnm=2, transpose=direction == "incoming", eps=eps, cache={})
    a, b = ab[:h], ab[h:]
    if direction == "bidirectional":
        half = h // 2
        torch.bmm(a[:half], b[:half].transpose(1, 2), out=tri[:half])
        torch.bmm(a[half:].transpose(1, 2), b[half:], out=tri[half:])
    else:
        torch.bmm(a, b.transpose(1, 2), out=tri)
    kk.k3(tri, z, packed, out, N=n, Np=np, cz=c, ch=h, residual=False,
          cfg=ent["k3"], lnm=1, eps=eps, cache={})
    return out, ab, tri


class _NativeTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, direction, eps, *weights):
        y, ab, tri = native_forward(z, mask, weights, direction, eps)
        ctx.save_for_backward(z, mask, ab, tri, *weights)
        ctx.direction, ctx.eps = direction, eps
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        z, mask, ab, tri, *weights = ctx.saved_tensors
        n, h = z.shape[0], tri.shape[0]
        # Differentiate the epilogue at the contraction actually produced by
        # native forward. Recompute LN/projection activations, not a Triton fwd.
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            x = z.detach().requires_grad_()
            w = [v.detach().requires_grad_() for v in weights]
            t = tri[:, :n, :n].detach().requires_grad_()
            ids = (0, 1, 6, 7, 8, 9)
            gy = torch.autograd.grad(_back(x, t, w, ctx.eps),
                                     (x, t, *(w[i] for i in ids)), dy)
        dx, dt, *gw = gy
        grads = dict(zip(ids, gw))
        a, b = ab[:h, :n, :n], ab[h:, :n, :n]
        if ctx.direction == "bidirectional":
            k = h // 2
            da, db = torch.empty_like(a), torch.empty_like(b)
            torch.bmm(dt[:k], b[:k], out=da[:k])
            torch.bmm(dt[:k].transpose(1, 2), a[:k], out=db[:k])
            torch.bmm(b[k:], dt[k:].transpose(1, 2), out=da[k:])
            torch.bmm(a[k:], dt[k:], out=db[k:])
        else:
            da = torch.bmm(dt, b)
            db = torch.bmm(dt.transpose(1, 2), a)
            if ctx.direction == "incoming":
                da, db = da.transpose(1, 2), db.transpose(1, 2)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            x = z.detach().requires_grad_()
            w = [v.detach().requires_grad_() for v in weights[:6]]
            gx = torch.autograd.grad(_front(x, w, mask, ctx.eps), (x, *w), (da, db))
        dx = (dx.float() + gx[0].float()).to(z.dtype)
        for i, v in enumerate(gx[1:]):
            grads[i] = ((grads[i].float() + v.float()).to(weights[i].dtype)
                        if i in grads else v)
        return dx, None, None, None, *(grads[i] for i in range(10))


@torch.compiler.disable
def triangle_multiplication_training(z, mask, *, weights, direction="outgoing", eps=1e-5):
    """Raw native update with first-order autograd; no dropout or residual here.

    FP32 surround recomputation requires TF32 disabled for its numerical contract.
    Unsupported shapes fail explicitly; this path never silently selects Triton.
    """
    if direction not in ("outgoing", "incoming", "bidirectional"):
        raise ValueError("Unknown triangle multiplication direction")
    if (z.ndim != 4 or z.shape[0] != 1 or z.shape[1] != z.shape[2]
            or z.dtype != torch.bfloat16 or not z.is_cuda):
        raise ValueError("Native training baseline requires CUDA BF16 [1,N,N,C]")
    if torch.cuda.get_device_capability(z.device) != (9, 0):
        raise ValueError("Native training baseline currently validated only on SM90")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Native reference backward requires torch.backends.cuda.matmul.allow_tf32=False")
    w = tuple(weights[k] for k in WEIGHT_KEYS)
    n, c, h = z.shape[1], z.shape[-1], w[2].shape[0]
    shapes = ((c,), (c,), (h,c), (h,c), (h,c), (h,c), (h,), (h,), (c,h), (c,c))
    if direction == "bidirectional" and h % 2:
        raise ValueError("Bidirectional hidden width must be even")
    for key, v, shape in zip(WEIGHT_KEYS, w, shapes):
        if tuple(v.shape) != shape or v.device != z.device or v.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"Invalid native training weight {key}: expected {shape} BF16/FP32 on {z.device}")
    if mask is None:
        mask = torch.ones((n,n), device=z.device, dtype=torch.float32)
    elif tuple(mask.shape) in ((n,n), (1,n,n)):
        mask = mask.reshape(n,n).to(torch.float32).contiguous()
    else:
        raise ValueError("Expected pair mask [N,N] or [1,N,N]")
    if mask.requires_grad or mask.device != z.device:
        raise ValueError("Mask must be non-differentiable and on the pair device")
    return _NativeTraining.apply(z[0].contiguous(), mask, direction, eps, *w).unsqueeze(0)


def module_update(module, pair, mask, *, bidirectional=False):
    if module.anthropic_row != "native_rebuilt":
        raise ValueError("Anthropic training requires explicit anthropic_row='native_rebuilt'")
    if module.ln_pair.eps != module.ln_out.eps:
        raise ValueError("Native TriMul requires equal LayerNorm epsilons")
    weights = dict(zip(WEIGHT_KEYS, (module.ln_pair.weight, module.ln_pair.bias,
        module.to_left_gate.weight, module.to_left.weight,
        module.to_right_gate.weight, module.to_right.weight,
        module.ln_out.weight, module.ln_out.bias, module.to_out.weight, module.to_gate.weight)))
    pairmask = None if mask is None else mask.unsqueeze(-1) & mask.unsqueeze(-2)
    direction = "bidirectional" if bidirectional else ("outgoing" if module.outgoing else "incoming")
    module.anthropic_selection = dict(row="native_rebuilt", forward="original K1 + cuBLAS + original K3",
        backward="PyTorch surround recomputation + cuBLAS contraction gradients",
        residual_dropout="external", direction=direction, compile="eager graph break")
    return triangle_multiplication_training(pair, pairmask, weights=weights,
                                            direction=direction, eps=module.ln_pair.eps)
