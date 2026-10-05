"""A100 (sm_80) hand-CUDA gated projections (extension ``gated_sm80``, ``sm80/gp_ops.cu``): bf16, forward and backward.

* ``fused_gate_out(gate, out_r, wo)`` = ``(sigmoid(gate) * out_r) @ wo^T`` -- the registry's ``gated_linear`` op (d_hidden -> d_out): the forward GEMM forms its
  A tile as ``bf16(sigmoid(g) v)`` in shared memory (the gated activation never touches HBM); the backward is the dgrad GEMM with the gate-backward epilogue
  (``dv``, ``dg`` and the gated ``a``) and ``dW = dO^T a`` on cuBLAS.
* ``sigmoid_gate_fused(gate, out)`` / ``gated_residual(x, gate, branch)``: the one-pass elementwise gates, forward and backward.
* the triangle-multiplication stages ``tm1`` / ``tm2`` are in ``kernels/tm1/cuda`` and ``kernels/tm2/cuda`` (one extension, this one).

Gate: sm_80, bf16, ``d_hidden`` and ``d_out`` multiples of 64, the engine backend not forced to Triton; ``MINIWORLD_GATED_SM80=0`` turns it off.  The GEMM tile is the
TriMul's (``kernels/trimul_inproj/cuda/sm80/wide_gemm.cuh``).
"""

import functools
import os
import warnings
from pathlib import Path

import torch

from miniworld_engine import settings

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"
_tile_dir = Path(__file__).resolve().parents[2] / "trimul_inproj" / "cuda" / "sm80"     # wide_gemm.cuh, sm80_common.cuh


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="gated_sm80", sources=[str(_dir / "gp_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", f"-I{_tile_dir}"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the extension; False, with one warning, when the toolchain fails (the Triton kernels then serve).  A process-level constant for
    ``torch.compile``."""
    global _FAILED
    if _FAILED:
        return False
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _FAILED = True
        warnings.warn(f"sm_80 gated-projection kernels unavailable, keeping the Triton path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


@functools.lru_cache(maxsize=8)
def _is_ampere(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (8, 0)


def shape_ok(dtype, d_hidden: int, d_out: int) -> bool:
    """Device-free part of the gate: bf16 with widths the 64-column tiles divide."""
    return dtype is torch.bfloat16 and d_hidden > 0 and d_out > 0 and d_hidden % 64 == 0 and d_out % 64 == 0


def enabled() -> bool:
    return os.environ.get("MINIWORLD_GATED_SM80", "1") != "0" and settings.current().engine_backend != "triton"


def serves(d_hidden: int, d_out: int, device: torch.device, dtype) -> bool:
    """The gate of ``fused_gate_out`` (a pure function of the shape, the dtype and the card, so the dispatch can fold it at trace time)."""
    if not enabled() or not shape_ok(dtype, d_hidden, d_out) or device.type != "cuda":
        return False
    return _is_ampere(device.index if device.index is not None else torch.cuda.current_device()) and _loads()


def serves_elementwise(*tensors: torch.Tensor) -> bool:
    """The one-pass gates: bf16 contiguous CUDA tensors of one shape on an sm_80 card."""
    t0 = tensors[0]
    if not enabled() or not t0.is_cuda or t0.dtype is not torch.bfloat16:
        return False
    if any(t.shape != t0.shape or t.dtype is not t0.dtype or t.device != t0.device for t in tensors):
        return False
    return _is_ampere(t0.device.index if t0.device.index is not None else torch.cuda.current_device()) and _loads()


# --------------------------------------------------------------------------------------------------------------------- the gated output projection
def _gate_out_fwd_fake(g2, r2, wo):
    """(M, N): the projection replaces DH with wo's out_features."""
    return g2.new_empty((g2.shape[0], wo.shape[0]))


@opaque(fake=_gate_out_fwd_fake, name="gated_projection_sm80_gate_out_fwd")
def _gate_out_fwd(g2: torch.Tensor, r2: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """``(sigmoid(g2) * r2) @ wo^T`` for flat (M, DH) operands and ``wo`` [N, DH]."""
    out = g2.new_empty((g2.shape[0], wo.shape[0]))
    with torch.cuda.device(g2.device):
        _ext().gp_fwd(r2, g2, wo, out)
    return out


def _gate_out_dgrad_fake(do2, wo, g2, r2):
    """(d_out_r, d_gate, gated), all (M, DH) like g2 -- not do2's (M, N)."""
    return torch.empty_like(g2), torch.empty_like(g2), torch.empty_like(g2)


@opaque(fake=_gate_out_dgrad_fake, name="gated_projection_sm80_gate_out_dgrad")
def _gate_out_dgrad(do2: torch.Tensor, wo: torch.Tensor, g2: torch.Tensor, r2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One kernel: d_a = do2 @ wo (GEMM, fp32 accumulate) + the gate-backward epilogue -> (d_out_r, d_gate, gated)."""
    dr, dg, a = torch.empty_like(g2), torch.empty_like(g2), torch.empty_like(g2)
    with torch.cuda.device(g2.device):
        _ext().gp_dgrad(do2, wo.t().contiguous(), g2, r2, dr, dg, a)
    return dr, dg, a


class _GateOut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, outr, wo):
        shape = gate.shape
        dh = shape[-1]
        g2 = gate.reshape(-1, dh).contiguous()
        r2 = outr.reshape(-1, dh).contiguous()
        wo = wo.contiguous()
        out2 = _gate_out_fwd(g2, r2, wo)
        ctx.save_for_backward(g2, r2, wo)
        ctx.shape = shape
        return out2.reshape(*shape[:-1], wo.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        g2, r2, wo = ctx.saved_tensors
        do2 = grad_out.reshape(-1, wo.shape[0]).to(g2.dtype).contiguous()
        d_r, d_g, a = _gate_out_dgrad(do2, wo, g2, r2)
        d_wo = do2.transpose(0, 1) @ a                      # [N, DH]
        return d_g.reshape(ctx.shape), d_r.reshape(ctx.shape), d_wo


#: From this output width the fused GEMM is bound by the gate transform (every 128-column tile of the output re-forms the same gated A rows) and the one-pass CUDA
#: gate + cuBLAS (the "split" path: 5 passes over the activation instead of 3, but each at its own roofline) is faster: measured on an A100, 128 / 256 wide the fused
#: kernel wins or ties, 384 / 768 wide the split path is 1.3-1.6x faster.
SPLIT_MIN_N = 384


def fused_gate_out(gate: torch.Tensor, out_r: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """``(sigmoid(gate) * out_r) @ wo^T``; gate / out_r [..., DH], wo [N, DH] -> [..., N].  Call ``serves()`` first."""
    if wo.shape[0] >= SPLIT_MIN_N:
        return torch.nn.functional.linear(sigmoid_gate_fused(gate, out_r), wo)
    return _GateOut.apply(gate, out_r, wo)


def gated_projection(gate: torch.Tensor, x: torch.Tensor, out_weight: torch.Tensor) -> torch.Tensor:
    """``(sigmoid(gate) * x) @ out_weight`` with ``out_weight`` [hd, d] (a right-multiplied matrix: ``triton_gated_projection``'s contract)."""
    return _GateOut.apply(gate, x, out_weight.t())


# ------------------------------------------------------------------------------------------------------------------------------- elementwise gates
def _sigmul_fake(gate, out):
    """Same shape and dtype as gate."""
    return torch.empty_like(gate)


@opaque(fake=_sigmul_fake, name="gated_projection_sm80_sigmul_fwd")
def _sigmul(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``sigmoid(gate) * out`` in one pass (contiguous operands)."""
    a = torch.empty_like(gate)
    with torch.cuda.device(gate.device):
        _ext().sigmul_fwd(gate, out, a)
    return a


def _sigmul_grad_fake(da, gate, out):
    """(dgate, dout), shaped like gate and out."""
    return torch.empty_like(gate), torch.empty_like(out)


@opaque(fake=_sigmul_grad_fake, name="gated_projection_sm80_sigmul_bwd")
def _sigmul_grad(da: torch.Tensor, gate: torch.Tensor, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of ``sigmoid(gate) * out`` -> ``(dgate, dout)``."""
    dg, do = torch.empty_like(gate), torch.empty_like(out)
    with torch.cuda.device(gate.device):
        _ext().sigmul_bwd(da, gate, out, dg, do)
    return dg, do


class _SigmoidGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, out):
        gate, out = gate.contiguous(), out.contiguous()
        ctx.save_for_backward(gate, out)
        return _sigmul(gate, out)

    @staticmethod
    def backward(ctx, da):
        gate, out = ctx.saved_tensors
        return _sigmul_grad(da.contiguous(), gate, out)


def sigmoid_gate_fused(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``sigmoid(gate) * out`` in one pass (bf16, same shape).  Call ``serves_elementwise()`` first."""
    return _SigmoidGate.apply(gate, out)


def _gres_fake(x, gate, branch):
    """Same shape and dtype as x."""
    return torch.empty_like(x)


@opaque(fake=_gres_fake, name="gated_projection_sm80_gated_residual_fwd")
def _gres(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """``x + gate * branch`` with the eager product rounding (contiguous operands)."""
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _ext().gres_fwd(x, gate, branch, y)
    return y


def _gres_grad_fake(dy, gate, branch):
    """(dgate, dbranch)."""
    return torch.empty_like(gate), torch.empty_like(branch)


@opaque(fake=_gres_grad_fake, name="gated_projection_sm80_gated_residual_bwd")
def _gres_grad(dy: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate and branch gradients of the residual product."""
    dg, db = torch.empty_like(gate), torch.empty_like(branch)
    with torch.cuda.device(gate.device):
        _ext().gres_bwd(dy, gate, branch, dg, db)
    return dg, db


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, branch):
        x, gate, branch = x.contiguous(), gate.contiguous(), branch.contiguous()
        ctx.save_for_backward(gate, branch)
        return _gres(x, gate, branch)

    @staticmethod
    def backward(ctx, dy):
        gate, branch = ctx.saved_tensors
        dg, db = _gres_grad(dy.contiguous(), gate, branch)
        return dy, dg, db


def gated_residual(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """``x + gate * branch`` (bf16, identical shapes).  Call ``serves_elementwise()`` first."""
    return _GatedResidual.apply(x, gate, branch)
