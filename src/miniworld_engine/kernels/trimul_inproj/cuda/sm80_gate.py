"""A100 (sm_80) hand-CUDA output gate of the TriMul split back half (``trimul_inproj/triton/gate_elem.py``'s contract), bf16: the elementwise passes in CUDA, the GEMMs
(``x_n @ Wg``, ``d_glogit @ Wg^T``, ``x_n^T @ d_glogit``) on cuBLAS, as the Triton path.  The kernels live in ``gated_projection/cuda/sm80`` (one extension for the gated GEMMs
and gates).

* ``gate_elem_train``: ``y = residual + dropscale * (proj * sigmoid(x_n @ Wg))`` and the saved gate;
* ``gate_elem_bwd_ew``: ``(d_proj, d_glogit) = (dy ds gate, dy ds proj gate (1 - gate))`` (``from_preact``: the saved tensor is the pre-activation);
* ``gate_elem_bwd``: those and the two backward GEMMs.
"""

import torch

from ..._compile import opaque
from ...gated_projection.cuda import sm80 as _gp


def serves(x: torch.Tensor, n: int) -> bool:
    """sm_80, bf16, a width divisible by the 16-byte vector (8 elements)."""
    return x.dtype is torch.bfloat16 and n % 8 == 0 and _gp.serves_elementwise(x)


def _fwd_fake(glogit, proj, residual, dropscale):
    """``(y, gate)``, both (M, N) like the pre-activation."""
    return torch.empty_like(glogit), torch.empty_like(glogit)


@opaque(fake=_fwd_fake, name="trimul_sm80_gate_elem_fwd")
def _fwd(glogit: torch.Tensor, proj: torch.Tensor, residual: torch.Tensor, dropscale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """y = residual + dropscale ⊙ (proj ⊙ sigmoid(glogit)) and gate = sigmoid(glogit), flat (M, N) operands, dropscale [L, N]."""
    y, gate = torch.empty_like(glogit), torch.empty_like(glogit)
    with torch.cuda.device(glogit.device):
        _gp._ext().gate_elem_fwd(glogit, proj, residual, dropscale, y, gate)
    return y, gate


def gate_elem_train(x_n: torch.Tensor, proj: torch.Tensor, Wg: torch.Tensor, residual: torch.Tensor, dropscale: torch.Tensor,
                    seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """TRAINING gate ``y = residual + dropscale ⊙ (proj ⊙ sigmoid(x_n @ Wg))`` and the saved gate; the operands as ``gate_elem.gate_elem_train``."""
    xn_flat = x_n.reshape(-1, x_n.shape[-1])
    m, n = xn_flat.shape[0], proj.shape[-1]
    glogit = xn_flat @ Wg
    return _fwd(glogit.contiguous(), proj.reshape(m, n).contiguous(), residual.reshape(m, n).contiguous(), dropscale.reshape(-1, n).contiguous())


def _bwd_ew_fake(dy, proj, gate, dropscale, from_preact):
    """(d_proj, d_glogit), both (M, N) == dy flattened over its last dim."""
    shape = (dy.numel() // dy.shape[-1], dy.shape[-1])
    return dy.new_empty(shape), dy.new_empty(shape)


@opaque(fake=_bwd_ew_fake, name="trimul_sm80_gate_elem_bwd_ew")
def _bwd_ew(dy: torch.Tensor, proj: torch.Tensor, gate: torch.Tensor, dropscale: torch.Tensor, from_preact: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """d_proj = dy ds gate, d_glogit = dy ds proj gate (1 - gate), flat (M, N) operands, dropscale [L, N]."""
    dproj, dglogit = torch.empty_like(dy), torch.empty_like(dy)
    with torch.cuda.device(dy.device):
        _gp._ext().gate_elem_bwd(dy, proj, gate, dropscale, dproj, dglogit, from_preact)
    return dproj, dglogit


def gate_elem_bwd_ew(dy: torch.Tensor, proj: torch.Tensor, gate: torch.Tensor, dropscale: torch.Tensor, seq_len: int | None = None,
                     from_preact: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Just the elementwise backward: ``(d_proj, d_glogit)``, both (M, N)."""
    n = dy.shape[-1]
    m = dy.numel() // n
    return _bwd_ew(dy.reshape(m, n).contiguous(), proj.reshape(m, n).contiguous(), gate.reshape(m, n).contiguous(), dropscale.reshape(-1, n).contiguous(), from_preact)


def gate_elem_bwd(dy: torch.Tensor, x_n: torch.Tensor, proj: torch.Tensor, gate: torch.Tensor, Wg: torch.Tensor, dropscale: torch.Tensor,
                  seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of the gate: ``(d_proj, dx_gate, dWg)`` (dy / proj / gate (M, N), x_n (M, K), Wg (K, N))."""
    d_proj, d_glogit = gate_elem_bwd_ew(dy, proj, gate, dropscale, seq_len)
    return d_proj, d_glogit @ Wg.t(), x_n.reshape(d_glogit.shape[0], -1).t() @ d_glogit
