"""The former Triton launches of the helpers the H100 CUDA path replaced with PyTorch statements.

Moved here from ``kernels/augmented_attention/cuda`` and ``kernels/trimul_inproj/cuda`` (no production code calls them: H100 / B200 launch no
Triton, and a raw launch in a kernel module is a traced path the compile-wrap coverage test rejects). The Triton kernels themselves stay in those
modules; ``test_ops_no_triton_gpu.py`` compares each PyTorch replacement against its launch here, bit for bit.
"""

from __future__ import annotations

import torch
import triton

from miniworld_engine.kernels.augmented_attention import cuda as cu
from miniworld_engine.kernels.augmented_attention.cuda import fused_pair as fp
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_pack as pack


def prep_do_triton(do, o, A, L, nh=cu.H, dh=cu.D, write_dob=True):
    """The Triton launch of ``cu._prep_do`` (``_prep_do_kernel``)."""
    do = do.reshape(A * L, nh * dh).contiguous()
    dob = torch.empty(A * L, nh * dh, device=do.device, dtype=torch.bfloat16) if write_dob else do
    dd = torch.empty(A, nh, L, device=do.device, dtype=torch.float32)
    cu._prep_do_kernel[((A * L) // 8,)](
        do, o.reshape(A * L, nh * dh), dob, dd, L, NH=nh, DH=dh, DP=64, ROWS=8, WRITE_DOB=write_dob
    )
    return dob, dd


def bias_prep_triton(bias_hll):
    """The Triton launch of ``cu._bias_prep`` (``_scale_bf16_kernel``)."""
    bias_hll = bias_hll.contiguous()
    out = torch.empty(bias_hll.shape, device=bias_hll.device, dtype=torch.bfloat16)
    n = bias_hll.numel()
    cu._scale_bf16_kernel[(triton.cdiv(n, 4096),)](bias_hll, out, n, cu.LOG2E, BLOCK=4096)
    return out


def wf_triton(wb: torch.Tensor, lnw: torch.Tensor, need_nat: bool):
    """The Triton launch of ``fp._wf`` (``_wf_kernel``)."""
    wb, lnw = wb.contiguous(), lnw.contiguous()
    n = wb.numel()
    l2 = torch.empty(wb.shape, device=wb.device, dtype=torch.bfloat16)
    nat = torch.empty_like(l2) if need_nat else l2
    fp._wf_kernel[(triton.cdiv(n, 1024),)](wb, lnw, l2, nat, n, fp.LOG2E, BLOCK=1024, NAT=need_nat, DPK=wb.shape[1], F32=False)
    return l2, (nat if need_nat else None)


def pack_triton_launch(dst: torch.Tensor, wl: torch.Tensor, wlg: torch.Tensor, wr: torch.Tensor, wrg: torch.Tensor) -> None:
    """The Triton launch of ``pack._pack_blocks`` (``_pack``)."""
    r, d = wl.shape
    pack._pack[(triton.cdiv(4 * r * d, 1024),)](
        dst, wl, wlg, wr, wrg, r, d,
        *wl.stride(), *wlg.stride(), *wr.stride(), *wrg.stride(),
        BLOCK=1024, num_warps=4)


def wf32_triton(wb: torch.Tensor, lnw: torch.Tensor):
    """The Triton launch of ``fp._wf32`` (``_wf_kernel``)."""
    wb, lnw = wb.contiguous(), lnw.contiguous()
    n = wb.numel()
    out = torch.empty(wb.shape, device=wb.device, dtype=torch.float32)
    fp._wf_kernel[(triton.cdiv(n, 1024),)](wb, lnw, out, out, n, 1.0, BLOCK=1024, NAT=False, DPK=wb.shape[1], F32=True)
    return out
