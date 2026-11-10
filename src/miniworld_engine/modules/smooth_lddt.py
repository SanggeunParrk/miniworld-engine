"""Smooth lDDT loss (AlphaFold 3 Algorithm 27, the diffusion loss): ``SmoothLddtLoss``.

A :class:`~miniworld_engine.modules.primitives.BackendModule` without parameters; ``_forward_pytorch`` (the dense reference,
``functional.smooth_lddt_loss``) is the source of truth. The kernels (``kernels/smooth_lddt``) build no [N, N] float tensor: the ground-truth pair
set is kept as bit masks / tile flags, the augments (rigid motions of one structure) share it, and the backward is analytic.
"""

import torch
import torch.nn.functional as F

from miniworld_engine.modules.functional import smooth_lddt_loss
from miniworld_engine.modules.primitives import BackendModule, random_input, wide

#: Atoms per tile of both kernel backends: the loss is computed on N padded up to a multiple of it (padding atoms are unresolved: no pair).
_TILE = 128


class SmoothLddtLoss(BackendModule):
    """``mean_{augment a, batch b} (1 - smooth_lddt_ab)`` of AlphaFold 3 Algorithm 27::

        lddt_ab = sum_{pairs (i, j)} keep_ij * mean_t sigmoid(t - |d_pred_ij - d_gt_ij|) / sum keep_ij,     t in (0.5, 1, 2, 4) A

    ``keep`` = both atoms resolved, ``i != j`` and ``d_gt(i, j) < 30 A`` for a nucleotide atom ``i``, ``15 A`` otherwise. No parameters.

    ``forward(x_pred, gt, is_nuc, mask)``:

    * ``x_pred`` ``[A, B, N, 3]`` -- predicted (denoised) atom coordinates of ``A`` augments of ``B`` structures; the loss is differentiable in it;
    * ``gt`` ``[B, N, 3]`` -- the ground-truth coordinates (one structure per batch entry, the same for every augment); data: it gets no gradient;
    * ``is_nuc`` / ``mask`` ``[B, N]`` bool -- nucleotide atoms (radius 30 A instead of 15 A) / resolved atoms.

    Returns the scalar loss in at least fp32 (fp32 for fp32 / bf16 ``x_pred``, fp64 for fp64). Every backend computes in fp32.

    Parameters
    ----------
    backend : str
        See :class:`BackendModule`. ``"cuda"`` (H100 sm_90a, B200 sm_100a): the engine's hand-written CUDA kernels, one cubin per arch
        (``kernels/smooth_lddt/cuda``: pair-set prep, forward, analytic deterministic backward), ``x_pred`` fp32 or bf16 (the gradient comes back
        in its dtype). ``"triton"``: the portable Triton kernels (``kernels/smooth_lddt/triton``), any CUDA card, ``x_pred`` fp32 or bf16. Both
        pad ``N`` to a multiple of 128 themselves (padded atoms are unresolved).
    """
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda", "triton")

    def __init__(self, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.set_backend(backend)

    def _forward_pytorch(self, x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dtype = wide(x_pred)
        return smooth_lddt_loss(x_pred.to(dtype), gt.detach().to(dtype), is_nuc, mask)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """One structure of ``length`` atoms (a compact random globule, ~1 atom per 10 A^3), ``samples`` augments of it with 1 A coordinate noise
        (``x_pred`` in ``dtype``; ``gt`` fp32, the coordinates are data), one atom in eight a nucleotide, 5 % unresolved atoms."""
        gt = random_input((1, length, 3), dtype=torch.float32, device=device, generator=generator, scale=1.6 * length ** (1 / 3))
        noise = random_input((samples, 1, length, 3), dtype=torch.float32, device=device, generator=generator)
        is_nuc = torch.rand((1, length), device=device, generator=generator) < 0.125
        mask = torch.rand((1, length), device=device, generator=generator) < 0.95
        return {"x_pred": (gt + noise).to(dtype), "gt": gt, "is_nuc": is_nuc, "mask": mask}

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> str | None:
        if x_pred.device.type != "cuda":
            return "the kernels run on CUDA tensors only"
        if x_pred.dtype not in (torch.float32, torch.bfloat16):
            return f"x_pred must be fp32 or bf16, got {x_pred.dtype}"
        if x_pred.dim() != 4 or x_pred.shape[-1] != 3:
            return f"x_pred must be [A, B, N, 3], got {tuple(x_pred.shape)}"
        _a, b, n, _ = x_pred.shape
        if tuple(gt.shape) != (b, n, 3):
            return f"gt must be [B, N, 3] = {(b, n, 3)}, got {tuple(gt.shape)}"
        if tuple(is_nuc.shape) != (b, n) or tuple(mask.shape) != (b, n):
            return f"is_nuc / mask must be [B, N] = {(b, n)}, got {tuple(is_nuc.shape)} / {tuple(mask.shape)}"
        if is_nuc.dtype != torch.bool or mask.dtype != torch.bool:
            return f"is_nuc / mask must be bool, got {is_nuc.dtype} / {mask.dtype}"
        if x_pred.numel() == 0:
            return "empty input"
        return None

    def _supports_cuda(self, x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> str | None:
        why = self._supports_kernels(x_pred, gt, is_nuc, mask)
        if why is not None:
            return why
        cc = torch.cuda.get_device_capability(x_pred.device)
        if cc not in ((8, 0), (9, 0), (10, 0)):
            return f"the CUDA kernels are the A100 (sm_80), H100 (sm_90a) and B200 (sm_100a) cubins, got sm_{cc[0]}{cc[1]}"
        return None

    def _supports_triton(self, x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> str | None:
        return self._supports_kernels(x_pred, gt, is_nuc, mask)

    @staticmethod
    def _padded(x_pred, gt, is_nuc, mask):
        """The kernels' inputs: N padded to a multiple of 128 (padded atoms unresolved), gt fp32 without gradient, flags as int8."""
        pad = -x_pred.shape[2] % _TILE
        if pad:
            x_pred = F.pad(x_pred, (0, 0, 0, pad))
            gt = F.pad(gt, (0, 0, 0, pad))
            is_nuc = F.pad(is_nuc, (0, pad))
            mask = F.pad(mask, (0, pad))
        return x_pred, gt.detach().float().contiguous(), is_nuc.to(torch.int8).contiguous(), mask.to(torch.int8).contiguous()

    def _forward_cuda(self, x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.smooth_lddt.cuda.function import smooth_lddt_cuda

        return smooth_lddt_cuda(*self._padded(x_pred, gt, is_nuc, mask))

    def _forward_triton(self, x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.smooth_lddt.triton.function import smooth_lddt_triton

        return smooth_lddt_triton(*self._padded(x_pred, gt, is_nuc, mask))
