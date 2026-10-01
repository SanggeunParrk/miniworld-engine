"""Public entry points of the fused ESMFold2 SWA atom DiT block (kernel family ``swa_dit``).

Provenance: moved from team-gm (``src/team_gm/modules/blocks/swa_fused_triton.py`` and ``swa_cuda/``, introduced by team-gm
commit 14f2c73 "research(swa): preserve opt-in fused atom transformer work", copied from team-gm 4fafa83). Kernel math is
unchanged; the engine owns the autotune keys, the config sets, the CUDA build and the runtime switches (``settings.swa_dit_*``).

The block is team-gm's ``SWAAtomBlock`` with ``block_style="esmfold2"`` -- the same equations as
``modules.swa_dit.SWADiTBlock`` -- over the flattened ``[N = A*B, S]`` atom sequence:

    mod = silu(c) @ Wmod^T -> shift_a | scale_a | gate_a | shift_f | scale_f | gate_f        (per (b, atom), hoisted)
    x   = rmsnorm(q) * (1 + scale_a) + shift_a
    q_h, k_h, v_h = split_heads(x @ Wqkv^T);  q_h, k_h = rope(rmsnorm_D(q_h)), rope(rmsnorm_D(k_h))
    o   = sliding-window attention (|i - j| <= half_window, keys / queries >= seqused masked, padding rows 0)
    q   = q + gate_a * ((sigmoid(x @ Wg^T) * o) @ Wo^T)
    y   = rmsnorm(q) * (1 + scale_f) + shift_f ;  q = q + gate_f * ((silu(y Wu1^T) * (y Wu2^T)) @ Wd^T)

Both RMSNorms and the per-head q/k RMSNorm use fp32's epsilon (what ``nn.RMSNorm(eps=None)`` does on bf16). The
modulation depends only on (b, atom): it is computed ONCE per (batch element, atom) from the augment-invariant
conditioning (``swa_dit_hoist_modulation``) and every augment ``a`` of batch element ``b`` -- rows ``(a*B + b)*S + s`` --
reads row ``b*S + s`` of it. A caller without that structure passes ``B = N`` (one modulation row per sequence row).

Served: bf16 or fp32 (all activations, conditioning and weights in one of the two), d_atom 128 with 4 heads of 32, SwiGLU
hidden 256, half window 64 (window 128), 3D-RoPE cos/sin of D/2 = 16 frequencies. bf16 runs the Triton kernels
(``triton/forward.py``, ``backward.py``) with hand-CUDA wgmma stages on sm_90; fp32 runs ``triton/forward_fp32.py`` and
``backward_fp32.py`` (Triton only), which keep the residual stream and every elementwise step in fp32, run the
projections as TF32 and the window attention on bf16 operands -- the precisions of the per-op fp32 path it replaces (see
those files). :func:`refusal` says why anything else is not served; it never raises.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: The contract the kernels are written for (literals in the CUDA tiles; tl.arange extents in Triton).
D_ATOM = 128
N_HEAD = 4
N_HIDDEN = 256
HALF_WINDOW = 64
FP32_EPS = float(torch.finfo(torch.float32).eps)


def refusal(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqused: torch.Tensor, wqkv: torch.Tensor,
            wg: torch.Tensor, wo: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, *, n_head: int, half_window: int,
            cond: torch.Tensor | None = None, wmod: torch.Tensor | None = None) -> str | None:
    """None if the fused block can run this call, else why it cannot. Never raises.

    ``q`` [N, S, C]; ``cos``/``sin`` the SWA attention params' RoPE tensors ([N or B, S, C/n_head/2] fp32);
    ``seqused`` [N] int32; the five block weights as the modules hold them (``Wqkv`` [3C, C], ``gate_proj`` [C, C],
    ``out_proj`` [C, C], ``w_up`` [2*hidden, C], ``w_down`` [C, hidden]). ``cond`` / ``wmod`` (the conditioning and the
    adaLN projection weight [6C, d_cond]) are checked when given. Only tensor metadata is read, so it traces under
    ``torch.compile`` without a graph break.
    """
    tensors = {"q": q, "Wqkv": wqkv, "gate_proj": wg, "out_proj": wo, "w_up": wu, "w_down": wd}
    if cond is not None:
        tensors["cond"] = cond
    if wmod is not None:
        tensors["adaln weight"] = wmod
    if q.dtype not in (torch.bfloat16, torch.float32):
        return f"the fused block runs bf16 or fp32, got q {q.dtype}"
    for name, t in tensors.items():
        if t.dtype != q.dtype:
            return (f"mixed dtypes: q is {q.dtype} but {name} is {t.dtype}; the fused block runs all-bf16 or all-fp32 "
                    f"(activations, conditioning and weights alike)")
    if q.dim() != 3:
        return f"q must be [N, S, d_atom], got {tuple(q.shape)}"
    c = q.shape[-1]
    if (c, n_head) != (D_ATOM, N_HEAD):
        return f"the kernels serve d_atom={D_ATOM} with {N_HEAD} heads, got d_atom={c}, n_head={n_head}"
    if half_window != HALF_WINDOW:
        return f"the fused block is validated at half_window={HALF_WINDOW} (window {2 * HALF_WINDOW}), got {half_window}"
    shapes = {"Wqkv": (wqkv, (3 * c, c)), "gate_proj": (wg, (c, c)), "out_proj": (wo, (c, c)),
              "w_up": (wu, (2 * N_HIDDEN, c)), "w_down": (wd, (c, N_HIDDEN))}
    for name, (t, want) in shapes.items():
        if tuple(t.shape) != want:
            return f"{name} must be {want} (SwiGLU hidden {N_HIDDEN}), got {tuple(t.shape)}"
    half = c // n_head // 2
    for name, t in (("cos", cos), ("sin", sin)):
        if t.dim() == 0 or t.dtype != torch.float32 or t.shape[-1] != half:
            return f"{name} must be fp32 [..., {half}], got {t.dtype} {tuple(t.shape)}"
        if t.requires_grad and torch.is_grad_enabled():
            return f"{name} requires grad; the fused block does not differentiate the RoPE angles"
    if seqused.dtype != torch.int32 or tuple(seqused.shape) != (q.shape[0],):
        return f"seqused must be int32 [N={q.shape[0]}], got {seqused.dtype} {tuple(seqused.shape)}"
    if cond is not None and wmod is not None:
        if cond.dim() != 3 or cond.shape[0] == 0 or cond.shape[1] != q.shape[1] or q.shape[0] % cond.shape[0] != 0:
            return f"cond must be [B, S, d_cond] with N % B == 0, got {tuple(cond.shape)} for q {tuple(q.shape)}"
        if tuple(wmod.shape) != (6 * c, cond.shape[-1]):
            return f"the adaLN weight must be [{6 * c}, d_cond={cond.shape[-1]}], got {tuple(wmod.shape)}"
    if not q.is_cuda:
        return "the input is not on a CUDA device"
    return None


def _mod_sm100(c_base: torch.Tensor, wmod: torch.Tensor) -> bool:
    """Whether the hoisted modulation runs on the sm_100a kernels: B200, bf16 c and Wmod, d_cond = C = 128, rows a multiple
    of 128, and the kernels load (``MINIWORLD_SWA_DIT_SM100=0`` or ``engine_backend="triton"`` keep the fp32 GEMM)."""
    import os

    from miniworld_engine import settings

    if os.environ.get("MINIWORLD_SWA_DIT_SM100", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    if not (c_base.is_cuda and c_base.dtype == torch.bfloat16 and wmod.dtype == torch.bfloat16):
        return False
    if c_base.shape[-1] != D_ATOM or tuple(wmod.shape) != (6 * D_ATOM, D_ATOM) or (c_base.numel() // D_ATOM) % 128:
        return False
    if torch.cuda.get_device_capability(c_base.device) != (10, 0):
        return False
    from miniworld_engine.kernels.swa_dit.dispatch import _sm100

    return _sm100(c_base.reshape(1, -1, D_ATOM), N_HIDDEN, HALF_WINDOW, FP32_EPS) is not None


def swa_dit_hoist_modulation(c_base: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """The block's adaLN modulation, once per (batch element, atom): ``[B*S, 6C]`` fp32 = silu(c) @ Wmod^T.

    ``c_base`` [B, S, d_cond] is the augment-invariant conditioning (``c[a*B + b] == c_base[b]``), or the per-row
    conditioning [N, S, d_cond] when there is no augment structure. silu runs in ``c_base``'s dtype, then the product is
    accumulated and kept in fp32 (as the engine's rmsnorm_adamod keeps its projections in registers). Differentiable.
    """
    if _mod_sm100(c_base, wmod):                         # B200, bf16, d_cond 128: the sm_100a mod_fwd / mod_bwd kernels
        from miniworld_engine.kernels.swa_dit.autograd import SWADiTModulationSm100
        from miniworld_engine.kernels.swa_dit.dispatch import swa_dit_mod_fwd_sm100

        c2 = c_base.reshape(-1, c_base.shape[-1]).contiguous()
        w = wmod.contiguous()
        if torch.is_grad_enabled() and (c2.requires_grad or w.requires_grad):
            return SWADiTModulationSm100.apply(c2, w)
        return swa_dit_mod_fwd_sm100(c2, w)
    a = F.silu(c_base).float()
    return (a.reshape(-1, a.shape[-1]) @ wmod.float().t()).contiguous()


def swa_dit_block(q: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqused: torch.Tensor,
                  wqkv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, B: int,
                  half_window: int = HALF_WINDOW) -> torch.Tensor:
    """The fused block, differentiable in ``q``, ``mod`` and the five weights. Returns [N, S, C] in q's dtype.

    ``q`` [N = A*B, S, C] bf16 or fp32 (the weights alike; gradients come back in each input's dtype); ``mod`` [B*S, 6C] fp32 (:func:`swa_dit_hoist_modulation`); ``cos``/``sin`` [B*S, C/8]
    or [B, S, C/8] fp32 -- one row per (batch element, atom), which is what ``build_attention_params`` repeats over the
    augments, so ``cos[:B]`` of its output is this argument; ``seqused`` [N] int32 (valid atoms front-packed per row).
    Check :func:`refusal` first. A call that records no gradient runs the inference forward (nothing saved).
    """
    from miniworld_engine.kernels.swa_dit.autograd import SWADiTBlockFunction
    from miniworld_engine.kernels.swa_dit.dispatch import swa_dit_block_fwd

    s = q.shape[1]
    cos = cos.reshape(-1, cos.shape[-1]).contiguous()
    sin = sin.reshape(-1, sin.shape[-1]).contiguous()
    mod = mod.reshape(-1, mod.shape[-1]).contiguous()
    if cos.shape[0] != B * s or mod.shape[0] != B * s or q.shape[0] % B != 0:
        msg = (f"swa_dit_block: B={B} batch elements of S={s} atoms need cos/sin/mod with {B * s} rows and N % B == 0; "
               f"got cos {tuple(cos.shape)}, mod {tuple(mod.shape)}, q {tuple(q.shape)}")
        raise ValueError(msg)
    q = q.contiguous()
    weights = [w.contiguous() for w in (wqkv, wg, wo, wu, wd)]
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, mod, *weights)):
        return SWADiTBlockFunction.apply(q, mod, cos, sin, seqused, *weights, B, half_window)
    return swa_dit_block_fwd(q, mod, cos, sin, seqused, *weights, B, half_window, FP32_EPS, False)[0]


__all__ = ["D_ATOM", "FP32_EPS", "HALF_WINDOW", "N_HEAD", "N_HIDDEN", "refusal", "swa_dit_block", "swa_dit_hoist_modulation"]
