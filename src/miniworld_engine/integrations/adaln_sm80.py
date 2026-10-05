"""Automatic dispatch to the A100 (sm_80) adaptive LayerNorm: hand-CUDA row passes and cuBLAS GEMMs, no Triton.

``AdaptiveLayerNorm`` computes ``y = sigmoid(scale) LN(x) + bias`` with ``[scale | bias] = (LN(cond) w) [Ws | Wb]^T + [sb | 0]`` (LN of x without affine, ``w`` the cond
norm's weight, ``Wb`` has no bias).  Contract: the engine's kernel backend (implementation TRITON or MINIWORLD) with the engine backend not forced to Triton, sm_80, bf16 or
fp32 operands (the module's compute dtype; fp32 runs on TF32 tensor cores), ``d_hidden`` and ``d_cond`` each one of 128 / 384 / 768, ``cond`` with the leading dims of ``x`` (one
conditioning per row) or, with no gradient, one conditioning shared by the samples of ``x`` (the first dim expanded: stride 0, or size 1).  ``MINIWORLD_ADALN_SM80=0`` turns it
off; a failed extension build warns once and keeps the module's path.

Inference (one opaque op): ``cond_ln`` (aff = LN(cond) w, one pass over the cond rows -- the L rows of a shared conditioning) -> one cuBLAS GEMM [S | B] = aff [Ws | Wb]^T (the weights
packed once per parameter version) -> ``adaln_epi`` (LN(x), sigmoid, the gate and the sum in one pass; row r reads table row r % period).
Training (an autograd Function; forward and backward each one opaque op): the same forward saving aff, the cond statistics, [S | B] and the x statistics; the backward
recomputes the gate from [S | B], writes ``D = [dscale | dy]`` and dx in one pass (``adaln_bwd_x``), then dcond_aff = D [Ws; Wb] and [dWs; dWb] = D^T aff on cuBLAS (fp32 outputs), then
``cond_ln_bwd`` (dcond, d w); the column sums (d sb, d w) and the casts of the weight gradients are one closing launch (``finish``) that adds the per-block partial rows in a fixed
order, so the backward is bit-reproducible.
Numerics and timings: ``docs/gpus/a100/adaptive_layernorm/adaptive_layernorm.md``.
"""

from __future__ import annotations

import contextlib
import os
import warnings
import weakref

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

WIDTHS = (128, 384, 768)
BF = torch.bfloat16
_packs: dict = {}
_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the sm_80 extension; False, with one warning, when the toolchain fails (the module path then serves).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _FAILED
    if _FAILED:
        return False
    try:
        _kernels().available()
    except Exception as exc:  # a toolchain problem keeps the module path
        _FAILED = True
        warnings.warn(f"sm_80 AdaLN kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _kernels():
    from miniworld_engine.kernels.adaln.cuda import sm80
    return sm80


def _period(x: torch.Tensor, cond: torch.Tensor, grad: bool) -> int | None:
    """Rows of x per distinct conditioning row: ``M`` (one conditioning per row), ``M / A`` (one conditioning shared by the A samples, a no-grad call), or None when
    ``cond`` does not describe the rows of ``x``."""
    lead, clead = tuple(x.shape[:-1]), tuple(cond.shape[:-1])
    m = 1
    for s in lead:
        m *= s
    if clead == lead:
        shared = len(lead) > 1 and lead[0] > 1 and cond.stride(0) == 0
        return m if (grad or not shared) else m // lead[0]
    if not grad and len(lead) > 1 and lead[0] > 1 and clead[0] == 1 and clead[1:] == lead[1:]:
        return m // lead[0]
    return None


def serves(module, x: torch.Tensor, cond: torch.Tensor, compute_dtype: torch.dtype, grad: bool) -> bool:
    """Whether ``module`` (an ``AdaptiveLayerNorm`` whose backend resolved to the engine's kernels) runs this call on the sm_80 path: ``compute_dtype`` is the dtype the module
    casts x / cond / the weights to, ``grad`` its ``needs_backward``."""
    if os.environ.get("MINIWORLD_ADALN_SM80", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    if not x.is_cuda or not cond.is_cuda or compute_dtype not in (BF, torch.float32):
        return False
    if x.shape[-1] not in WIDTHS or cond.shape[-1] not in WIDTHS or x.numel() == 0:
        return False
    if module.ln_cond.weight is None or module.to_scale.bias is None or module.to_bias.bias is not None:
        return False
    if torch.cuda.get_device_capability(x.device) != (8, 0):
        return False
    if compute_dtype is torch.float32 and not grad and x.shape[-1] == 128 and cond.shape[-1] == 128 and x.numel() // 128 < fp32_atom_min_rows():
        return False                 # fp32 inference of a few rows at the atom width: Triton's fused kernel is 10-12 % faster there than the TF32 kernel's launch + weight-staging prologue (the registry's N = 1024, A = 5)
    return _period(x, cond, grad) is not None and _loads()


def fp32_atom_min_rows() -> int:
    """Fewest rows of fp32 inference at the atom width (d = dc = 128) the sm_80 path serves (default 8192; ``MINIWORLD_ADALN_FP32_ATOM_MIN_ROWS=0`` serves them all: the fused TF32 kernel runs at any row count)."""
    return int(os.environ.get("MINIWORLD_ADALN_FP32_ATOM_MIN_ROWS", "8192"))


@contextlib.contextmanager
def _tf32(on: bool):
    """cuBLAS on TF32 tensor cores for the fp32 path's GEMMs, whatever the caller's allow_tf32 (restored after): the fp32 path is the TF32 recipe."""
    if not on:
        yield
        return
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` with an fp32 result (the gradients' GEMMs): bf16 operands accumulate into an fp32 output, fp32 operands run on TF32 tensor cores."""
    if a.dtype is BF:
        return torch.mm(a, b, out_dtype=torch.float32)
    with _tf32(True):
        return torch.mm(a, b)


def _wgrad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x^T y`` (a weight gradient: the sum over the rows of x [M, n1] and y [M, n2]) as an fp32 [n1, n2]: the GEMM is tall and skinny (K = M rows), and cuBLAS picks a pathological kernel
    (12 TFLOP/s against 120-150) for an fp32 result whose first dim is the smaller one (measured at M = 147456: [128 x 256], [256 x 128] transposed), so the larger operand goes first and the
    small result is transposed."""
    if x.shape[1] >= y.shape[1]:
        return _mm(x.t(), y)
    return _mm(y.t(), x).t()


_SIDE: dict = {}


class Branch:
    """GEMMs on a second stream beside the current stream's chain of row kernels and GEMMs: the gate GEMM of a ConditionedTransition (nothing needs it until the last pass) and the weight
    gradients (nothing needs them until the closing pass) are tensor-bound, the row kernels between them memory-bound, and the small-M GEMMs leave SMs idle in their last wave.  Every ``run`` starts after
    everything queued on the current stream so far; ``join`` (or ``wait`` for one result) makes the current stream wait.  The pattern of ``kernels/trimul_inproj/cuda/sm80.py``'s backward: the inputs
    are ``record_stream``-ed to the side stream, the outputs to the current stream at the join, so the caching allocator never recycles a tensor one stream still uses; CUDA-graph capture and replay fork
    and join as they do eagerly.  ``MINIWORLD_ADALN_BRANCH=0`` runs everything on the current stream."""

    def __init__(self, device: torch.device):
        self.main = torch.cuda.current_stream(device)
        self.enabled = os.environ.get("MINIWORLD_ADALN_BRANCH", "1") != "0"
        if self.enabled:
            if device not in _SIDE:
                _SIDE[device] = torch.cuda.Stream(device=device)
            self.side = _SIDE[device]
        self.outs: list[torch.Tensor] = []

    def run(self, fn, *inputs: torch.Tensor):
        """``fn()`` on the side stream (it may allocate); returns its tensor."""
        if not self.enabled:
            return fn()
        self.side.wait_stream(self.main)
        for t in inputs:
            t.record_stream(self.side)
        with torch.cuda.stream(self.side):
            out = fn()
        self.outs.append(out)
        return out

    def mark(self):
        """An event at the end of what the side stream has queued so far (None when the branch is off): ``wait`` for it later, while more work is queued behind it."""
        if not self.enabled:
            return None
        event = torch.cuda.Event()
        event.record(self.side)
        return event

    def wait(self, event, *outs: torch.Tensor) -> None:
        """The current stream waits for ``event`` (a ``mark``) and then uses ``outs``, results of the side stream."""
        if event is not None:
            self.main.wait_event(event)
            for t in outs:
                t.record_stream(self.main)

    def join(self) -> None:
        if self.enabled and self.outs:
            self.main.wait_stream(self.side)
            for t in self.outs:
                t.record_stream(self.main)


def _mm_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` in the operand dtype (the forward's GEMM: one rounding of the fp32 accumulator): bf16, or fp32 on TF32 tensor cores."""
    if a.dtype is BF:
        return torch.mm(a, b)
    with _tf32(True):
        return torch.mm(a, b)


# ------------------------------------------------------------------------------------------------------------ inference
def cached(tag: str, tensors: tuple, build):
    """``build()`` (packed / rounded copies of ``tensors``) kept per parameter version: the entry is valid while every tensor is the same live object at the same ``_version`` (a ``weakref`` check, so
    a temporary -- the cast of a master weight under autocast -- never hits a stale entry through a recycled address, and no entry keeps a tensor alive).  Inference reads the packs on every
    call, and repacking costs launches."""
    key = _capture.scoped((tag, *[(t.data_ptr(), t._version) for t in tensors]))
    if key is None:                  # inside a capture of unknown id: no caching
        return build()
    hit = _packs.get(key)
    if hit is not None and all(r() is t for r, t in zip(hit[0], tensors, strict=True)):
        return hit[1]
    _capture.prune(_packs)           # an eager entry serves eager calls only; a capture packs once, recorded; finished captures' entries go
    if len(_packs) >= 256:
        for k in [k for k, (refs, _) in _packs.items() if any(r() is None for r in refs)]:
            del _packs[k]
        if len(_packs) >= 256:
            _packs.clear()
    value = build()
    _packs[key] = (tuple(weakref.ref(t) for t in tensors), value)
    return value


def master(params: tuple, dt: torch.dtype) -> bool:
    """fp32 master parameters under bf16 kernels: the kernels get ``dt`` casts of the parameters (made outside autograd) and the weight gradients go back unrounded in fp32."""
    return dt is BF and params[0].dtype is torch.float32


def cast_params(params: tuple, dt: torch.dtype, cache: bool) -> tuple:
    """``params`` in ``dt`` (the parameters themselves when they already are); with ``cache`` the casts are kept per parameter version, keyed on the parameters (a cast is a new tensor
    each call), so the packs built from them hit too."""
    if all(p.dtype is dt for p in params):
        return tuple(params)
    def build():
        return tuple(p if p.dtype is dt else p.to(dt) for p in params)

    return cached(f"cast_{dt}", tuple(params), build) if cache else build()


def _wcat(ws: torch.Tensor, wb: torch.Tensor, cache: bool) -> torch.Tensor:
    """[Ws; Wb] [2 d, dc]; with ``cache`` kept per parameter version."""
    return cached("wcat", (ws, wb), lambda: torch.cat([ws, wb])) if cache else torch.cat([ws, wb])


def _inference_fake(x, cond, lnw, ws, sb, wb, eps_x, eps_c):
    return torch.empty_like(x)


@opaque(fake=_inference_fake, name="adaln_sm80_inference")
def inference(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws: torch.Tensor, sb: torch.Tensor, wb: torch.Tensor, eps_x: float,
              eps_c: float) -> torch.Tensor:
    """x [M, d] and cond [P, dc] contiguous rows (P divides M: row r of x takes conditioning row r % P), lnw [dc] fp32, Ws / Wb [d, dc], sb [d] (x's dtype) -> y [M, d]."""
    k = _kernels()
    with torch.cuda.device(x.device):
        if k.atom_supported(x, cond):                                       # d = dc = 128, bf16: the whole forward in one kernel
            return k.atom_fwd(x, cond, lnw, ws, wb, sb, eps_x, eps_c)[0]
        if k.atom_tf32_supported(x, cond):                                  # d = dc = 128, fp32: the same on TF32 tensor cores
            return k.atom_fwd_tf32(x, cond, lnw, ws, wb, sb, eps_x, eps_c)
        aff, _ = k.cond_ln(cond, lnw, x.dtype, eps_c)
        sbt = _mm_op(aff, _wcat(ws, wb, True).t())
        y, _ = k.epilogue(x, sbt, sb, eps_x)
    return y


# -------------------------------------------------------------------------------------------------------------- training
def _train_fwd_fake(x, cond, lnw, ws, sb, wb, eps_x, eps_c):
    m, d = x.shape
    p, dc = cond.shape
    e = x.new_empty
    if _kernels().atom_supported(x, cond):                                    # the fused atom kernels keep only the statistics
        return [torch.empty_like(x), e((0,)), e((p, 2), dtype=torch.float32), e((0,)), e((m, 2), dtype=torch.float32), e((0,))]
    return [torch.empty_like(x), e((p, dc)), e((p, 2), dtype=torch.float32), e((p, 2 * d)), e((m, 2), dtype=torch.float32), e((2 * d, dc))]


@opaque(fake=_train_fwd_fake, name="adaln_sm80_train_fwd")
def train_fwd(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws: torch.Tensor, sb: torch.Tensor, wb: torch.Tensor, eps_x: float,
              eps_c: float) -> list[torch.Tensor]:
    """The forward saving what the backward reads: [y, aff [M, dc], cond (mean, rstd) [M, 2], [S | B] [M, 2 d], x (mean, rstd) [M, 2], [Ws; Wb] [2 d, dc]]; cond is one row per row of x.
    At d = dc = 128 (bf16) one kernel writes y and the two statistics, and aff / [S | B] / the packed weights are empty (the backward kernel recomputes them)."""
    k = _kernels()
    with torch.cuda.device(x.device):
        if k.atom_supported(x, cond):
            y, xst, cst = k.atom_fwd(x, cond, lnw, ws, wb, sb, eps_x, eps_c, stats=True)
            return [y, x.new_empty((0,)), cst, x.new_empty((0,)), xst, x.new_empty((0,))]
        aff, cst = k.cond_ln(cond, lnw, x.dtype, eps_c, stats=True)
        wcat = torch.cat([ws, wb])                                         # the backward's dgrad GEMM reads it again: saved instead of packed twice
        sbt = _mm_op(aff, wcat.t())
        y, xst = k.epilogue(x, sbt, sb, eps_x, stats=True)
    return [y, aff, cst, sbt, xst, wcat]


def like(t: torch.Tensor, fp32: bool) -> torch.Tensor:
    """The dtype / shape reference ``finish`` casts to: ``t`` itself, or (``fp32``: the weight gradient stays fp32) a stride-0 fp32 view of ``t``'s shape, no allocation."""
    if not fp32 or t.dtype is torch.float32:
        return t
    return torch.empty((), dtype=torch.float32, device=t.device).expand(t.shape)


def _train_bwd_fake(dy, x, cond, lnw, ws, sb, wb, aff, cst, sbt, xst, wcat, fp32_grads=False):
    """Fake of ``train_bwd``: the gradient shapes and dtypes."""
    gd = torch.float32 if fp32_grads else ws.dtype
    return [torch.empty_like(x), torch.empty_like(cond), torch.empty(lnw.shape, dtype=lnw.dtype, device=lnw.device),
            torch.empty(ws.shape, dtype=gd, device=ws.device), torch.empty(sb.shape, dtype=gd, device=sb.device),
            torch.empty(wb.shape, dtype=gd, device=wb.device)]


@opaque(fake=_train_bwd_fake, name="adaln_sm80_train_bwd")
def train_bwd(dy: torch.Tensor, x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws: torch.Tensor, sb: torch.Tensor, wb: torch.Tensor, aff: torch.Tensor,
              cst: torch.Tensor, sbt: torch.Tensor, xst: torch.Tensor, wcat: torch.Tensor, fp32_grads: bool = False) -> list[torch.Tensor]:
    """[dx, dcond, d lnw, dWs, d sb, dWb] (each in its input's dtype; the weights' and the bias' fp32 when ``fp32_grads``: the master parameters' gradients, unrounded) from the gradient dy [M, d] and the forward's saved tensors."""
    k = _kernels()
    d = x.shape[1]
    with torch.cuda.device(x.device):
        if k.atom_supported(x, cond):                                        # d = dc = 128, bf16: the backward in one kernel, the two weight gradients on cuBLAS
            dx, dcond, dsc, aff, psb, plw = k.atom_bwd(dy, x, cond, xst, cst, lnw, ws, wb, sb)
            dsb, dlnw, dws, dwb = k.finish([psb, plw], [like(sb, fp32_grads), lnw], [_wgrad(dsc, aff), _wgrad(dy, aff)], [like(ws, fp32_grads), like(wb, fp32_grads)])
            return [dx, dcond, dlnw, dws, dsb, dwb]
        dm, dx, psb = k.bwd_x(dy, x, xst, sbt, sb, None, x.dtype)
        br = Branch(x.device)
        dw = br.run(lambda: _wgrad(dm, aff), dm, aff)                        # [2 d, dc] fp32: [dWs; dWb], beside the data gradient's GEMM and the cond row pass
        dca = _mm(dm, wcat)                                                  # [M, dc] fp32: dcond_aff
        dcond, pw = k.cond_bwd(dca, cond, cst, lnw, None)
        br.join()
        dsb, dlnw, dws, dwb = k.finish([psb, pw], [like(sb, fp32_grads), lnw], [dw[:d], dw[d:]], [like(ws, fp32_grads), like(wb, fp32_grads)])
    return [dx, dcond, dlnw, dws, dsb, dwb]


class _Training(torch.autograd.Function):
    """Takes the parameters themselves: the casts to the kernels' dtype happen here, outside autograd, and an fp32 master's weight gradients come back unrounded."""

    @staticmethod
    def forward(ctx, x, cond, lnw, ws, sb, wb, eps_x, eps_c):
        ctx.fp32_grads = master((ws,), x.dtype)
        ws, sb, wb = cast_params((ws, sb, wb), x.dtype, False)
        y, *saved = train_fwd(x, cond, lnw, ws, sb, wb, eps_x, eps_c)
        ctx.save_for_backward(x, cond, lnw, ws, sb, wb, *saved)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, cond, lnw, ws, sb, wb, *saved = ctx.saved_tensors
        dx, dcond, dlnw, dws, dsb, dwb = train_bwd(dy.contiguous(), x, cond, lnw, ws, sb, wb, *saved, ctx.fp32_grads)
        return dx, dcond, dlnw, dws, dsb, dwb, None, None


# ------------------------------------------------------------------------------------------------------------ module entry
def update(module, x: torch.Tensor, cond: torch.Tensor, compute_dtype: torch.dtype, grad: bool) -> torch.Tensor:
    """``AdaptiveLayerNorm`` forward on the sm_80 path (``serves`` must have accepted the call); ``grad`` = ``needs_backward``."""
    d, dc = x.shape[-1], cond.shape[-1]
    xs = x.to(compute_dtype).reshape(-1, d).contiguous()
    if grad:                         # one conditioning per row (a shared one is expanded: autograd sums the per-row gradients)
        cs = cond.to(compute_dtype).expand(*x.shape[:-1], dc).reshape(-1, dc).contiguous()
    else:
        cs = cond if _period(x, cond, False) == xs.shape[0] else cond[0]          # shared: the first sample's rows are the table
        cs = cs.to(compute_dtype).reshape(-1, dc).contiguous()
    ps = (module.to_scale.weight, module.to_scale.bias, module.to_bias.weight)
    eps = (float(module.ln_in.eps), float(module.ln_cond.eps))
    if grad:
        y = _Training.apply(xs, cs, module.ln_cond.weight, *ps, *eps)
    else:
        ws, sb, wb = cast_params(ps, compute_dtype, True)
        y = inference(xs, cs, module.ln_cond.weight, ws, sb, wb, *eps)
    return y.reshape(x.shape)
