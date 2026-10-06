"""Fused token DiT TRAINING block on H100 (sm_90) and B200 (sm_100), wired to DiTBlock's parameter contract.

One autograd Function per ``DiTBlock`` call, forward and backward, with no Triton and no quack, in one of two precisions --
bf16 (bf16 inputs, or ``compute_dtype=bf16``) or fp32 (fp32 inputs, no ``compute_dtype``; TF32 tensor cores):

  * GEMMs: cuBLAS, operands in the path's dtype (bf16, or fp32 on TF32 tensor cores whatever the caller's allow_tf32), fp32
    accumulation, fp32 weight gradients;
  * the six conditioning projections as two GEMMs (every block-internal LayerNorm of cond shares its statistics, so its
    weight folds into the projection; the two output gates read raw cond), q|k|v|g as one GEMM;
  * the pair bias: a CUDA LayerNorm of the pair rows and one GEMM, head-major, the shared key mask folded in as -inf (fp32: one
    row kernel each way, ``pair_bias`` / ``pair_bias_bwd``, LN(pair) never stored);
  * fp32 only, copies the bf16 path makes and this one does not: no cond copy (the gate GEMM reads cond), no v copy (QK-norm:
    the core reads v from qkvg), dQ / dK / dV written by the core straight into the q|k|v|g gradient, the dxt / dxa GEMMs
    straight into their blocks of the conditioning gradient; the row kernels run one resident wave of blocks (``row_grid``);
  * the attention core: ``kernels/augmented_attention/cuda/sm100`` -- bf16: attn_fwd2, attn_dkv + attn_dqb; fp32: attn_fwd_tf32,
    attn_dkv_tf32 + attn_dqb_tf32 (kind::tf32 MMAs, fp32 softmax);
  * every elementwise / row step: ``kernels/conditioned_transition/cuda/token_dit_train_rows.cu``; in bf16 the expand GEMM
    carries the SwiGLU (``gemm_swiglu2_sm100.cu -DSAVE_AB``, which also writes [a | b] for the backward);
  * the residual stream fp32 inside the block, the output in the input's dtype.

``serves()`` is the whole gate: autograd on, the engine's kernel backend (implementation TRITON or MINIWORLD), B200, a bf16 or
fp32 input (bf16 operands when the input or ``compute_dtype`` is bf16, else fp32), the token widths (cond 384 / pair 128 /
transition n = 2) at a head layout of LAYOUTS (16 x 48, 24 x 32, 12 x 64 at d 768, 16 x 64 at d 1024; bf16 and fp32 alike),
B == 1, a key mask of [B, L] or none (``DiTBlock`` hands an [A, B, L] mask shared by the samples over as its [B, L] row),
LayerNorm eps 1e-5. Anything else keeps the module path. MINIWORLD_TOKEN_DIT_TRAIN=0 turns it off.

Any A and L: the kernels tile L by 128 and pair the samples, so ``block`` pads L to the next multiple of 128 and A to an even
count -- zero single / cond rows and samples, zero pair rows and columns, the padded keys masked out -- and returns the first
A samples' first L rows. The padding is computed and dropped; it adds nothing to any gradient.

The precision follows the tensors, not an enclosing autocast: the forward and backward run with autocast off (bf16-mixed
training hands an fp32 diffusion block fp32 activations, which take the fp32 / TF32 kernels).

The forward and the backward are each one opaque op (``kernels/_compile.opaque``) with a fake implementation, inside an
autograd Function, so torch.compile keeps them as single nodes (as the H100 TriMul training path does).

Unlike the research stack runner (branch ``b200/token-dit``), a DiTBlock call sees one block, so the pair bias and the cond
LayerNorm are computed per block rather than once per stack.
"""

from __future__ import annotations

import contextlib
import os

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

D, DC, DP, H, DH = 768, 384, 128, 16, 48          # the default layout; serves() takes LAYOUTS
#: (heads, d_single): 16 x 48, 24 x 32, 12 x 64 and 16 x 64 (d 1024), bf16 and fp32 (the TF32 cores are built per layout too)
LAYOUTS = ((16, 768), (24, 768), (12, 768), (16, 1024))
QM = 128                                          # the attention kernels' query / key tile: L is padded to a multiple of it
EPS = 1e-5
BF = torch.bfloat16

ATT = ("ada_ln_in.ln_cond.weight", "ada_ln_in.to_scale.weight", "ada_ln_in.to_scale.bias", "ada_ln_in.to_bias.weight",
       "to_scale.weight", "to_scale.bias", "to_query.weight", "to_query.bias", "to_key.weight", "to_value.weight",
       "to_gate.weight", "to_out.weight", "ln_pair.weight", "to_bias.weight")
QKN = ("norm_query.weight", "norm_key.weight")
TRN = ("ada_ln_in.ln_cond.weight", "ada_ln_in.to_scale.weight", "ada_ln_in.to_scale.bias", "ada_ln_in.to_bias.weight",
       "to_scale.weight", "to_scale.bias", "expand_a.weight", "expand_b.weight", "squeeze.weight")


def serves(module, single, cond, pair, mask, compute_dtype=None) -> bool:
    if os.environ.get("MINIWORLD_TOKEN_DIT_TRAIN", "1") == "0" or not torch.is_grad_enabled():
        return False
    if settings.current().engine_backend == "triton":
        return False
    a = module.attention
    # the engine's kernels (TRITON or MINIWORLD resolve to them), as the sm_90 / sm_100 attention cores gate on
    from miniworld_engine.modules.dispatch import KernelBackend
    if a._backend != KernelBackend.TRITON:
        return False
    cap = torch.cuda.get_device_capability(single.device) if single.is_cuda else None
    if cap not in ((9, 0), (10, 0)):
        return False
    if single.dtype not in (BF, torch.float32) or compute_dtype not in (None, BF, torch.float32):
        return False
    if not any(t.requires_grad for t in (single, cond, pair)) and not any(p.requires_grad for p in module.parameters()):
        return False
    # any A and L >= 1: ``block`` pads L to a multiple of 128 and A to an even count (the kernels' tiles / sample pairs)
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[0] < 1 or single.shape[2] < 1:
        return False
    d = single.shape[-1]
    if tuple(cond.shape) != (*single.shape[:3], DC) or tuple(pair.shape) != (1, single.shape[2], single.shape[2], DP):
        return False
    if (a.n_head, d) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * d:
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) == (1, single.shape[2])):
        return False
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, module.transition.ada_ln_in.ln_in,
             module.transition.ada_ln_in.ln_cond)
    return all(n.eps == EPS for n in norms)


def _f32(t):
    return t.detach().float().contiguous()


def operand_dtype(single, compute_dtype=None) -> torch.dtype:
    """The dtype of every GEMM operand / saved activation: bf16 when the input or ``compute_dtype`` is bf16, else fp32."""
    return BF if (single.dtype is BF or compute_dtype is BF) else torch.float32


@contextlib.contextmanager
def _tf32(on: bool):
    """cuBLAS on TF32 tensor cores for the fp32 path's GEMMs (restored after): the fp32 path is the TF32 recipe, and IEEE fp32
    GEMMs ran the block 5-6x slower."""
    if not on:
        yield
        return
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


_WGRAD_SPLIT = os.environ.get("MINIWORLD_TDIT32_WGRAD_SPLIT", "1") != "0"


def _mm32(a, b):
    """a @ b with an fp32 result (weight gradients): bf16 operands accumulate into fp32 output, fp32 ones already are.

    fp32 (TF32) weight gradients with few output tiles (dWo 768 x 768, dWg 1536 x 384, dWn 3072 x 384, dWsq 768 x 1536: under
    one wave of 128 x 128 tiles) and a long reduction (the A L rows) ran at 370-580 TF/s against ~740 for the wide ones: they are
    split over the rows into S <= 8 chunks as one batched GEMM, then summed (MINIWORLD_TDIT32_WGRAD_SPLIT=0: one GEMM)."""
    if a.dtype is BF:
        return torch.mm(a, b, out_dtype=torch.float32)
    n, m = a.shape
    k = b.shape[1]
    tiles = -(-n // 128) * -(-k // 128)
    nsm = torch.cuda.get_device_properties(a.device).multi_processor_count
    s = 1
    if _WGRAD_SPLIT and tiles < 128 and b.is_contiguous() and a.t().is_contiguous():
        while s < 8 and tiles * s * 2 <= 2 * nsm and m % (2 * s * 128) == 0:
            s *= 2
    if s == 1:
        return torch.mm(a, b)
    return torch.bmm(a.t().reshape(s, m // s, n).transpose(1, 2), b.reshape(s, m // s, k)).sum(0)


LOG2E = 1.4426950408889634
QS90 = LOG2E / DH ** 0.5           # the sm_90a core's q scale: its logits are in log2 units (16 x 48 only)


def _sm90(dev) -> bool:
    """H100: the sm_90a attention cores and their operand conventions; B200 otherwise (``serves`` admits only these two)."""
    return torch.cuda.get_device_capability(dev) == (9, 0)


def _key_mask(mask, A, L, perm=None):
    """The sm_90a cores' additive key mask, [A, L] fp32 (0 = attend), or None; ``perm``: its keys in that order (the TF32
    core's)."""
    if mask is None:
        return None
    m = mask.reshape(L) if perm is None else mask.reshape(L)[perm]
    return torch.where(m, 0.0, -1e30).to(torch.float32).expand(A, L).contiguous()


_PACKS: dict = {}
_PACK_STREAMS: dict = {}

def _native_rpb():
    return int(os.environ.get("MINIWORLD_TOKEN_DIT_H100_TRAIN_RPB", "2"))


def _native90(dev):
    return os.environ.get("MINIWORLD_TOKEN_DIT_H100_TRAIN_NATIVE", "1") == "1" and _sm90(dev)


_NATIVE_CALLER_TASKS: set[int] = set()


def _activate_native_caller(*, force: bool = False):
    """Limit installed hooks to backward tasks that actually execute our block."""
    if (not (force or os.environ.get("MINIWORLD_DIT_H100_CALLER_NATIVE", "0") == "1")
            or torch.compiler.is_compiling()):
        return
    task = torch._C._current_graph_task_id()
    if task >= 0 and task not in _NATIVE_CALLER_TASKS:
        _NATIVE_CALLER_TASKS.add(task)
        torch.autograd.Variable._execution_engine.queue_callback(lambda: _NATIVE_CALLER_TASKS.discard(task))


def _native_caller_output(output, *tensors, force: bool = False):
    """Bind accumulation after the native output node finishes backward."""
    if (torch.compiler.is_compiling() or not torch.is_grad_enabled()
            or not output.requires_grad or output.grad_fn is None):
        return output
    _native_caller_hooks(*tensors, force=force)

    def activate(grad_inputs, grad_outputs):
        # A tensor hook runs before the native backward. Activating there
        # would leave a task tag when that backward raises, since engine
        # completion callbacks do not run on errors. This node hook runs
        # only after a successful backward, before its leaf consumers.
        _activate_native_caller(force=force)

    output.grad_fn.register_hook(activate)
    return output


def _native_caller_hooks(*tensors, force: bool = False):
    """Keep caller autograd edges while replacing their CUDA sum/add operations.

    Hooks bracket individual existing nodes; no mode escapes to other nodes or
    the caller, and gradients remain ordinary Tensors. This experimental option
    is for the manual CUDA-graph path, not a torch.compile transformation.
    """
    if (not (force or os.environ.get("MINIWORLD_DIT_H100_CALLER_NATIVE", "0") == "1")
            or torch.compiler.is_compiling()):
        return
    import threading
    from torch.autograd.graph import get_gradient_edge
    from torch.utils._python_dispatch import TorchDispatchMode
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

    kernels = ext(rpb=_native_rpb())

    class CallerMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            try:
                return self._dispatch(func, args, kwargs)
            except BaseException:
                # Engine completion callbacks do not run after a node throws.
                # The engine restores its dispatch stack; release our task tag.
                _NATIVE_CALLER_TASKS.discard(torch._C._current_graph_task_id())
                raise

        def _dispatch(self, func, args, kwargs):
            kwargs = kwargs or {}
            if func is torch.ops.aten.sum.dim_IntList:
                dims = args[1] if len(args) > 1 else kwargs.get("dim")
                keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
                x = args[0]
                if (tuple(dims or ()) == (0,) and x.is_cuda and x.ndim >= 1
                        and x.is_contiguous() and x.dtype in (BF, torch.float32)
                        and kwargs.get("dtype") in (None, x.dtype)):
                    out = kernels.caller_sum90(x)
                    return out if keep else out.squeeze(0)
            elif func is torch.ops.aten.add_.Tensor:
                dst, src = args[:2]
                if (dst.is_cuda and src.device == dst.device and src.dtype == dst.dtype
                        and dst.dtype in (BF, torch.float32) and dst.shape == src.shape
                        and kwargs.get("alpha", 1) == 1):
                    # Keep ATen's overlap checks for partially aliased views.
                    # Exact self-add is safe; disjoint allocations are usual
                    # for AccumulateGrad's destination and incoming gradient.
                    overlap = torch._C._overlaps(dst, src)
                    if not overlap or (dst.data_ptr() == src.data_ptr()
                                       and dst.stride() == src.stride()):
                        if dst.is_contiguous() and src.is_contiguous():
                            kernels.caller_add90(dst, src)
                            return dst
                        if (dst.ndim == 2 and src.ndim == 2
                                and dst.t().is_contiguous() and src.t().is_contiguous()):
                            # Column-major matrix gradients need only metadata
                            # views; both operands retain identical element order.
                            kernels.caller_add90(dst.t(), src.t())
                            return dst
            elif func in (torch.ops.aten.zeros.default, torch.ops.aten.new_zeros.default):
                if func is torch.ops.aten.zeros.default:
                    out = torch.empty(args[0], **kwargs)
                else:
                    out = args[0].new_empty(args[1], **kwargs)
                if out.is_cuda and out.is_contiguous():
                    kernels.caller_zero90(out)
                    return out
            elif func is torch.ops.aten.zero_.default:
                if args[0].is_cuda and args[0].is_contiguous():
                    kernels.caller_zero90(args[0])
                    return args[0]
            return func(*args, **kwargs)

    def install(node):
        if node.metadata.get("miniworld_h100_native_caller"):
            return
        state = threading.local()

        def pre(grad_outputs):
            state.mode = (CallerMode() if torch._C._current_graph_task_id() in _NATIVE_CALLER_TASKS else None)
            if state.mode is not None:
                state.mode.__enter__()

        def post(grad_inputs, grad_outputs):
            if state.mode is not None:
                state.mode.__exit__(None, None, None)
            del state.mode

        node.register_prehook(pre)
        node.register_hook(post)
        node.metadata["miniworld_h100_native_caller"] = True

    for tensor in tensors:
        if not tensor.requires_grad:
            continue
        node = get_gradient_edge(tensor).node
        if node.name().endswith("AccumulateGrad"):
            install(node)
        elif node.name() in ("ExpandBackward0", "AsStridedBackward0") and tensor.stride(0) == 0:
            install(node)
            for ancestor, _ in node.next_functions:
                if ancestor is not None and ancestor.name().endswith("AccumulateGrad"):
                    install(ancestor)



def _pack(P, qk, dev, at):
    """The block's weights in the kernels' layouts (GEMM operands in ``at``, folded cond / pair LayerNorm weights, fp32 bias
    vectors). Rebuilt only when a parameter changes -- an optimizer step bumps ``_version`` -- so a training step pays
    it once per block, not once per forward. Scoped to the CUDA-graph capture (``kernels._capture``): an eager pack is never
    reused inside a capture, whose replays would otherwise run on the weights of capture time after every optimizer step."""
    native = _native90(dev)
    pipeline = native and os.environ.get("MINIWORLD_TOKEN_DIT_H100_TRAIN_PACK_ASYNC", "0") == "1"
    key = (native, pipeline, *((t.data_ptr(), t._version) for t in P.values()))
    slot = _capture.scoped((next(iter(P.values())).data_ptr(), at))
    hit = None if slot is None else _PACKS.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    if native:
        from miniworld_engine.integrations.token_dit import WEIGHTS, QK_WEIGHTS
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext
        names = WEIGHTS + (QK_WEIGHTS if qk else ())
        keys = ("w1", "w2", "Wraw", "Wn", "Wg", "bs1", "bs2", "bg1", "bg2", "Wqkvg", "bqkvg",
                "nq", "nk", "wp", "Wb", "Wf", "Wf_at", "Wf_l2", "Wo", "Wab", "Wsq")
        T = ext(rpb=_native_rpb())
        weights = [P[n] for n in names]
        if pipeline:
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            if idx not in _PACK_STREAMS:
                _PACK_STREAMS[idx] = torch.cuda.Stream(device=dev)
            stream = _PACK_STREAMS[idx]
            out = T.train_alloc90(weights, qk, at is torch.float32)
            stream.wait_stream(torch.cuda.current_stream(dev))
            cond_ready, ready = torch.cuda.Event(), torch.cuda.Event()
            with torch.cuda.stream(stream):
                T.train_stage90(weights, out, qk, at is torch.float32, 0, 9)
                cond_ready.record(stream)
                T.train_stage90(weights, out, qk, at is torch.float32, 9, 21)
                ready.record(stream)
            W = dict(zip(keys, out, strict=True))
            W["_cond_ready"], W["_pack_ready"] = cond_ready, ready
        else:
            W = dict(zip(keys, T.train_pack90(weights, qk, at is torch.float32), strict=True))
        if slot is not None:
            _capture.prune(_PACKS)
            _PACKS[slot] = (key, W)
        return W
    g = lambda n: P[n]
    D = g("attention.to_out.weight").shape[0]
    DH = D // g("attention.to_bias.weight").shape[0]
    w1, w2 = _f32(g("attention.ada_ln_in.ln_cond.weight")), _f32(g("transition.ada_ln_in.ln_cond.weight"))
    Wraw = torch.cat([_f32(g("attention.ada_ln_in.to_scale.weight")), _f32(g("attention.ada_ln_in.to_bias.weight")),
                      _f32(g("transition.ada_ln_in.to_scale.weight")), _f32(g("transition.ada_ln_in.to_bias.weight"))])
    wp, Wb = _f32(g("attention.ln_pair.weight")), _f32(g("attention.to_bias.weight"))
    Wf = Wb * wp
    W = {
        "w1": w1, "w2": w2, "Wraw": Wraw, "Wn": torch.cat([Wraw[:2 * D] * w1, Wraw[2 * D:] * w2]).to(at),   # cond-LN weights folded
        "Wg": torch.cat([g("attention.to_scale.weight"), g("transition.to_scale.weight")]).detach().to(at),
        "bs1": _f32(g("attention.ada_ln_in.to_scale.bias")), "bs2": _f32(g("transition.ada_ln_in.to_scale.bias")),
        "bg1": _f32(g("attention.to_scale.bias")), "bg2": _f32(g("transition.to_scale.bias")),
        "Wqkvg": torch.cat([g("attention.to_query.weight"), g("attention.to_key.weight"), g("attention.to_value.weight"),
                         g("attention.to_gate.weight")]).detach().to(at),
        "bqkvg": torch.cat([g("attention.to_query.bias").detach().float(), torch.zeros(3 * D, device=dev)]).to(at),
        "nq": _f32(g("attention.norm_query.weight")) if qk else torch.ones(DH, device=dev),
        "nk": _f32(g("attention.norm_key.weight")) if qk else torch.ones(DH, device=dev),
        "wp": wp, "Wb": Wb, "Wf": Wf, "Wf_at": Wf.to(at), "Wf_l2": (Wf * LOG2E).to(at),    # natural / log2 units
        "Wo": g("attention.to_out.weight").detach().to(at).contiguous(),
        "Wab": torch.cat([g("transition.expand_a.weight"), g("transition.expand_b.weight")]).detach().to(at),
        "Wsq": g("transition.squeeze.weight").detach().to(at).contiguous(),
    }
    if slot is not None:
        _capture.prune(_PACKS)
        _PACKS[slot] = (key, W)
    return W


_GSW: dict = {}


def _swiglu_gemm(xt, wab):
    """The sm_100a expand GEMM with the SwiGLU epilogue that also writes [a | b] (``gemm_swiglu2_sm100.cu -DSAVE_AB``) where
    it applies (bf16, B200, M >= its row threshold, MINIWORLD_TOKEN_DIT_GEMM_SWIGLU); None keeps cuBLAS + the SwiGLU rows."""
    from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu
    if not gemm_swiglu.supported(xt, wab):
        return None
    idx = xt.device.index if xt.device.index is not None else torch.cuda.current_device()
    if idx not in _GSW:
        try:
            _GSW[idx] = gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2, save_ab=True)
        except Exception as exc:         # a failed build keeps cuBLAS + the row pass
            import warnings
            warnings.warn(f"sm100 SwiGLU GEMM unavailable, keeping cuBLAS: {exc!r}", RuntimeWarning, stacklevel=2)
            _GSW[idx] = None
    return _GSW[idx]


def _names_for(qk):
    return ["attention." + n for n in ATT + (QKN if qk else ())] + ["transition." + n for n in TRN]


def _own_cond(cond, fp32) -> bool:
    """fp32 path with an fp32 cond: the gate GEMM (and its weight gradient) reads cond itself, no operand copy is made."""
    return fp32 and cond.dtype is torch.float32


def _saved_like(single, cond, pair, fp32, qk=True, H=H):
    """Shapes / dtypes of the forward's saved activations (the fake implementation and the contract of _fwd). The fp32 path
    saves no copy of cond (_own_cond), no copy of v (QK-norm: the core reads it from qkvg) and no LN(pair) (pair_bias_bwd
    recomputes it): those slots are empty [0, n] tensors."""
    A, _, L, D = single.shape
    M, R, e = A * L, L * L, single.new_empty
    f32, at = torch.float32, (torch.float32 if fp32 else BF)
    return [e((M, D), dtype=f32), e((M, DC), dtype=at), e((0 if _own_cond(cond, fp32) else M, DC), dtype=at), e((M, 2), dtype=f32),
            e((M, 4 * D), dtype=at), e((M, 2 * D), dtype=at), e((M, 2), dtype=f32), e((M, D), dtype=at), e((M, 4 * D), dtype=at),
            e((M, 2 * H), dtype=f32), *((e((M, D), dtype=at), e((M, D), dtype=at), e((0 if fp32 else M, D), dtype=at)) if qk else ()),
            e((H, L, L), dtype=at), e((M, D), dtype=f32), e((A, H, L), dtype=f32), e((M, D), dtype=at), e((M, D), dtype=at),
            e((M, D), dtype=f32), e((M, 2), dtype=f32), e((M, D), dtype=at), e((M, 4 * D), dtype=at), e((M, D), dtype=at),
            e((0 if fp32 else R, DP), dtype=at), e((R, 2), dtype=f32)]


def _fwd_fake(single, cond, pair, mask, params, qk, eq, ek, fp32, heads):
    return [torch.empty_like(single), *_saved_like(single, cond, pair, fp32, qk, heads)]


@opaque(fake=_fwd_fake, name="token_dit_train_sm100_fwd")
def _fwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, params: list[torch.Tensor],
         qk: bool, eq: float, ek: float, fp32: bool, heads: int) -> list[torch.Tensor]:
    """The block's forward: [out, *saved activations] (every output freshly allocated); ``fp32``: fp32 GEMM operands (on TF32
    tensor cores, whatever the caller's allow_tf32) and the TF32 attention kernels, else bf16. Autocast off: the precision is
    ``fp32``'s, and an enclosing bf16 autocast would otherwise turn the fp32 path's GEMM outputs bf16."""
    with _tf32(fp32), torch.autocast("cuda", enabled=False):
        return _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, heads)


def _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, H):
    from miniworld_engine.kernels.augmented_attention import cuda as sm90
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext
    A, _, L, D = single.shape
    DH = D // H
    T = ext(D, DH, rpb=_native_rpb() if _native90(single.device) else 8)
    names = _names_for(qk)
    P = dict(zip(names, params, strict=False))
    M, R, dev = A * L, L * L, single.device
    at = torch.float32 if fp32 else BF
    h100 = _sm90(dev)
    x = torch.empty(M, D, device=dev)                                          # the fp32 residual, written by adaln_a
    c2 = cond.reshape(M, DC).contiguous()
    own_c = _own_cond(cond, fp32)
    chat = torch.empty(M, DC, device=dev, dtype=at); cbf = None if own_c else torch.empty_like(chat)
    cst = torch.empty(M, 2, device=dev)
    pipeline = native and os.environ.get("MINIWORLD_TOKEN_DIT_H100_TRAIN_PACK_ASYNC", "0") == "1"
    W = _pack(P, qk, dev, at) if pipeline else None
    if native:
        T.cond_prep90(cond, chat, cbf, cst, EPS)
    else:
        T.cond_prep(c2, chat, cbf, cst, EPS)
    W = W if pipeline else _pack(P, qk, dev, at)
    if pipeline:
        torch.cuda.current_stream(dev).wait_event(W["_cond_ready"])
    Wn, Wg = W["Wn"], W["Wg"]
    G = torch.mm(chat, Wn.t())                                                  # [M, 4D] s1 | sh1 | s2 | sh2
    Gg = torch.mm(c2 if own_c else cbf, Wg.t())                                 # [M, 2D] gate1 | gate2
    bs1, bs2, bg1, bg2 = W["bs1"], W["bs2"], W["bg1"], W["bg2"]
    xa = torch.empty(M, D, device=dev, dtype=at); xst = torch.empty(M, 2, device=dev)
    T.adaln_a(single.reshape(M, D).contiguous(), G, bs1, xa, xst, EPS, x)  # reads the input, writes x on the way
    Wqkvg, bqkvg = W["Wqkvg"], W["bqkvg"]
    if pipeline:
        torch.cuda.current_stream(dev).wait_event(W["_pack_ready"])
    qkvg = torch.addmm(bqkvg, xa, Wqkvg.t())
    rqk = torch.empty(M, 2 * H, device=dev)
    nq, nk = W["nq"], W["nk"]
    if qk:                                     # fp32: v is not copied, the core reads it as a column view of qkvg
        qn = torch.empty(M, D, device=dev, dtype=at); kn = torch.empty_like(qn); vc = None if fp32 else torch.empty_like(qn)
        T.qknorm(qkvg, nq, nk, qn, kn, vc, rqk, eq, ek, qk)
        vsave = vc if vc is not None else qkvg.new_empty(0, D)
        if vc is None:
            vc = qkvg[:, 2 * D:3 * D]
    else:                                      # the core reads q / k / v as column views of qkvg: no copies
        qn, kn, vc = (qkvg[:, i * D:(i + 1) * D] for i in range(3))
    p2 = pair.reshape(R, DP)
    pst = torch.empty(R, 2, device=dev)
    if fp32:                                   # LN(pair) folded straight into the bias, fp32 FMAs, key mask as -inf; LN(pair) not kept
        ph = pair.new_empty(0, DP, dtype=at)
        bias = torch.empty(H, L, L, device=dev)
        T.pair_bias(p2, W["Wf"], None if mask is None else mask.reshape(L).to(torch.bool).contiguous(), bias, pst, L, EPS)
    else:
        ph = torch.empty(R, DP, device=dev, dtype=at)
        T.pair_ln(p2, ph, pst, EPS)
        bias = torch.mm(W["Wf_at"], ph.t()).view(H, L, L)                       # head-major, natural units; Wf = Wb diag(wp)
        if mask is not None:
            bias.masked_fill_(~mask.reshape(1, 1, L), float("-inf"))
    O, LSE = sm100.forward_tf32(qn, kn, vc, bias, A, L, H, DH) if fp32 else sm100.forward(qn, kn, vc, bias, A, L, H, DH)
    og = torch.empty(M, D, device=dev, dtype=at)
    T.gate_o(O, qkvg, og)
    Wo = W["Wo"]
    y = torch.mm(og, Wo.t())
    x1 = torch.empty_like(x); xt = torch.empty(M, D, device=dev, dtype=at); x1st = torch.empty(M, 2, device=dev)
    T.res_adaln_b(x, y, Gg, bg1, G, bs2, x1, xt, x1st, EPS)
    Wab = W["Wab"]
    h = torch.empty(M, 2 * D, device=dev, dtype=at)
    gsw = _swiglu_gemm(xt, Wab)
    if gsw is not None:                                                         # expand GEMM + SwiGLU, a | b saved on the way
        ab = torch.empty(M, 4 * D, device=dev, dtype=at)
        gsw(xt, Wab, h, ab)
    else:
        ab = torch.mm(xt, Wab.t())                                              # [M, 2 * 2D] a | b
        rows.swiglu_rows(ab, h)
    Wsq = W["Wsq"]
    z = torch.mm(h, Wsq.t())
    out = torch.empty(M, D, device=dev, dtype=single.dtype)
    T.res_c(x1, z, Gg, bg2, out)
    # custom-op outputs may not alias each other: without QK-norm q / k / v are views of qkvg -- not outputs, recovered in _bwd
    qkv_out = [qn, kn, vsave] if qk else []
    cbf = cbf if cbf is not None else c2.new_empty(0, DC, dtype=at)
    return [out.view(single.shape), x, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, *qkv_out, bias, O, LSE, og, y, x1, x1st, xt,
            ab, z, ph, pst]


def _tf32_attn_bwd(T, qs, k, v, dob, bias_p, mask, lse, dd, A, L):
    """The H100 fp32 block's attention backward on the TF32 CUDA core: dQ (for the unscaled q), dK, dV [A L, 768] fp32 and dbias
    [16, L, L] fp32 in natural key order. qs: q in exp2 units; bias_p: the forward's key-permuted bias (log2 units). The
    core takes q, k, dO transposed as well (tiled transposes, ``T.transpose``), and its bias / LSE / D in the permuted orders."""
    from miniworld_engine.kernels.augmented_attention import cuda as sm90
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    perm = rows.key_perm(L, qs.device)
    inv = torch.argsort(perm)

    def tr(t):
        out = t.new_empty(t.shape[1], t.shape[0])
        T.transpose(t, out)
        return out

    bias_tp = bias_p[:, perm][..., inv].transpose(1, 2).contiguous()           # [h, key, q position]
    dq, dk, dv, db_p = sm90.tf32_backward(qs, k, v, dob, tr(qs), tr(k), tr(dob), bias_p, bias_tp, lse, dd,
                                          lse[..., perm].contiguous(), dd[..., perm].contiguous(),
                                          _key_mask(mask, A, L), _key_mask(mask, A, L, perm), A, L)
    return dq, dk, dv, db_p[..., inv]


def _native_attn_bwd90(T, qs, k, v, dob, bias, mask, lse, dd, A, L, fp32):
    from miniworld_engine.kernels.augmented_attention import cuda as sm90
    dq, dk, dv = (torch.empty_like(qs, dtype=torch.float32) for _ in range(3))
    db = torch.empty((A, H, L, L), device=qs.device, dtype=torch.float32)
    km, kmp = T.train_keymask90(mask, qs, A, L, fp32)
    if fp32:
        def tr(t):
            out = t.new_empty((t.shape[1], t.shape[0]))
            T.transpose(t, out)
            return out
        bt, lp, dp = T.train_tf32_bwd_layout90(bias, lse, dd)
        core = sm90._ext("attn_tf32", ("DB_PER_SAMPLE=1",))
        core.attn_tf32_dqb(qs, dob, k, v, tr(k), bias, lse, dd, kmp, dq, db)
        core.attn_tf32_dkv(qs, dob, tr(qs), tr(dob), k, v, bt, lp, dp, km, dk, dv)
    else:
        sm90._ext("attn_dq", () if L % 192 == 0 else ("NWG=2",)).attn_dq(qs, k, v, dob, bias, km, lse, dd, dq)
        sm90._ext("attn_dkv", ("DB_PER_SAMPLE=1",)).attn_dkv(qs, k, v, dob, bias, km, lse, dd, dk, dv, db)
    return dq, dk, dv, T.train_db_reduce90(db, fp32)

def _native_final90(T, names, params, W, allp, dWn, dWf, dWqkvg, dWo, dWg, dWab, dWsq, qk):
    from miniworld_engine.integrations.token_dit import WEIGHTS, QK_WEIGHTS
    canonical = WEIGHTS + (QK_WEIGHTS if qk else ())
    P = dict(zip(names, params, strict=True))
    # The placeholders are not read by entries whose deterministic reduction is fused into the writer.
    src = [dWqkvg[:D], allp, dWqkvg[D:2*D], dWqkvg[2*D:3*D], dWqkvg[3*D:], dWo,
           W["w1"], dWn[:D], allp, dWn[D:2*D], dWg[:D], allp, W["wp"], dWf,
           W["w2"], dWn[2*D:3*D], allp, dWn[3*D:], dWg[D:], allp,
           dWab[:2*D], dWab[2*D:], dWsq]
    if qk:
        src.extend([allp, allp])
    out = T.train_final90([P[n] for n in canonical], src, dWn, W["Wraw"], W["w1"], W["w2"], allp, dWf, W["Wb"], W["wp"],
                          os.environ.get("MINIWORLD_TOKEN_DIT_H100_GRAD_REDUCE", "1") == "1")
    result = dict(zip(canonical, out, strict=True))
    return [result[n] for n in names]


def _bwd_fake(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads):
    return [torch.empty_like(single), torch.empty_like(cond), torch.empty_like(pair), *(torch.empty_like(p) for p in params)]


@opaque(fake=_bwd_fake, name="token_dit_train_sm100_bwd")
def _bwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, params: list[torch.Tensor],
         saved: list[torch.Tensor], dout: torch.Tensor, qk: bool, eq: float, ek: float, fp32: bool, heads: int) -> list[torch.Tensor]:
    """The block's backward: [d single, d cond, d pair, *d params] in the inputs' dtypes (autocast off, as ``_fwd``)."""
    with _tf32(fp32), torch.autocast("cuda", enabled=False):
        return _bwd_body(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads)


def _bwd_body(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, H):
    from miniworld_engine.kernels.augmented_attention import cuda as sm90
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext
    D = single.shape[-1]
    DH = D // H
    T = ext(D, DH, rpb=_native_rpb() if _native90(single.device) else 8)
    h100 = _sm90(single.device)
    if not (qk or h100):                       # the forward's q / k / v: column views of qkvg (see _fwd_body)
        saved = [*saved[:10], *(saved[8][:, i * D:(i + 1) * D] for i in range(3)), *saved[10:]]   # after qkvg (8), rqk (9)
    (x, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, qn, kn, vc, bias, O, LSE, og, y, x1, x1st, xt, ab, z, ph, pst) = saved
    if vc.numel() == 0:                        # fp32 + QK-norm: v as the forward's core read it, a column view of qkvg
        vc = qkvg[:, 2 * D:3 * D]
    names = _names_for(qk)
    at = torch.float32 if fp32 else BF
    W = _pack(dict(zip(names, params, strict=False)), qk, single.device, at)
    if W.get("_pack_ready") is not None:
        torch.cuda.current_stream(single.device).wait_event(W["_pack_ready"])
    Wn, Wraw, Wg, Wqkvg, Wo, Wab, Wsq, Wf = W["Wn"], W["Wraw"], W["Wg"], W["Wqkvg"], W["Wo"], W["Wab"], W["Wsq"], W["Wf"]
    nq, nk, bs1, bs2, bg1, bg2, w1, w2, wp, Wb = (W[k] for k in ("nq", "nk", "bs1", "bs2", "bg1", "bg2", "w1", "w2", "wp", "Wb"))
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    sdt, cdt, pdtype = single.dtype, cond.dtype, pair.dtype
    native = _native90(dev)
    c2, p2 = (None if native else cond.reshape(M, DC).contiguous()), pair.reshape(R, DP)
    dout = dout.reshape(M, D).to(sdt).contiguous()
    # per-block column sums: bg2 bs2 bg1 bs1 bq | dwq dwk (per head); rows: the row kernels' block count (T.part_rows)
    allp = torch.empty(T.part_rows(M, fp32), 7 * D, device=dev)
    part = [allp[:, i * D:(i + 1) * D] for i in range(5)]
    pwqk = allp[:, 5 * D:]
    dG = torch.empty(M, 4 * D, device=dev, dtype=at); dGg = torch.empty(M, 2 * D, device=dev, dtype=at)
    dz = torch.empty(M, D, device=dev, dtype=at)
    T.res_c_bwd(dout, z, Gg, bg2, dz, dGg, part[0])
    dh = torch.mm(dz, Wsq)                                                      # [M, 2D]
    dab = torch.empty_like(ab); h = torch.empty_like(dh)
    T.swiglu_bwd(dh, ab, dab, h)
    dWsq = _mm32(dz.t(), h)
    # fp32: the dxt / dxa GEMMs write straight into their column blocks of dG (the row kernels then copy nothing)
    dxt = torch.mm(dab, Wab, out=dG[:, 3 * D:]) if fp32 else torch.mm(dab, Wab)
    dWab = _mm32(dab.t(), xt)
    dx1 = torch.empty(M, D, device=dev); dy = torch.empty(M, D, device=dev, dtype=at)
    T.res_adaln_b_bwd(dout, dxt, x1, x1st, G, bs2, Gg, bg1, y, dx1, dy, dG, dGg, part[1], part[2])
    dog = torch.mm(dy, Wo)
    dWo = _mm32(dy.t(), og)
    dob = torch.empty(M, D, device=dev, dtype=at); dd = torch.empty(A, H, L, device=dev)
    dqkvg = torch.empty(M, 4 * D, device=dev, dtype=at)
    T.gate_o_bwd(dog, O, qkvg, dob, dd, dqkvg, L)
    # fp32: dQ / dK / dV land in their column blocks of dqkvg; qknorm_bwd then rewrites dq / dk in place (QK-norm) or only sums
    DQ, DK, DV, DB = (sm100.backward_tf32(qn, kn, vc, dob, bias, LSE, dd, A, L, H, DH, dq=dqkvg[:, :D], dk=dqkvg[:, D:2 * D],
                                          dv=dqkvg[:, 2 * D:3 * D]) if fp32
                      else sm100.backward(qn, kn, vc, dob, bias, LSE, dd, A, L, H, DH))
    T.qknorm_bwd(DQ, DK, DV, qkvg, rqk, nq, nk, dqkvg, part[4], pwqk, qk)
    dxa = torch.mm(dqkvg, Wqkvg, out=dG[:, D:2 * D]) if fp32 else torch.mm(dqkvg, Wqkvg)
    dWqkvg = _mm32(dqkvg.t(), xa)
    dx = torch.empty(M, D, device=dev, dtype=sdt)
    T.adaln_a_bwd(dxa, x, xst, G, bs1, dx1, dx, dG, part[3])
    if not native:
        sums = allp.sum(0) if qk else allp[:, :5 * D].sum(0)
        pb = sums[:5 * D].view(5, D)
        dwqk = sums[5 * D:].view(2, H, DH).sum(1).reshape(-1) if qk else torch.zeros(2 * DH, device=dev)
    dchat = torch.mm(dG, Wn)
    dWn = _mm32(dG.t(), chat)
    dcg = torch.mm(dGg, Wg)
    dWg = _mm32(dGg.t(), c2 if cbf.numel() == 0 else cbf)
    dc = torch.empty(M, DC, device=dev, dtype=cdt)
    if native:
        T.cond_bwd90(dchat, dcg, cond, cst, dc)
        dbv = DB.view(H, R)
        dWf = _mm32(dbv, ph)
        dph = _mm32(dbv.t(), W["Wf_at"])
        dpair = torch.empty(R, DP, device=dev, dtype=pdtype)
        T.pair_ln_bwd(dph, p2, pst, dpair)
        outs = _native_final90(T, names, params, W, allp, dWn, dWf, dWqkvg, dWo, dWg, dWab, dWsq, qk)
        return [dx.view(A, 1, L, D), dc.view(A, 1, L, DC), dpair.view(1, L, L, DP), *outs]
    T.cond_bwd(dchat, dcg, c2, cst, dc)
    dWu = torch.empty_like(dWn); dw12 = torch.zeros(2, DC, device=dev)
    Ws1, Wb1, Ws2, Wb2 = (Wraw[i * D:(i + 1) * D] for i in range(4))
    T.unfold_lnw(dWn, Ws1, Wb1, Ws2, Wb2, w1, w2, dWu, dw12)
    dpair = torch.empty(R, DP, device=dev, dtype=pdtype)
    if fp32:                                   # LN(pair) recomputed: dpair and the dWf partials in one pass
        pwf = torch.empty(T.pair_grid(R), H * DP, device=dev)
        T.pair_bias_bwd(DB.view(H, R), p2, pst, Wf, dpair, pwf)
        dWf = pwf.sum(0).view(H, DP)
    else:
        dbv = DB.view(H, R).to(at)                                              # masked key columns are 0 (P = 0)
        dWf = _mm32(dbv, ph)                                                    # [H, DP]
        dph = _mm32(dbv.t(), Wf.to(at))                                         # [R, DP]
        T.pair_ln_bwd(dph, p2, pst, dpair)
    dWs1, dWb1, dWs2, dWb2 = dWu.split(D)
    grads = {
        "attention.ada_ln_in.ln_cond.weight": dw12[0], "attention.ada_ln_in.to_scale.weight": dWs1,
        "attention.ada_ln_in.to_scale.bias": pb[3], "attention.ada_ln_in.to_bias.weight": dWb1,
        "attention.to_scale.weight": dWg[:D], "attention.to_scale.bias": pb[2],
        "attention.to_query.weight": dWqkvg[:D], "attention.to_query.bias": pb[4], "attention.to_key.weight": dWqkvg[D:2 * D],
        "attention.to_value.weight": dWqkvg[2 * D:3 * D], "attention.to_gate.weight": dWqkvg[3 * D:],
        "attention.norm_query.weight": dwqk[:DH], "attention.norm_key.weight": dwqk[DH:],
        "attention.to_out.weight": dWo, "attention.ln_pair.weight": (dWf * Wb).sum(0), "attention.to_bias.weight": dWf * wp,
        "transition.ada_ln_in.ln_cond.weight": dw12[1], "transition.ada_ln_in.to_scale.weight": dWs2,
        "transition.ada_ln_in.to_scale.bias": pb[1], "transition.ada_ln_in.to_bias.weight": dWb2,
        "transition.to_scale.weight": dWg[D:], "transition.to_scale.bias": pb[0],
        "transition.expand_a.weight": dWab[:2 * D], "transition.expand_b.weight": dWab[2 * D:], "transition.squeeze.weight": dWsq,
    }
    # custom-op outputs may not alias each other: the gradients above are views of a few shared buffers, so each leaves as
    # its own tensor in the parameter's dtype -- one multi-tensor copy kernel for all of them (23 copies were ~45 us)
    outs = [torch.empty(p.shape, device=dev, dtype=p.dtype) for p in params]
    torch._foreach_copy_(outs, [grads[n].reshape(p.shape) for n, p in zip(names, params, strict=False)])
    return [dx.view(A, 1, L, D), dc.view(A, 1, L, DC), dpair.view(1, L, L, DP), *outs]


class _Block(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, cond, pair, mask, qk, eq, ek, fp32, heads, *params):
        out, *saved = _fwd(single, cond, pair, mask, list(params), qk, eq, ek, fp32, heads)
        ctx.save_for_backward(single, cond, pair, *params, *saved)
        ctx.meta = (mask, qk, eq, ek, fp32, len(params), heads)
        return out

    @staticmethod
    def backward(ctx, dout):
        mask, qk, eq, ek, fp32, npar, heads = ctx.meta
        vals = ctx.saved_tensors
        single, cond, pair = vals[:3]
        params, saved = list(vals[3:3 + npar]), list(vals[3 + npar:])
        dx, dc, dpair, *pg = _bwd(single, cond, pair, mask, params, saved, dout.contiguous(), qk, eq, ek, fp32, heads)
        if _native90(single.device):
            _activate_native_caller(force=True)
        return (dx, dc, dpair, None, None, None, None, None, None, *(g if ctx.needs_input_grad[9 + i] else None for i, g in enumerate(pg)))


def _pad(single, cond, pair, mask, Ap, Lp):
    """The inputs padded to Ap samples of Lp rows (``block``): zero single / cond rows and samples, zero pair rows and columns,
    the padded keys masked out of every softmax. The padded queries and samples are computed and dropped, so their output
    gradient is zero and they add nothing to any gradient; the masked keys get P = 0, so they add nothing either."""
    F = torch.nn.functional
    A, _, L, _ = single.shape
    rows = (0, 0, 0, Lp - L, 0, 0, 0, Ap - A)
    if mask is None:
        mask = torch.ones(1, L, device=single.device, dtype=torch.bool)
    return (F.pad(single, rows), F.pad(cond, rows), F.pad(pair, (0, 0, 0, Lp - L, 0, Lp - L)),
            F.pad(mask.to(torch.bool), (0, Lp - L), value=False))


def block(module, single, cond, pair, mask=None, compute_dtype=None):
    """One DiTBlock (attention + transition, both residuals) through the fused training path, in the precision
    ``operand_dtype`` picks. Call ``serves`` first. L is padded to a multiple of 128 and A to an even count where needed
    (``_pad``; a padded call returns the slice of the real samples and rows)."""
    a = module.attention
    qk = bool(a.use_qk_norm)
    eq = float(a.norm_query.effective_eps(torch.float32)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(torch.float32)) if qk else 0.0
    fp32 = operand_dtype(single, compute_dtype) is torch.float32
    A, _, L, _ = single.shape
    Ap, Lp = A + A % 2, -(-L // QM) * QM
    padded = (Ap, Lp) != (A, L)
    if padded:
        single, cond, pair, mask = _pad(single, cond, pair, mask, Ap, Lp)
    out = _Block.apply(single, cond, pair.contiguous(), mask, qk, eq, ek, fp32, int(a.n_head),
                       *[module.get_parameter(n) for n in _names_for(qk)])
    return out[:A, :, :L] if padded else out


__all__ = ["block", "serves"]
