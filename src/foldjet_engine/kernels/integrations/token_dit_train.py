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
  * the attention core: B200 ``kernels/augmented_attention/cuda/sm100`` -- bf16: attn_fwd2, attn_dkv + attn_dqb; fp32:
    attn_fwd_tf32, attn_dkv_tf32 + attn_dqb_tf32 (kind::tf32 MMAs, fp32 softmax); H100 the sm_90a cores of
    ``kernels/augmented_attention/cuda`` (bf16 wgmma; fp32 TF32 wgmma), which write og = sigmoid(g) o themselves (no O kept);
  * every elementwise / row step: ``kernels/conditioned_transition/cuda/token_dit_train_rows.cu``; in bf16 the expand GEMM
    carries the SwiGLU on B200 (``gemm_swiglu2_sm100.cu -DSAVE_AB``, which also writes [a | b] for the backward);
  * H100 only: the native path (always on): fused weight pack / final gradient
    kernels, graph-owned cuBLAS workspaces for the qualified buckets, and the caller-gradient hooks (``_native_caller_*``);
  * the residual stream fp32 inside the block, the output in the input's dtype.

``serves()`` is the whole gate: autograd on, the engine's kernels (the block's ``cuda`` backend, through its kernel view), H100 or B200, a
bf16 or fp32 input (bf16 operands when the input or ``compute_dtype`` is bf16, else fp32), the token widths (cond 384 / pair
128, or 256 on H100: Protenix-v2's diffusion pair, the same kernels built at that width / transition n = 2) at a head layout
of LAYOUTS (16 x 48, 24 x 32, 12 x 64 at d 768, 16 x 64 at d 1024; H100 and B200,
bf16 and fp32 alike), B == 1, a key mask of [B, L] or none (``DiTBlock`` hands an [A, B, L] mask shared by the samples
over as its [B, L] row), LayerNorm eps 1e-5. Anything else is refused: the block's ``cuda`` backend
raises ``BackendUnsupported``, never another path.

Any A and L: the kernels tile L by 128 (and the sm_100a forward pairs the samples), so ``block`` pads L to the next multiple of
128 and, on B200, A to an even count -- zero single / cond rows and samples, zero pair rows and columns, the padded keys masked
out -- and returns the first A samples' first L rows. The H100 cores take any A. The padding is computed and dropped; it adds
nothing to any gradient.

The precision follows the tensors, not an enclosing autocast: the forward and backward run with autocast off (bf16-mixed
training hands an fp32 diffusion block fp32 activations, which take the fp32 / TF32 kernels).

The forward and the backward are each one opaque op (``kernels/_compile.opaque``) with a fake implementation, inside an
autograd Function, so torch.compile keeps them as single nodes (as the H100 TriMul training path does).

Unlike the research stack runner (branch ``b200/token-dit``), a DiTBlock call sees one block, so the pair bias and the cond
LayerNorm are computed per block rather than once per stack.
"""

from __future__ import annotations

import os
from typing import cast

import torch

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)
from foldjet_engine.kernels import _capture
from foldjet_engine.kernels._compile import opaque
from foldjet_engine.kernels._device import num_sms
from foldjet_engine.kernels._nvcc import build_failed
from foldjet_engine.kernels._tf32 import tf32_matmul as _tf32

D, DC, DP, H, DH = 768, 384, 128, 16, 48  # the default layout; serves() takes LAYOUTS
H100_DC = 768  # the other conditioning width the H100 training block serves (row kernels built per cond width)
#: (heads, d_single): 16 x 48, 24 x 32, 12 x 64 and 16 x 64 (d 1024), bf16 and fp32 (the TF32 cores are built per layout too)
LAYOUTS = ((16, 768), (24, 768), (12, 768), (16, 1024))
SHARED_COND = True  # a conditioning shared by the samples runs its GEMMs on L rows (``_shared_cond``); False: per sample (the A / B switch of the tests)
PAIR_APB = True  # the H100 bf16 pair bias on the AttentionPairBias tensor-core kernels (``_pair_apb``); False: the TF32 row kernels (the A / B switch of the tests)
QM = 128  # the attention kernels' query / key tile: L is padded to a multiple of it
EPS = 1e-5
BF = torch.bfloat16

ATT = (
    "ada_ln_in.ln_cond.weight",
    "ada_ln_in.to_scale.weight",
    "ada_ln_in.to_scale.bias",
    "ada_ln_in.to_bias.weight",
    "to_scale.weight",
    "to_scale.bias",
    "to_query.weight",
    "to_query.bias",
    "to_key.weight",
    "to_value.weight",
    "to_gate.weight",
    "to_out.weight",
    "ln_pair.weight",
    "to_bias.weight",
)
QKN = ("norm_query.weight", "norm_key.weight")
TRN = (
    "ada_ln_in.ln_cond.weight",
    "ada_ln_in.to_scale.weight",
    "ada_ln_in.to_scale.bias",
    "ada_ln_in.to_bias.weight",
    "to_scale.weight",
    "to_scale.bias",
    "expand_a.weight",
    "expand_b.weight",
    "squeeze.weight",
)


def serves(module, single, cond, pair, mask, compute_dtype=None, extra=None) -> bool:
    if extra is not None and not (
        single.is_cuda
        and _sm90(single.device)
        and extra.ndim == 2
        and tuple(extra.shape) == (single.shape[2], single.shape[2])
        and extra.dtype in (BF, torch.float32)
    ):
        return False  # the [L, L] logit term is folded into the H100 block's bias GEMM output; nowhere else
    if not torch.is_grad_enabled():
        return False
    a = module.attention
    # the engine's kernels (the kernel view sets ``_backend``), as the sm_90 / sm_100 attention cores gate on

    if a._backend != "portable":
        return False
    cap = torch.cuda.get_device_capability(single.device) if single.is_cuda else None
    if cap not in ((9, 0), (10, 0)):
        return False
    if single.dtype not in (BF, torch.float32) or compute_dtype not in (
        None,
        BF,
        torch.float32,
    ):
        return False
    if not any(t.requires_grad for t in (single, cond, pair)) and not any(p.requires_grad for p in module.parameters()):
        return False
    # any A and L >= 1: ``block`` pads L to a multiple of 128 and A to an even count (the kernels' tiles / sample pairs)
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[0] < 1 or single.shape[2] < 1:
        return False
    d = single.shape[-1]
    # the pair: 128 channels, or 256 (Protenix-v2's diffusion pair) / 512 (IntelliFold2's) on H100 -- the pair kernels are built per
    # width (``ext(dp=)``)
    dp = pair.shape[-1] if pair.ndim == 4 else 0
    if dp != a.ln_pair.weight.numel() or not (dp == DP or (dp in (256, 512) and cap in ((9, 0), (10, 0)))):
        return False
    # cond 768 (ESMFold2 / Boltz-2 widths) on H100: the training row kernels take the cond width as a build argument (``rows_ext(dc=)``,
    # prebuilt for 128 / 256 / 384 / 768; the pre-redesign engine ran cond 768 on them through its batched route, PSB_COND)
    dc = cond.shape[-1] if cond.ndim == 4 and (cond.shape[-1] == DC or (cond.shape[-1] == H100_DC and cap in ((9, 0), (10, 0)))) else DC
    if tuple(cond.shape) != (*single.shape[:3], dc) or tuple(pair.shape) != (
        1,
        single.shape[2],
        single.shape[2],
        dp,
    ):
        return False
    if (a.n_head, d) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * d:
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) == (1, single.shape[2])):
        return False
    # the model's options (kernels/integrations/token_dit.options): the defaults everywhere, the others on H100
    from foldjet_engine.kernels.conditioned_transition.cuda import VARIANT_DEFAULT
    from foldjet_engine.kernels.integrations import token_dit

    opts = token_dit.options(module)
    if opts is None or a.ln_pair.eps != EPS:
        return False
    if opts == VARIANT_DEFAULT or cap == (9, 0):
        return True
    # B200, a model's options (the training rows' OPT_* bits, the torch pack). 2026-11-17: the flat q / k LayerNorm (RoseTTAFold3, qknorm
    # mode 2) leaves q unscaled for the B200 cores (they apply 1 / sqrt(head dim) themselves; the H100 exp2 scale had made it RED) and the
    # identity AdaLN in fp32 (Chai-1) reads the fp32 cond itself as its raw conditioning: both served.
    return True


def _f32(t):
    return t.detach().float().contiguous()


def operand_dtype(single, compute_dtype=None) -> torch.dtype:
    """The dtype of every GEMM operand / saved activation: bf16 when the input or ``compute_dtype`` is bf16, else fp32."""
    return BF if (single.dtype is BF or compute_dtype is BF) else torch.float32


def _mm32(a, b, out=None):
    """a @ b with an fp32 result (weight gradients): bf16 operands accumulate into fp32 output, fp32 ones already are.

    fp32 (TF32) weight gradients with few output tiles (dWo 768 x 768, dWg 1536 x 384, dWn 3072 x 384, dWsq 768 x 1536: under
    one wave of 128 x 128 tiles) and a long reduction (the A L rows) ran at 370-580 TF/s against ~740 for the wide ones: they are
    split over the rows into S <= 8 chunks as one batched GEMM, then summed.

    ``out``: an fp32 tensor of the result's shape to write instead (B200: a weight gradient straight into its leaf)."""
    if a.dtype is BF:
        return torch.mm(a, b, out_dtype=torch.float32) if out is None else torch.mm(a, b, out_dtype=torch.float32, out=out)
    n, m = a.shape
    k = b.shape[1]
    tiles = -(-n // 128) * -(-k // 128)
    nsm = num_sms(a.device)
    s = 1
    # (B200, 2026-11-17: only from 4096 rows -- below, the batched GEMM + its sum cost more than the one GEMM: 17 against ~8 us at 1024 rows)
    if tiles < 128 and m >= 4096 and not _sm90(a.device) and b.is_contiguous() and a.t().is_contiguous():
        # (Hopper: the split costs ~440 us a step at A = 48, L768 -- measured dWo / dWg / dWn / dWsq 207 / 284 / 436 / 444 us as
        #  one GEMM against 319 / 321 / 548 / 627 us split in four)
        while s < 8 and tiles * s * 2 <= 2 * nsm and m % (2 * s * 128) == 0:
            s *= 2
    if s == 1:
        return torch.mm(a, b) if out is None else torch.mm(a, b, out=out)
    parts = torch.bmm(a.t().reshape(s, m // s, n).transpose(1, 2), b.reshape(s, m // s, k))
    return parts.sum(0) if out is None else torch.sum(parts, 0, out=out)


_ONES: dict = {}

#: B200: independent stages of a captured training step as parallel branches of the graph (FOLDJET_ENGINE_TOKEN_B200_STREAMS=0: one chain)
_STREAMS = os.environ.get("FOLDJET_ENGINE_TOKEN_B200_STREAMS", "1") != "0"
#: B200: weight gradients straight into their leaves, the rest of the final gradients beside the chain (``_final_b200``; 0: one final kernel)
_FINAL_B200 = os.environ.get("FOLDJET_ENGINE_TOKEN_B200_FINAL", "1") != "0"
#: B200 bf16, one sample (2026-11-17, part 7): attn_dkv reads the bias as stored (``sm100.backward(bias_nt=True)``, -DBIASNT=1) -- no
#: bias_transpose launch on the backward chain (the same values). FOLDJET_ENGINE_TOKEN_B200_BIASNT=0: the transpose as before; 1: one
#: sample only (part 7); 2 (part 10, default): every sample count -- the bias is shared by the samples, attn_dkv's tile does not depend on one
_BIASNT_B200 = int(os.environ.get("FOLDJET_ENGINE_TOKEN_B200_BIASNT", "2"))
#: B200 training (part 10): both to_scale weight gradients on side stream 0 right after dWo (dGg is complete there) instead of behind
#: the four q | k | v | g weight gradients at the end of side stream 2, the step's last branch (FOLDJET_ENGINE_TOKEN_B200_GEARLY=0: as before)
_GEARLY_B200 = os.environ.get("FOLDJET_ENGINE_TOKEN_B200_GEARLY", "1") != "0"
#: B200 bf16 training (part 7): attn_dqb on side stream 2 beside attn_dkv (both read only their inputs; dQ / dbias joined before
#: qknorm_bwd / the pair backward). The two grids are heads x (L / 128)^2 and A heads L / 128 work items: 16 + 16 CTAs at L 128.
_DQB_SIDE_B200 = os.environ.get("FOLDJET_ENGINE_TOKEN_B200_DQB_SIDE", "1") != "0"


class _Side:
    """Side streams of one forward / backward call on a B200 inside a CUDA-graph capture (``bias_only_dit.cuda.side``: eager calls and
    the other cards never fork -- ``run`` then calls in place). A captured step on one stream is a chain; the stages that read nothing
    of the main chain run as parallel branches: forward, the gate GEMM and the pair pass (joined before the core); backward, the six
    weight-gradient GEMMs (leaves: only the final gradients read them) and the pair backward (it needs dbias only). One-sample steps
    are launch-bound (~75 kernels of 2-8 us on a few SMs each), so a branch runs beside the chain, not behind it (2026-11-17).

    The rules of ``side``: ``run`` orders the side stream after everything launched so far on the current stream; every tensor a side
    kernel reads or writes stays referenced until ``join`` (the callers' locals), and the main chain reads no side result before it."""

    def __init__(self, device, n: int = 2):
        from foldjet_engine.kernels.bias_only_dit.cuda import side

        self.device = device
        self.streams = [side.fork(device) if _STREAMS else None for _ in range(n)]

    def run(self, k: int, fn):
        s = self.streams[k]
        if s is None:
            return fn()
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            return fn()

    def after(self, k: int, j: int) -> None:
        """Side stream ``k`` waits for everything launched on side stream ``j`` so far (both forked; no-op otherwise)."""
        sk, sj = self.streams[k], self.streams[j]
        if sk is not None and sj is not None:
            sk.wait_stream(sj)

    def wait(self, k: int) -> None:
        """The current stream waits for everything launched on side stream ``k`` so far (no-op when not forked)."""
        sk = self.streams[k]
        if sk is not None:
            torch.cuda.current_stream(self.device).wait_stream(sk)

    def join(self) -> None:
        for s in self.streams:
            if s is not None:
                torch.cuda.current_stream(self.device).wait_stream(s)


def _colsum(t):
    """Column sums of a tall fp32 [n, k] (the row kernels' per-block partial rows) as one cuBLAS gemv with a cached ones vector: B200
    (2026-11-17). ``t.sum(0)`` is a strided reduction per column there: 20 us for [888, 1536], 13 us for [256, 3840]; gemv ~3 us (no
    tensor cores: full fp32, deterministic)."""
    key = (t.shape[0], t.device)
    o = _ONES.get(key)
    if o is None:
        o = _ONES[key] = torch.ones(t.shape[0], device=t.device)
    return torch.mv(t.t(), o)


LOG2E = 1.4426950408889634
QS90 = LOG2E / DH**0.5  # the sm_90a core's q scale: its logits are in log2 units (16 x 48 only)


def _sm90(dev) -> bool:
    """H100: the sm_90a attention cores and their operand conventions; B200 otherwise (``serves`` admits only these two)."""
    return torch.device(dev).type == "cuda" and torch.cuda.get_device_capability(dev) == (9, 0)


def _key_mask(mask, A, L, perm=None):
    """The sm_90a cores' additive key mask, [A, L] fp32 (0 = attend), or None; ``perm``: its keys in that order (the TF32
    core's)."""
    if mask is None:
        return None
    m = mask.reshape(L) if perm is None else mask.reshape(L)[perm]
    return torch.where(m, 0.0, -1e30).to(torch.float32).expand(A, L).contiguous()


_PACKS: dict = _capture.weight_cache("token_dit_train.packs")


def _native_rpb():
    return 2


def _native90(dev):
    return _sm90(dev)


def _nrows(dev):
    """The fused weight pack, conditioning prep / backward and final-gradient kernels of the native path (generic CUDA in
    ``token_dit_train_rows.cu``): H100, and B200 since 2026-11-17 (its torch pack and per-leaf gradient copies were ~60 small kernels
    of a training step). The attention-side native pieces (``train_bias90``, ``_native_attn_bwd90``, owned GEMMs) stay H100 only."""
    return torch.device(dev).type == "cuda" and torch.cuda.get_device_capability(dev) in ((9, 0), (10, 0))


def _empty_uniform90(dev, A=None, L=None):
    return _native90(dev) and A == 5 and L in (384, 768)


def _owned_gemms90(single):
    """GEMMs whose captured scratch belongs to this graph, never a framework handle."""
    if not (_native90(single.device) and single.shape[0] == 5 and single.shape[2] in (384, 768)):
        return torch.mm, _mm32, torch.addmm
    from foldjet_engine.kernels.conditioned_transition.cuda.train import (
        owned_gemm_ext,
    )

    backend = owned_gemm_ext()
    tf32 = torch.backends.cuda.matmul.allow_tf32

    def mm(a, b):
        return backend.owned_mm(a, b, tf32, False)

    def mm32(a, b):
        return backend.owned_mm(a, b, tf32, True)

    def addmm(bias, a, b):
        return backend.owned_addmm(bias, a, b, tf32)

    return mm, mm32, addmm


def _fp8_train(W, M, D, fp32, plain, h100, names):
    """The EXPERIMENTAL FP8 training placements (``conditioned_transition.cuda.token_dit_fp8_train``; default off) of this call:
    (the module, the placements in force, their e4m3 weight operands). Off: (module, empty, {}) and the bf16 route unchanged (the
    module's import builds nothing)."""
    from foldjet_engine.kernels.conditioned_transition.cuda import token_dit_fp8_train as f8

    places = f8.active(M, D, fp32, plain, h100)
    if not places:
        return f8, frozenset(), {}
    return f8, places, f8.weights(W, places & frozenset(names))


_NATIVE_CALLER_TASKS: set[int] = set()


def _activate_native_caller(*, force: bool = False):
    """Limit installed hooks to backward tasks that actually execute our block."""
    if not force or torch.compiler.is_compiling():
        return
    task = torch._C._current_graph_task_id()
    if task >= 0 and task not in _NATIVE_CALLER_TASKS:
        _NATIVE_CALLER_TASKS.add(task)
        torch.autograd.Variable._execution_engine.queue_callback(lambda: _NATIVE_CALLER_TASKS.discard(task))


def _native_caller_output(output, *tensors, force: bool = False):
    """Bind accumulation after the native output node finishes backward."""
    if torch.compiler.is_compiling() or not torch.is_grad_enabled() or not output.requires_grad or output.grad_fn is None:
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
    """Keep the caller's autograd edges while replacing the CUDA sum of a sample-expanded input's gradient.

    A caller tensor expanded over the samples (stride 0: a shared conditioning / pair) has an ``ExpandBackward0`` / ``AsStridedBackward0``
    node whose sum over the samples runs on ``caller_sum90`` (a fixed-order fp32 sum per element). Hooks bracket that one node; no mode
    escapes to other nodes or the caller, and gradients remain ordinary Tensors. Not a torch.compile transformation.

    The leaves' ``AccumulateGrad`` nodes are NOT hooked: their ``grad += new`` is ATen's ``add_`` in the autograd engine, no Python. (They
    were: a pre- and a post-hook, a ``TorchDispatchMode`` entered and left, and a Python ``__torch_dispatch__`` per accumulation -- one per
    parameter per backward, ~2,800 per FoldJet training micro step of a 24-block token DiT -- to run ``caller_add90``, which computes
    exactly what ``add_`` does: T(float(a) + float(b)) per element. The gradients are bitwise the same without it.)
    """
    if not force or torch.compiler.is_compiling():
        return
    # a leaf (every parameter) has an AccumulateGrad node: nothing to hook; only a sample-expanded view has the sum
    views = [t for t in tensors if t.requires_grad and not t.is_leaf and t.dim() and t.stride(0) == 0]
    if not views:
        return  # the usual call: no class, no extension lookup per block call
    import threading

    from torch.autograd.graph import get_gradient_edge
    from torch.utils._python_dispatch import TorchDispatchMode

    from foldjet_engine.kernels.conditioned_transition.cuda.train import ext

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
                if (
                    tuple(dims or ()) == (0,)
                    and x.is_cuda
                    and x.ndim >= 1
                    and x.is_contiguous()
                    and x.dtype in (BF, torch.float32)
                    and kwargs.get("dtype") in (None, x.dtype)
                ):
                    out = kernels.caller_sum90(x)
                    return out if keep else out.squeeze(0)
            elif func is torch.ops.aten.add_.Tensor:
                dst, src = args[:2]
                if (
                    dst.is_cuda
                    and src.device == dst.device
                    and src.dtype == dst.dtype
                    and dst.dtype in (BF, torch.float32)
                    and dst.shape == src.shape
                    and kwargs.get("alpha", 1) == 1
                ):
                    # Keep ATen's overlap checks for partially aliased views.
                    # Exact self-add is safe; disjoint allocations are usual
                    # for AccumulateGrad's destination and incoming gradient.
                    overlap = torch._C._overlaps(dst, src)
                    if not overlap or (dst.data_ptr() == src.data_ptr() and dst.stride() == src.stride()):
                        if dst.is_contiguous() and src.is_contiguous():
                            kernels.caller_add90(dst, src)
                            return dst
                        if dst.ndim == 2 and src.ndim == 2 and dst.t().is_contiguous() and src.t().is_contiguous():
                            # Column-major matrix gradients need only metadata
                            # views; both operands retain identical element order.
                            kernels.caller_add90(dst.t(), src.t())
                            return dst
            elif func in (
                torch.ops.aten.zeros.default,
                torch.ops.aten.new_zeros.default,
            ):
                out = torch.empty(args[0], **kwargs) if func is torch.ops.aten.zeros.default else args[0].new_empty(args[1], **kwargs)
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
            state.mode = CallerMode() if torch._C._current_graph_task_id() in _NATIVE_CALLER_TASKS else None
            if state.mode is not None:
                state.mode.__enter__()

        def post(grad_inputs, grad_outputs):
            if state.mode is not None:
                state.mode.__exit__(None, None, None)
            del state.mode

        node.register_prehook(pre)
        node.register_hook(post)
        node.metadata["miniworld_h100_native_caller"] = True

    for tensor in views:
        node = get_gradient_edge(tensor).node
        if node.name() in ("ExpandBackward0", "AsStridedBackward0"):
            install(node)


def _pack(P, qk, dev, at, opts=None):
    """The block's weights in the kernels' layouts (GEMM operands in ``at``, folded cond / pair LayerNorm weights, fp32 bias
    vectors). Rebuilt only when a parameter changes -- an optimizer step bumps ``_version`` -- so a training step pays
    it once per block, not once per forward. Scoped to the CUDA-graph capture (``kernels._capture``): an eager pack is never
    reused inside a capture, whose replays would otherwise run on the weights of capture time after every optimizer step.

    ``opts``: a model's options (None: the defaults), packed by the torch branch from the block's leaves (``_leaves``): no gating
    query -> the gate rows' bias +1e4 (sigmoid = 1, its gradient 0); the up-gate transition -> Wab = [W_a; W_b; W_u]; the flat q / k
    LayerNorm -> nq / nk / nqb / nkb its [D] affine."""
    # (B200, 2026-11-17: a model's options too -- the canonical leaves through the native pack, then ``_pack_options``; the torch pack was
    # ~25 small kernels, ~60 us of a Boltz-2 / RoseTTAFold3 / Chai-1 training step)
    native = _nrows(dev) and (opts is None or not _sm90(dev))
    key = (native, opts, *((t.data_ptr(), t._version) for t in P.values()))
    slot = _capture.scoped((next(iter(P.values())).data_ptr(), at))
    hit = None if slot is None else _PACKS.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    if native:
        from foldjet_engine.kernels.conditioned_transition.cuda.train import ext
        from foldjet_engine.kernels.integrations.token_dit import QK_WEIGHTS, WEIGHTS

        names = WEIGHTS + (QK_WEIGHTS if qk else ())
        keys = (
            "w1",
            "w2",
            "Wraw",
            "Wn",
            "Wg",
            "bs1",
            "bs2",
            "bg1",
            "bg2",
            "Wqkvg",
            "bqkvg",
            "nq",
            "nk",
            "wp",
            "Wb",
            "Wf",
            "Wf_at",
            "Wf_l2",
            "Wo",
            "Wab",
            "Wsq",
        )
        d_ = P["attention.to_out.weight"].shape[0]
        T = ext(
            d_,
            d_ // P["attention.to_bias.weight"].shape[0],
            rpb=_native_rpb(),
            dc=P["attention.ada_ln_in.to_scale.weight"].shape[1],
            n=P["transition.expand_a.weight"].shape[0] // d_,
            dp=P["attention.ln_pair.weight"].shape[0],
        )
        weights = [P[n] for n in names]
        W = dict(zip(keys, T.train_pack90(weights, qk, at is torch.float32), strict=True))
        if opts is not None:
            _pack_options(W, P, opts, at, d_)
        if slot is not None:
            _capture.prune(_PACKS)
            _PACKS[slot] = (key, W, tuple(P.values()))  # keeps the key tensors alive
        return W
    g = lambda n: P[n]
    D = g("attention.to_out.weight").shape[0]
    DH = D // g("attention.to_bias.weight").shape[0]
    w1, w2 = (
        _f32(g("attention.ada_ln_in.ln_cond.weight")),
        _f32(g("transition.ada_ln_in.ln_cond.weight")),
    )
    Wraw = torch.cat(
        [
            _f32(g("attention.ada_ln_in.to_scale.weight")),
            _f32(g("attention.ada_ln_in.to_bias.weight")),
            _f32(g("transition.ada_ln_in.to_scale.weight")),
            _f32(g("transition.ada_ln_in.to_bias.weight")),
        ]
    )
    wp, Wb = _f32(g("attention.ln_pair.weight")), _f32(g("attention.to_bias.weight"))
    Wf = Wb * wp
    W: dict[str, torch.Tensor] = {
        "w1": w1,
        "w2": w2,
        "Wraw": Wraw,
        "Wn": torch.cat([Wraw[: 2 * D] * w1, Wraw[2 * D :] * w2]).to(at),  # cond-LN weights folded
        "Wg": torch.cat([g("attention.to_scale.weight"), g("transition.to_scale.weight")]).detach().to(at),
        "bs1": _f32(g("attention.ada_ln_in.to_scale.bias")),
        "bs2": _f32(g("transition.ada_ln_in.to_scale.bias")),
        "bg1": _f32(g("attention.to_scale.bias")),
        "bg2": _f32(g("transition.to_scale.bias")),
        "Wqkvg": torch.cat(
            [
                g("attention.to_query.weight"),
                g("attention.to_key.weight"),
                g("attention.to_value.weight"),
                g("attention.to_gate.weight"),
            ]
        )
        .detach()
        .to(at),
        "bqkvg": torch.cat(
            [
                g("attention.to_query.bias").detach().float(),
                torch.zeros(3 * D, device=dev),
            ]
        ).to(at),
        "nq": _f32(g("attention.norm_query.weight")) if qk else torch.ones(DH, device=dev),
        "nk": _f32(g("attention.norm_key.weight")) if qk else torch.ones(DH, device=dev),
        "wp": wp,
        "Wb": Wb,
        "Wf": Wf,
        "Wf_at": Wf.to(at),
        "Wf_l2": (Wf * LOG2E).to(at),  # natural / log2 units
        "Wo": g("attention.to_out.weight").detach().to(at).contiguous(),
        "Wab": torch.cat([g("transition.expand_a.weight"), g("transition.expand_b.weight")]).detach().to(at),
        "Wsq": g("transition.squeeze.weight").detach().to(at).contiguous(),
    }
    if opts is not None:
        _pack_options(W, P, opts, at, D)
    if slot is not None:
        _capture.prune(_PACKS)
        _PACKS[slot] = (key, W, tuple(P.values()))  # keeps the key tensors alive
    return W


def _pack_options(W, P, opts, at, D):
    """A model's options on a pack of the canonical leaves (in place): no gating query -> the gate logits' bias +1e4 (the gate weight leaf
    is zero); the up-gate transition -> Wab = [W_a; W_b; W_u]; the flat q / k LayerNorm -> its [D] affine."""
    if not opts[3]:
        W["bqkvg"][3 * D :] = 1.0e4
    if opts[4]:
        W["Wab"] = torch.cat([W["Wab"], P["up"].detach().to(at)]).contiguous()
    if opts[5]:
        W["nq"], W["nqb"], W["nk"], W["nkb"] = (_f32(P[k]) for k in ("kq.wq", "kq.bq", "kq.wk", "kq.bk"))


_GSW: dict = {}


def _swiglu_gemm(xt, wab):
    """The sm_100a expand GEMM with the SwiGLU epilogue that also writes [a | b] (``gemm_swiglu2_sm100.cu -DSAVE_AB``) where
    it applies (bf16, B200, M >= its row threshold); None keeps cuBLAS + the SwiGLU rows."""
    from foldjet_engine.kernels.conditioned_transition.cuda import gemm_swiglu

    if not gemm_swiglu.supported(xt, wab):
        return None
    idx = xt.device.index if xt.device.index is not None else torch.cuda.current_device()
    if idx not in _GSW:
        try:
            _GSW[idx] = gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2, save_ab=True)
        except Exception as exc:  # a build / load failure is an error, never a fallback
            build_failed("sm100 SwiGLU GEMM", exc)
            _GSW[idx] = None
    return _GSW[idx]


#: B200 bf16 training up to this many rows (round 3 part 6): dchat = dG Wn ([M, 4D] x [4D, cond]) as ONE split-K cluster launch
#: (``tdt_epi.mm``, K split 8, the partial tiles summed in rank order over distributed shared memory) -- cuBLAS split K into a
#: separate reduce kernel, both on the step's tail after the last row pass. Measured r101 L 128 203.9 -> 198.0 us, r144 L 256 214.2 ->
#: 207.0, r261 L 128 186.6 -> 174-185; at 1024 rows (r286 S 8) 6 % slower, so not there. 0: cuBLAS
_DC_B200_ROWS = int(os.environ.get("FOLDJET_ENGINE_TOKEN_B200_DC_ROWS", "256"))


#: The option leaves after the canonical ones (``_names_for`` with a model's options): the up-gate weight (Boltz-2) and the flat q / k
#: LayerNorm's affine (RoseTTAFold3), and the module parameter each one is.
UP_LEAF = ("up", "transition.up.weight")
KQ_LEAVES = (
    ("kq.wq", "attention.norm_query.weight"),
    ("kq.bq", "attention.norm_query.bias"),
    ("kq.wk", "attention.norm_key.weight"),
    ("kq.bk", "attention.norm_key.bias"),
)


def _dc_epi(h100, fp32, dG, Wn) -> bool:
    """B200 bf16, at most ``_DC_B200_ROWS`` rows: dchat by ``tdt_epi.mm`` (its ``supported`` and the weight's TMA layout)."""
    if h100 or fp32 or dG.shape[0] > _DC_B200_ROWS or Wn.dtype is not BF or not Wn.is_contiguous() or Wn.data_ptr() % 16:
        return False
    from foldjet_engine.kernels.conditioned_transition.cuda import tdt_epi

    return tdt_epi.supported("dc", dG) and Wn.shape[1] % 64 == 0


def _names_for(qk, opts=None):
    """The block's leaf names: the canonical ATT (+ QKN) + TRN, then the option leaves of ``opts`` (``token_dit.options``)."""
    names = ["attention." + n for n in ATT + (QKN if qk else ())] + ["transition." + n for n in TRN]
    if opts is not None and opts[4]:
        names.append(UP_LEAF[0])
    if opts is not None and opts[5]:
        names += [k for k, _ in KQ_LEAVES]
    return names


def _leaves(module, names):
    """The tensors of ``names``: the module's parameters, the constant leaves of a part the model does not have
    (``token_dit._leaf``), the option leaves."""
    from foldjet_engine.kernels.integrations import token_dit

    extra = dict([UP_LEAF, *KQ_LEAVES])
    return [module.get_parameter(extra[n]) if n in extra else token_dit._leaf(module, n) for n in names]


def _opt_bits(opts):
    """(the training rows' OPT_* bits, the qknorm mode) of a model's options: OPT_IDENT 1, OPT_PAR 2, OPT_UP 4; qknorm 2 = the flat
    q / k LayerNorm."""
    bits = (0 if opts[1] else 1) | (2 if opts[0] else 0) | (4 if opts[4] else 0)
    return bits, (2 if opts[5] else 0)


def _shared_cond(single, cond) -> bool:
    """The conditioning is shared by the samples: ``block`` hands the H100 default-options block ONE [1, 1, L, dc] conditioning for A > 1
    samples (the caller's stride-0 expand, sliced). Every conditioning GEMM -- G, Gg forward; dchat, dcg and the dWn, dWg weight
    gradients backward -- then runs on L rows instead of A L, the row kernels read row r % L of G / Gg, and the backward sums dG / dGg
    over the samples first (``sample_sum2``: fp32, sample order, deterministic)."""
    return single.shape[0] > 1 and cond.shape[0] == 1


def _own_cond(cond, fp32) -> bool:
    """fp32 path with an fp32 cond: the gate GEMM (and its weight gradient) reads cond itself, no operand copy is made. Not on
    the H100 native path, whose cond may be sample-expanded (stride 0): it reads cond through ``cond_prep90`` and keeps cbf."""
    return fp32 and cond.dtype is torch.float32 and not _nrows(cond.device)


def _nd(params, qk, D):
    """(transition expansion n, pair channels of ln_pair) read off the block's parameters: expand_a is [n D, D], ln_pair [dp]."""
    P = dict(zip(_names_for(qk), params, strict=False))
    return P["transition.expand_a.weight"].shape[0] // D, P["attention.ln_pair.weight"].shape[0]


def _saved_like(single, cond, pair, fp32, qk=True, H=H, n=2, dp=DP, up=False, nox=False, hsave=False, rx1=False, kq=False):
    """Shapes / dtypes of the forward's saved activations (the fake implementation and the contract of _fwd). The fp32 path
    saves no copy of cond (_own_cond), no copy of v (QK-norm: the core reads it from qkvg) and no LN(pair) (pair_bias_bwd
    recomputes it): those slots are empty [0, n] tensors. H100: q / k / v are always copies (the q pass scales q; the TF32 core's
    v^T is built from v) and no O slot (the cores write og), and fp32 keeps LN(pair) (the dWf / dpair GEMMs read it)."""
    A, _, L, D = single.shape
    M, R, e = A * L, L * L, single.new_empty
    Mc = L if _shared_cond(single, cond) else M  # rows of the conditioning products
    f32, at = torch.float32, (torch.float32 if fp32 else BF)
    h100 = _sm90(single.device)
    copies = qk or kq or h100  # kq: the flat q / k LayerNorm (its qn / kn are copies too)
    apb = _pair_apb(h100, fp32, dp, H, pair.dtype)
    return [
        e((0 if nox else M, D), dtype=f32),  # the fp32 residual (nox: not kept, the input is read instead)
        e((Mc, cond.shape[-1]), dtype=at),
        e((0 if _own_cond(cond, fp32) else Mc, cond.shape[-1]), dtype=at),
        e((Mc, 2), dtype=f32),
        e((Mc, 4 * D), dtype=at),
        e((Mc, 2 * D), dtype=at),
        e((M, 2), dtype=f32),
        e((M, D), dtype=at),
        e((M, 4 * D), dtype=at),
        e((M, 2 * H), dtype=f32),
        *(
            (
                e((M, D), dtype=at),
                e((M, D), dtype=at),
                e((0 if (fp32 and not h100) else M, D), dtype=at),
            )
            if copies
            else ()
        ),
        e((H, L, L), dtype=at),
        *([] if h100 else [e((M, D), dtype=f32)]),
        e((A, H, L), dtype=f32),
        e((M, D), dtype=at),
        e((M, D), dtype=at),
        e((0 if rx1 else M, D), dtype=f32),  # x1 (rx1: not kept, recomputed from the input)
        e((M, 2), dtype=f32),
        e((M, D), dtype=at),
        e((M, (3 if up else 2) * n * D), dtype=at),  # a | b of the SwiGLU: two [M, n D] halves (a | b | u up-gated)
        e((M, D), dtype=at),
        e((0 if ((fp32 and not h100) or _pair_fused(h100, fp32, dp)) else R, dp), dtype=at),
        e((0 if (apb and dp == DP) else R, 2), dtype=f32),  # (mean, rstd) of the pair rows (the 128-wide APB backward recomputes them)
        e((H * R if (apb and dp != DP) else 0,), dtype=f32),  # the wide APB forward's unmasked fp32 bias (its backward's S2)
        e(
            (M, n * D) if hsave else (0, n * D), dtype=at
        ),  # hsave: h = SwiGLU(a, b) as the squeeze GEMM read it (the backward does not rewrite it)
    ]


def _pair_apb(h100, fp32, dp, heads, pdt) -> bool:
    """The H100 bf16 block's pair bias on the AttentionPairBias tensor-core kernels (``augmented_attention.cuda.apb``): one pass over the pair
    each way at the HBM rate -- the forward is Wf . LN(pair) on bf16 mma with the row statistics from the tensor cores (no per-element float
    work), the backward gives dpair and the per-block dWf partials in one more pass; a pair width wider than 128 (256: Protenix-v2 / Chai-1 /
    ESMFold2, 512: IntelliFold2; 16 heads) saves its row statistics and its fp32 unmasked bias for the backward (``pair_bias_fwd_train``).
    The TF32 row kernels of ``_pair_fused`` serve the other head counts. Needs a bf16 pair (the kernels read bf16 words)."""
    return bool(
        PAIR_APB and h100 and not fp32 and pdt is BF and ((dp == 128 and heads in (8, 12, 16, 24)) or (dp in (256, 512) and heads == 16))
    )


def _pair_fused(h100, fp32, dp) -> bool:
    """The H100 bf16 block builds its bias with the one-pass pair kernel (LN(pair) Wf^T on TF32 tensor cores, the extra term added in the same
    pass, no LN(pair) written) and differentiates it with its one-pass backward (dpair and the dWf partials): it keeps no LN(pair) either.
    Every pair width it serves (128; 256, Protenix-v2's diffusion pair; 512, IntelliFold2's): the kernels are built per width. B200 bf16
    since 2026-11-17 (natural units, the key mask as -inf in the same pass): it ran the LN(pair) rows, a bias GEMM and a mask fill forward,
    two GEMMs and the LN backward rows on the way back (r144 master L 512: 88 + 102 us of 414)."""
    return not fp32 and dp in (128, 256, 512)


def _pair_bwd_apb(h100, fp32, DB, p2, wf_at, dpair, H, L):
    """B200 bf16, pair 128 (2026-11-17): AttentionPairBias's one-pass pair backward (``apb`` ``pair_bias_bwd``: TMA-fed tiles, bf16 tensor
    cores, LN(pair) recomputed) -- 46 against 80 us at L 512 for the training rows' ``pair_bias_bwd`` (per-row TF32 tiles). dpair is
    written into ``dpair``; returns dWf [H, 128] (the blocks' partial rows added in block order: bitwise repeatable), or None where
    it does not serve (the caller's pass then)."""
    if h100 or fp32 or p2.shape[1] != 128 or p2.dtype is not torch.bfloat16 or not p2.is_contiguous() or L % 16:
        return None
    from foldjet_engine.kernels.augmented_attention.cuda import apb

    if H not in apb.PAIR_WIDTHS[128] or (H, 384) not in apb.GEOMETRY:
        return None
    E = apb.ext()
    o = 2 * 384 + apb.width(H, 384)  # dWf's offset in a partial row (apb.acc_n)
    acc = torch.empty(E.pair_bwd_blocks(p2, H, L), apb.acc_n(H, 384, 128), device=p2.device)
    E.pair_bias_bwd(p2, DB.reshape(H, L, L).contiguous(), wf_at, dpair, acc, L, EPS, 384)
    return _colsum(acc[:, o : o + H * 128]).view(H, 128)


def _pair_fwd(T, W, pair, p2, pst, mask, extra, mm, h100, fp32, at, H, L, dpw):
    """The pair -> bias pass of the forward: (LN(pair) where it is kept, the head-major bias, pst, bu); fills ``pst`` with the row (mean, rstd).
    H100 bf16 (``_pair_apb``): the AttentionPairBias tensor-core pass (bf16 bias in log2 units, no key mask: ``train_bias90`` applies it); a pair
    wider than 128 returns its own (mean, rstd) and the unmasked fp32 bias ``bu`` for the backward (else ``bu`` is empty)."""
    R = p2.shape[0]
    bu = p2.new_empty(0, dtype=torch.float32)
    if _pair_apb(h100, fp32, dpw, H, pair.dtype):
        from foldjet_engine.kernels.augmented_attention.cuda import apb

        ph = pair.new_empty(0, dpw, dtype=at)
        if dpw == apb.DP:
            bias = cast(torch.Tensor, apb.pair_bias(p2, W["Wf_l2"], None, L, EPS, 0.0))  # the backward recomputes the row statistics
        else:  # + (mean, rstd) and the unmasked fp32 bias
            bias, pst, bu_ = cast(tuple, apb.pair_bias(p2, W["Wf_l2"], None, L, EPS, 0.0, saved=True))
            bu = bu_.view(-1)
        if extra is not None:  # an [L, L] logit term for every head, in the bias's log2 units
            bias.add_(extra.view(1, L, L), alpha=LOG2E)
        return ph, bias, pst, bu
    if _pair_fused(h100, fp32, dpw):
        ph = pair.new_empty(0, dpw, dtype=at)
        bias = torch.empty(H, L, L, device=p2.device, dtype=at)
        from foldjet_engine.kernels.augmented_attention.cuda import apb

        if not h100 and extra is None and p2.is_contiguous() and H in apb.PAIR_WIDTHS.get(dpw, ()):
            # B200 bf16 (2026-11-17): AttentionPairBias's one-pass forward with the row stats (bulk-loaded tiles, the bf16 folded weight):
            # 16 pair rows / ns against 4.6 for ``pair_bias`` below (57 us at L 512); natural units, masked keys -inf, (mean, rstd) for
            # ``pair_bias_bwd``
            apb.ext().pair_bias_fwd_stats(
                p2, W["Wf_at"], None if mask is None else mask.reshape(L).to(torch.bool).contiguous(), bias, pst, L, EPS, float("-inf")
            )
        else:
            T.pair_bias(
                p2,
                (W["Wf"].float() * LOG2E).contiguous() if h100 else W["Wf"].float().contiguous(),
                None if h100 or mask is None else mask.reshape(L).to(torch.bool).contiguous(),
                bias,
                pst,
                L,
                EPS,
                None if extra is None else extra.reshape(-1).float().contiguous(),
                LOG2E if h100 else 1.0,
            )  # H100: log2 units; B200: natural units, masked keys -inf; rounded once to bf16
    elif h100 or not fp32:
        ph = torch.empty(R, dpw, device=p2.device, dtype=at)
        T.pair_ln(p2, ph, pst, EPS)
        # head-major bias, Wf = Wb diag(wp): natural units for sm_100a, log2 units (and key-permuted for the TF32 core) for sm_90a
        bias = mm(W["Wf_l2"] if h100 else W["Wf_at"], ph.t()).view(H, L, L)
        if extra is not None:  # an [L, L] logit term for every head, in the bias's log2 units (the H100 block only): one add
            bias.add_(extra.view(1, L, L), alpha=LOG2E)
    else:  # B200 fp32: LN(pair) folded straight into the bias, fp32 FMAs, key mask as -inf; LN(pair) not kept
        ph = pair.new_empty(0, dpw, dtype=at)
        bias = torch.empty(H, L, L, device=p2.device)
        from foldjet_engine.kernels.augmented_attention.cuda import apb

        mk = None if mask is None else mask.reshape(L).to(torch.bool).contiguous()
        if L >= 256 and p2.is_contiguous() and p2.data_ptr() % 16 == 0 and H in apb.PAIR_WIDTHS.get(dpw, ()):
            # from L 256: AttentionPairBias's TMA-fed TF32 pass with the row stats (``pair_bias32_tc``: 86 against 131 us at L 768)
            apb.ext().pair_bias32_tc(p2, W["Wf"], mk, bias, pst, L, EPS, float("-inf"), False)
        else:
            T.pair_bias(p2, W["Wf"], mk, bias, pst, L, EPS)
    return ph, bias, pst, bu


def _fwd_fake(single, cond, pair, mask, params, qk, eq, ek, fp32, heads, extra=None, variant=None):
    n, dp = _nd(params, qk, single.shape[-1])
    _, plain = _variant(variant)
    from foldjet_engine.kernels.conditioned_transition.cuda import token_dit_fp8_train as f8

    act = f8.active(single.shape[0] * single.shape[2], single.shape[-1], fp32, plain, _sm90(single.device))
    nox, hsave, rx1 = "nox" in act, "hsave" in act and "squeeze" not in act and "dexpand" not in act, "rx1" in act
    return [
        torch.empty_like(single),
        *_saved_like(single, cond, pair, fp32, qk, heads, n, dp, bool(variant and variant[4]), nox, hsave, rx1, bool(variant and variant[5])),
    ]


@opaque(fake=_fwd_fake, name="token_dit_train_sm100_fwd")
def _fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    params: list[torch.Tensor],
    qk: bool,
    eq: float,
    ek: float,
    fp32: bool,
    heads: int,
    extra: torch.Tensor | None = None,
    variant: list[float] | None = None,
) -> list[torch.Tensor]:
    """The block's forward: [out, *saved activations] (every output freshly allocated); ``fp32``: fp32 GEMM operands (on TF32
    tensor cores, whatever the caller's allow_tf32) and the TF32 attention kernels, else bf16. Autocast off: the precision is
    ``fp32``'s, and an enclosing bf16 autocast would otherwise turn the fp32 path's GEMM outputs bf16. ``variant``: the model's
    options (``token_dit.options``; None: the defaults)."""
    with _tf32(fp32), torch.autocast("cuda", enabled=False):
        return _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, heads, extra, variant)


def _variant(variant):
    """(the options tuple, plain = the defaults) of a ``variant`` argument."""
    from foldjet_engine.kernels.conditioned_transition.cuda import VARIANT_DEFAULT

    v = VARIANT_DEFAULT if variant is None else tuple(float(t) for t in variant)
    return v, v == VARIANT_DEFAULT


def _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, H, extra=None, variant=None):
    v, plain = _variant(variant)
    opt, qkm = _opt_bits(v)
    par, up, ada_eps = bool(v[0]), bool(v[4]), float(v[2])
    if qk:
        qkm = 1
    mm, _mm32, addmm = _owned_gemms90(single)
    from foldjet_engine.kernels.augmented_attention import cuda as sm90
    from foldjet_engine.kernels.augmented_attention.cuda import sm100
    from foldjet_engine.kernels.conditioned_transition import cuda as rows
    from foldjet_engine.kernels.conditioned_transition.cuda.train import ext

    A, _, L, D = single.shape
    DH = D // H
    nexp, dpw = _nd(params, qk, D)
    T = ext(
        D,
        DH,
        rpb=_native_rpb() if _nrows(single.device) else 8,
        empty_uniform=_empty_uniform90(single.device, single.shape[0], single.shape[2]),
        dc=cond.shape[-1],
        n=nexp,
        dp=dpw,
    )
    names = _names_for(qk, None if plain else v)
    P = dict(zip(names, params, strict=True))
    M, R, dev = A * L, L * L, single.device
    at = torch.float32 if fp32 else BF
    h100 = _sm90(dev)
    native = _native90(dev)
    nrows = _nrows(dev)
    Mc = L if _shared_cond(single, cond) else M  # rows of the conditioning products (L: shared by the samples)
    assert Mc == M or (native and plain)
    # q / k / v copies: QK-norm (per head, or the flat q / k LayerNorm: qkm 2), or the sm_90a cores' pre-scaled q (see _saved_like). B200
    # took the flat LayerNorm's q / k as raw column views of qkvg before 2026-11-17 (no LayerNorm at all: RoseTTAFold3 training RED)
    copies = bool(qk or qkm or h100)
    c2 = None if nrows else cond.reshape(M, cond.shape[-1]).contiguous()  # native rows: cond_prep90 reads cond as given
    own_c = _own_cond(cond, fp32)
    chat = torch.empty(Mc, cond.shape[-1], device=dev, dtype=at)
    cbf = None if own_c else torch.empty_like(chat)
    # B200 inside a capture: the gate GEMM (0) and the pair pass (1) beside the chain, joined before the core; the conditioning prep (2)
    # beside the weight pack (2026-11-17, part 3: issued first, so its few blocks are not queued behind the pack's full grid)
    sd = _Side(dev, 3)
    cst = torch.empty(Mc, 2, device=dev)
    if nrows and not h100:
        sd.run(2, lambda: T.cond_prep90(cond, chat, cbf, cst, EPS))
        W = _pack(P, qk, dev, at, None if plain else v)
        sd.wait(2)
    else:
        if nrows:
            T.cond_prep90(cond, chat, cbf, cst, EPS)
        else:
            T.cond_prep(c2, chat, cbf, cst, EPS)
        W = _pack(P, qk, dev, at, None if plain else v)
    if opt & 1:  # the identity AdaLN: its tables read the raw conditioning (cbf; fp32 with an fp32 cond: cond itself), not LN(cond)
        chat.copy_(cast(torch.Tensor, c2 if own_c else cbf))  # (a type cast only)
    Wn, Wg = W["Wn"], W["Wg"]
    p2 = pair.reshape(R, dpw)
    apb_pair = _pair_apb(h100, fp32, dpw, H, pair.dtype)
    pst = torch.empty(0 if apb_pair else R, 2, device=dev)  # (apb_pair: the AttentionPairBias route's own statistics, see _pair_fwd)
    bu = torch.empty(0, device=dev)
    if h100:
        G = mm(chat, Wn.t())  # [M, 4D] s1 | sh1 | s2 | sh2
        Gg = mm(cast(torch.Tensor, c2 if own_c else cbf), Wg.t())  # [M, 2D] gate1 | gate2
    else:
        ph, bias, pst, bu = sd.run(1, lambda: _pair_fwd(T, W, pair, p2, pst, mask, extra, mm, h100, fp32, at, H, L, dpw))
        Gg = sd.run(0, lambda: mm(cast(torch.Tensor, c2 if own_c else cbf), Wg.t()))
        G = mm(chat, Wn.t())  # [M, 4D] s1 | sh1 | s2 | sh2 (Gg [M, 2D] gate1 | gate2: read after the join)
    bs1, bs2, bg1, bg2 = W["bs1"], W["bs2"], W["bg1"], W["bg2"]
    xa = torch.empty(M, D, device=dev, dtype=at)
    xst = torch.empty(M, 2, device=dev)
    xt = torch.empty(M, D, device=dev, dtype=at)
    f8, f8p, f8w = _fp8_train(W, M, D, fp32, plain, h100, ("qkvg", "out", "expand", "squeeze"))
    hsave = "hsave" in f8p and "squeeze" not in f8p and "dexpand" not in f8p  # keep h for the backward (swiglu_bwd then writes no h)
    sin = single.reshape(M, D).contiguous()
    # "nox" (docs/fp8_training.md, round 2): the fp32 residual x is float(input) exactly, so the saved fp32 copy (written 2 u, read 2 u by
    # res_adaln_b and 2 u by adaln_a_bwd; u = M D bf16 bytes) is dropped: both kernels read the input itself (1 u each); bit-identical
    x = None if "nox" in f8p else torch.empty(M, D, device=dev)  # else the fp32 residual, written by adaln_a
    T.adaln_a(
        sin, G, bs1, xa, xst, ada_eps, x, opt, xt if par else None, bs2 if par else None
    )  # reads the input, writes x on the way (the parallel block: xt as well)
    Wqkvg, bqkvg = W["Wqkvg"], W["bqkvg"]
    qkvg = f8.mm(*f8.q8(xa), f8w["qkvg"], bias=bqkvg) if "qkvg" in f8p else addmm(bqkvg, xa, Wqkvg.t())
    rqk = torch.empty(M, 2 * H, device=dev)
    nq, nk = W["nq"], W["nk"]
    if copies:  # B200 fp32: v is not copied, the core reads it as a column view of qkvg
        qn = torch.empty(M, D, device=dev, dtype=at)
        kn = torch.empty_like(qn)
        vc = None if (fp32 and not h100) else torch.empty_like(qn)
        if qkm == 2:  # the flat q / k LayerNorm (affine [D], its eps)
            qs = (LOG2E / DH**0.5) if h100 else 1.0  # H100: q scaled for the exp2 core; the B200 cores scale q themselves
            T.qknorm(qkvg, nq, nk, qn, kn, vc, rqk, float(v[6]), float(v[6]), 2, qs, W["nqb"], W["nkb"])
        else:
            T.qknorm(qkvg, nq, nk, qn, kn, vc, rqk, eq, ek, qk, (LOG2E / DH**0.5) if h100 else 1.0)  # H100: q scaled for the exp2 core
        vsave = vc if vc is not None else qkvg.new_empty(0, D)
        if vc is None:
            vc = qkvg[:, 2 * D : 3 * D]
    else:  # the core reads q / k / v as column views of qkvg: no copies
        qn, kn, vc = (qkvg[:, i * D : (i + 1) * D] for i in range(3))
    O = None
    if h100:
        ph, bias, pst, bu = _pair_fwd(T, W, pair, p2, pst, mask, extra, mm, h100, fp32, at, H, L, dpw)
        km = kmp = perm = None
        if native:
            bias, km, kmp = T.train_bias90(bias, mask, A, fp32)
        elif fp32:  # TF32 core: key-permuted bias / mask
            perm = rows.key_perm(L, dev)
            bias = bias[..., perm].contiguous()
        if fp32:  # TF32 core: v^T, key-permuted bias / mask
            vt = torch.empty(D, M, device=dev)
            T.transpose(vc, vt)  # v^T [D, M]
            og, LSE = sm90.tf32_forward(
                qn,
                kn,
                vt,
                bias,
                kmp if native else _key_mask(mask, A, L, perm),
                A,
                L,
                qkvg,
                H,
                DH,
            )  # sigmoid(g) o, no O
            del vt
        else:
            og, LSE = sm90.forward(qn, kn, vc, bias, km if native else _key_mask(mask, A, L), A, L, qkvg, H, DH)  # sigmoid(g) o, no O
    else:
        sd.join()  # the bias and Gg are in
        if not fp32 and mask is not None and not _pair_fused(h100, fp32, dpw):  # (the one-pass pair_bias wrote the mask)
            bias.masked_fill_(~mask.reshape(1, 1, L), float("-inf"))
        O, LSE = sm100.forward_tf32(qn, kn, vc, bias, A, L, H, DH) if fp32 else sm100.forward(qn, kn, vc, bias, A, L, H, DH)
        og = torch.empty(M, D, device=dev, dtype=at)
        T.gate_o(O, qkvg, og)
    Wo = W["Wo"]
    y = f8.mm(*f8.q8(og), f8w["out"]) if "out" in f8p else mm(og, Wo.t())
    rx1 = (
        "rx1" in f8p
    )  # round 4: x1 (fp32 residual after the attention branch) is not stored; res_c and the backward recompute it from the input
    x1 = torch.empty(0, D, device=dev) if rx1 else torch.empty(M, D, device=dev)
    x1st = torch.empty(M, 2, device=dev)
    xq = None  # the e4m3 copy of xt (+ row scales) made by res_adaln_b itself ("expand"; round 4: no separate q8 pass over xt)
    if "expand" in f8p and not par and not up and f8.FUSE_Q8 and at is BF:
        xq = (torch.empty(M, D, device=dev, dtype=f8.E4M3), torch.empty(M, 1, device=dev))
        T.res_adaln_b_q8(sin if x is None else x, y, Gg, bg1, G, bs2, x1, xt, x1st, xq[0], xq[1].view(-1), ada_eps, opt)
    else:
        T.res_adaln_b(
            sin if x is None else x, y, Gg, bg1, G, bs2, x1, xt, x1st, ada_eps, opt
        )  # the parallel block: x1 only (xt is adaln_a's)
    Wab = W["Wab"]
    h = torch.empty(M, nexp * D, device=dev, dtype=at)
    gsw = None if (up or f8p) else _swiglu_gemm(xt, Wab)
    if gsw is not None:  # expand GEMM + SwiGLU, a | b saved on the way
        ab = torch.empty(M, 2 * nexp * D, device=dev, dtype=at)
        gsw(xt, Wab, h, ab)
    elif "squeeze" in f8p:  # FP8 squeeze: h made straight as e4m3 + row scales (no bf16 h)
        ab = f8.mm(*(xq or f8.q8(xt)), f8w["expand"]) if "expand" in f8p else mm(xt, Wab.t())
        h8, sh = f8.swiglu_q8(ab)
    else:
        ab = (
            f8.mm(*(xq or f8.q8(xt)), f8w["expand"]) if "expand" in f8p else mm(xt, Wab.t())
        )  # [M, 2 * n D] a | b ([M, 3 n D] a | b | u up-gated)
        rows.swiglu_rows(ab, h, up=up)
    Wsq = W["Wsq"]
    z = f8.mm(h8, sh, f8w["squeeze"]) if "squeeze" in f8p else mm(h, Wsq.t())
    out = torch.empty(M, D, device=dev, dtype=single.dtype)
    if rx1:
        T.res_c(x1, z, Gg, bg2, out, sin if x is None else x, y, bg1)
    else:
        T.res_c(x1, z, Gg, bg2, out)
    # custom-op outputs may not alias each other: without QK-norm q / k / v are views of qkvg -- not outputs, recovered in _bwd
    qkv_out = [qn, kn, vsave] if copies else []
    if cbf is None:
        assert c2 is not None  # cbf is None only when _own_cond (never native), which built c2
        cbf = c2.new_empty(0, cond.shape[-1], dtype=at)
    # the sm_90a cores write og and the backward needs no O: H100 keeps none (not even a placeholder, which AOT mis-strides)
    return [
        out.view(single.shape),
        single.new_empty(0, D, dtype=torch.float32) if x is None else x,
        chat,
        cbf,
        cst,
        G,
        Gg,
        xst,
        xa,
        qkvg,
        rqk,
        *qkv_out,
        bias,
        *([] if h100 else [O]),
        LSE,
        og,
        y,
        x1,
        x1st,
        xt,
        ab,
        z,
        ph,
        pst,
        bu,
        h if hsave else h.new_empty(0, nexp * D),
    ]


def _tf32_attn_bwd(T, qs, k, v, dob, bias_p, mask, lse, dd, A, L):
    """The H100 fp32 block's attention backward on the TF32 CUDA core: dQ (for the unscaled q), dK, dV [A L, 768] fp32 and dbias
    [16, L, L] fp32 in natural key order. qs: q in exp2 units; bias_p: the forward's key-permuted bias (log2 units). The
    core takes q, k, dO transposed as well (tiled transposes, ``T.transpose``), and its bias / LSE / D in the permuted orders."""
    from foldjet_engine.kernels.augmented_attention import cuda as sm90
    from foldjet_engine.kernels.conditioned_transition import cuda as rows

    perm = rows.key_perm(L, qs.device)
    inv = torch.argsort(perm)
    nh_ = bias_p.shape[0]
    dh_ = qs.shape[1] // nh_

    def tr(t):
        out = t.new_empty(t.shape[1], t.shape[0])
        T.transpose(t, out)
        return out

    bias_tp = bias_p[:, perm][..., inv].transpose(1, 2).contiguous()  # [h, key, q position]
    dq, dk, dv, db_p = sm90.tf32_backward(
        qs,
        k,
        v,
        dob,
        tr(qs),
        tr(k),
        tr(dob),
        bias_p,
        bias_tp,
        lse,
        dd,
        lse[..., perm].contiguous(),
        dd[..., perm].contiguous(),
        _key_mask(mask, A, L),
        _key_mask(mask, A, L, perm),
        A,
        L,
        nh_,
        dh_,
    )
    return dq, dk, dv, db_p[..., inv]


def _native_attn_bwd90(T, qs, k, v, dob, bias, mask, lse, dd, A, L, fp32, dbf32=False):
    """... `dbf32`: the pair-bias backward reads dbias in fp32 and in the log2-unit bias's scale (the natural-unit dbias / LOG2E): fp32, scaled, unrounded."""
    from foldjet_engine.kernels.augmented_attention import cuda as sm90

    H = bias.shape[0]  # this layout's heads (the module-level H is the default 16)
    DH = qs.shape[1] // H
    tf32_defs = sm90.tf32_defs(H, DH)
    bf_defs = sm90.fwd_defs(H, DH)  # attn_fwd / dq / dkv / dqb build defines of this layout (none at 16 x 48)

    dq, dk, dv = (torch.empty_like(qs, dtype=torch.float32) for _ in range(3))
    db = None if (not fp32 and A % 3 == 0) else torch.empty((A, H, L, L), device=qs.device, dtype=torch.float32)
    km, kmp = T.train_keymask90(mask, qs, A, L, fp32)
    empty_defs = ("TOKEN_EMPTY_UNIFORM=1",) if _empty_uniform90(qs.device, A, L) else ()
    if fp32:

        def tr(t):
            out = t.new_empty((t.shape[1], t.shape[0]))
            T.transpose(t, out)
            return out

        bt, lp, dp = T.train_tf32_bwd_layout90(bias, lse, dd)
        # per-sample dbias buffer + reduce: the TF32 dQ pass with C chunk buffers is bitwise repeatable too, but measured no
        # faster (dQ 2365 us against 1766 + 613 us), so not used here
        core = sm90._ext("attn_tf32", ("DB_PER_SAMPLE=1", *tf32_defs, *empty_defs))
        core.attn_tf32_dqb(qs, dob, k, v, tr(k), bias, lse, dd, kmp, dq, db)
        core.attn_tf32_dkv(qs, dob, tr(qs), tr(dob), k, v, bt, lp, dp, km, dk, dv)
    elif A % 3 == 0:
        # dbias in the dQ pass: the three samples of a work item are summed in shared memory and added with ONE TMA bulk reduce,
        # in a fixed order (bitwise repeatable). The per-sample variant below writes an [A, H, L, L] fp32 buffer (1.8 GB at A48
        # L768), takes 1.4 ms in dK / dV for the atomics and 0.6 ms in the reduce: 2.6 ms against 1.6 ms for this pass.
        dbc = torch.zeros(sm90.dqb_chunks(A), H, L, L, device=qs.device, dtype=torch.float32)
        sm90._ext("attn_dqb", (*bf_defs, *empty_defs)).attn_dqb(qs, k, v, dob, bias, km, lse, dd, dq, dbc)
        dbn = dbc.sum(0)  # the chunk buffers summed in order: bitwise repeatable
        sm90._ext("attn_dkv", ("DBIAS=0", *bf_defs, *empty_defs)).attn_dkv(qs, k, v, dob, bias, km, lse, dd, dk, dv, dbn)
        return dq, dk, dv, (dbn.mul_(1.0 / LOG2E) if dbf32 else dbn.to(torch.bfloat16))
    else:
        sm90._ext("attn_dq", (*(() if L % 192 == 0 else ("NWG=2",)), *sm90.dq_defs(H, DH), *empty_defs)).attn_dq(
            qs, k, v, dob, bias, km, lse, dd, dq
        )
        sm90._ext("attn_dkv", ("DB_PER_SAMPLE=1", *bf_defs, *empty_defs)).attn_dkv(qs, k, v, dob, bias, km, lse, dd, dk, dv, db)
    return dq, dk, dv, T.train_db_reduce90(db, fp32, 1.0 / LOG2E if dbf32 else 1.0, dbf32)


def _native_final90(T, names, params, W, allp, dWn, dWf, dWqkvg, dWo, dWg, dWab, dWsq, qk):
    from foldjet_engine.kernels.integrations.token_dit import QK_WEIGHTS, WEIGHTS

    canonical = WEIGHTS + (QK_WEIGHTS if qk else ())
    P = dict(zip(names, params, strict=True))
    D = dWqkvg.shape[1]  # this layout's token width (768, or 1024 for 16 x 64); the module-level D is the default
    # The placeholders are not read by entries whose deterministic reduction is fused into the writer.
    src = [
        dWqkvg[:D],
        allp,
        dWqkvg[D : 2 * D],
        dWqkvg[2 * D : 3 * D],
        dWqkvg[3 * D :],
        dWo,
        W["w1"],
        dWn[:D],
        allp,
        dWn[D : 2 * D],
        dWg[:D],
        allp,
        W["wp"],
        dWf,
        W["w2"],
        dWn[2 * D : 3 * D],
        allp,
        dWn[3 * D :],
        dWg[D:],
        allp,
        dWab[: dWab.shape[0] // 2],
        dWab[dWab.shape[0] // 2 :],
        dWsq,
    ]
    if qk:
        src.extend([allp, allp])
    out = T.train_final90(
        [P[n] for n in canonical],
        src,
        dWn,
        W["Wraw"],
        W["w1"],
        W["w2"],
        allp,
        dWf,
        W["Wb"],
        W["wp"],
        True,
    )
    result = dict(zip(canonical, out, strict=True))
    return [result[n] for n in names if n in result]  # (an option leaf after the canonical ones is the caller's)


#: the canonical leaves (WEIGHTS + QK_WEIGHTS order) no weight-gradient GEMM writes: biases, cond-LN weights, the pair weights
_REST = (1, 6, 8, 11, 12, 13, 14, 16, 19, 23, 24)
_COND_W = (
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_bias.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_bias.weight",
)


def _final_b200(T, sd, lf, names, qk, W, P, allp, dWn, early, D):
    """B200 (2026-11-17, part 3): the final gradients beside the chain, into the leaves ``lf`` (the GEMMs wrote the weight leaves already):
    side stream 0 after dWn -- the four cond-AdaLN weight leaves dWn diag(w) with the cond-LN partial rows (``final_lnw100``), the
    partial-row reductions (``final_reduce100``; every row kernel has written its rows: the caller issues this after the last one);
    side stream 1 after the pair backward and stream 0 -- the remaining leaves (``final_rest100``). Fixed orders throughout: bitwise
    the values of ``train_final90``. Returns (the leaves in ``names`` order, the buffers to keep alive until the caller's join)."""
    from foldjet_engine.kernels.integrations.token_dit import QK_WEIGHTS, WEIGHTS

    canonical = WEIGHTS + (QK_WEIGHTS if qk else ())
    outs = [lf[n] for n in canonical]
    fill = [i in _REST for i in range(len(canonical))]

    def lnw():
        part = T.final_lnw100(dWn, *(P[n] for n in _COND_W), W["w1"], W["w2"], [lf[n] for n in _COND_W])
        return T.final_reduce100(allp, part, (7 if qk else 5) * D)

    rp, lp = sd.run(0, lnw)
    sd.after(1, 0)
    dWf = early[1]
    sd.run(1, lambda: T.final_rest100(outs, fill, rp, lp, dWf, W["Wb"], W["wp"]))
    return [lf[n] for n in names], (rp, lp, dWf, dWn)


def _rest_b200(T, sd, lf, names, W, P, sums, dWn, early, qkm, D, H, DH):
    """B200 (2026-11-17, part 3), a model's options outside the fused final kernels (Chai-1's identity AdaLN, RoseTTAFold3's flat q / k
    LayerNorm): the weight leaves are the GEMMs' (the caller's), the sigmoid AdaLN's four cond leaves and cond-LN partial rows come from
    ``final_lnw100`` on side stream 0 after dWn (None: the identity AdaLN, whose GEMMs wrote them), the small leaves -- biases, cond-LN
    weights, the pair weights, the q / k norm affine -- one multi-tensor copy on side stream 1 after the pair backward. ``sums``: the
    column sums of the row kernels' partial rows (made by the caller after the last row pass). The tail was ~48 us after the last row
    pass (RoseTTAFold3 L 128: unfold_lnw, its reduction, the sums and a 25-leaf copy of ~57 MB)."""

    def lnw():
        if dWn is None:
            return None
        part = T.final_lnw100(dWn, *(P[n] for n in _COND_W), W["w1"], W["w2"], [lf[n] for n in _COND_W])
        return part.view(2, -1, part.shape[-1]).sum(1)  # [2, DC]: each branch's partial rows in order

    dw12 = sd.run(0, lnw)
    sd.after(1, 0)
    dWf = early[1]

    def small():
        pb = sums[: 5 * D].view(5, D)
        z = lf["attention.ada_ln_in.ln_cond.weight"].new_zeros(2, lf["attention.ada_ln_in.ln_cond.weight"].numel())
        w12 = z if dw12 is None else dw12
        g = {
            "attention.ada_ln_in.ln_cond.weight": w12[0],
            "attention.ada_ln_in.to_scale.bias": pb[3],
            "attention.to_scale.bias": pb[2],
            "attention.to_query.bias": pb[4],
            "attention.ln_pair.weight": (dWf * W["Wb"]).sum(0),
            "attention.to_bias.weight": dWf * W["wp"],
            "transition.ada_ln_in.ln_cond.weight": w12[1],
            "transition.ada_ln_in.to_scale.bias": pb[1],
            "transition.to_scale.bias": pb[0],
        }
        if qkm == 1:
            dwqk = sums[5 * D :].view(2, H, DH).sum(1).reshape(-1)
            g.update({"attention.norm_query.weight": dwqk[:DH], "attention.norm_key.weight": dwqk[DH:]})
        if qkm == 2:  # the flat q / k LayerNorm's affine
            kq = sums[5 * D : 9 * D].view(4, D)
            g.update({"kq.wq": kq[0], "kq.wk": kq[1], "kq.bq": kq[2], "kq.bk": kq[3]})
        ks = [n for n in names if n in g]
        torch._foreach_copy_([lf[n] for n in ks], [g[n].reshape(lf[n].shape) for n in ks])  # pyright: ignore[reportPrivateImportUsage]

    sd.run(1, small)
    return [lf[n] for n in names], (dw12, dWf, dWn, sums)


def _bwd_fake(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads, extra=None, variant=None):
    return [
        torch.empty_like(single),
        torch.empty_like(cond),
        torch.empty_like(pair),
        *(torch.empty_like(p) for p in params),
        *([] if extra is None else [torch.empty_like(extra)]),
    ]


@opaque(fake=_bwd_fake, name="token_dit_train_sm100_bwd")
def _bwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    params: list[torch.Tensor],
    saved: list[torch.Tensor],
    dout: torch.Tensor,
    qk: bool,
    eq: float,
    ek: float,
    fp32: bool,
    heads: int,
    extra: torch.Tensor | None = None,
    variant: list[float] | None = None,
) -> list[torch.Tensor]:
    """The block's backward: [d single, d cond, d pair, *d params] in the inputs' dtypes (autocast off, as ``_fwd``)."""
    with _tf32(fp32), torch.autocast("cuda", enabled=False):
        return _bwd_body(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads, extra, variant)


def _xgrad(DB, extra, H, L, scale=1.0):
    """d extra [L, L] (the extra term enters every head's logits once, in natural units: the sum of dbias over the heads) as a
    one-element list, or none when the caller does not differentiate it. ``scale``: DB's scale to natural units (the APB route's
    fp32 dbias is the natural one / LOG2E)."""
    if extra is None:
        return []
    s = DB.reshape(H, L, L).sum(0, dtype=torch.float32)
    return [(s * scale if scale != 1.0 else s).to(extra.dtype)]


def _apb_pair_bwd(p2, DB, wf_l2, pst, bu, L, H, dp, dpair):
    """dpair (written into ``dpair``) and dWf [H, dp] fp32 of the APB pair-bias backward. ``DB``: [H, L, L] fp32, the log2-unit bias's
    gradient (natural dbias / LOG2E: ``_native_attn_bwd90(.., dbf32)``), ``wf_l2`` the forward's bf16 weights (log2 units) -- so
    dxh = sum_h dbias_h Wf_h, its row means and S2 = sum_h dbias_h bias_h are the natural ones, and dWf = dbias^T LN(pair) comes out
    divided by LOG2E. The per-block dWf partials (one row per block, no atomics) are summed in block order: deterministic."""
    from foldjet_engine.kernels.augmented_attention.cuda import apb

    acc = torch.empty(apb.ext().pair_bwd_blocks(p2, H, L), apb.acc_n(H, 384, dp), device=p2.device)
    apb.pair_bias_bwd(p2, DB, wf_l2, L, EPS, acc=acc, d=384, saved=None if dp == apb.DP else (pst, bu.view(H, L, L)), out=dpair)
    o = 2 * 384 + apb.width(H, 384)
    return acc[:, o : o + H * dp].sum(0).view(H, dp) * LOG2E


def _bwd_body(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, H, extra=None, variant=None):
    v, plain = _variant(variant)
    opt, qkm = _opt_bits(v)
    par, up = bool(v[0]), bool(v[4])
    if qk:
        qkm = 1
    mm, mm32, _addmm = _owned_gemms90(single)
    from foldjet_engine.kernels.augmented_attention import cuda as sm90
    from foldjet_engine.kernels.augmented_attention.cuda import sm100
    from foldjet_engine.kernels.conditioned_transition.cuda.train import ext

    D = single.shape[-1]
    DH = D // H
    nexp, dpw = _nd(params, qk, D)
    T = ext(
        D,
        DH,
        rpb=_native_rpb() if _nrows(single.device) else 8,
        empty_uniform=_empty_uniform90(single.device, single.shape[0], single.shape[2]),
        dc=cond.shape[-1],
        n=nexp,
        dp=dpw,
    )
    h100 = _sm90(single.device)
    saved = list(saved)
    if not (qk or qkm or h100):  # the forward's q / k / v: column views of qkvg (see _fwd_body)
        saved = [
            *saved[:10],
            *(saved[8][:, i * D : (i + 1) * D] for i in range(3)),
            *saved[10:],
        ]  # after qkvg (8), rqk (9)
    O = None if h100 else saved.pop(14)  # H100 keeps og only (the cores write it)
    (
        x,
        chat,
        cbf,
        cst,
        G,
        Gg,
        xst,
        xa,
        qkvg,
        rqk,
        qn,
        kn,
        vc,
        bias,
        LSE,
        og,
        y,
        x1,
        x1st,
        xt,
        ab,
        z,
        ph,
        pst,
        bu,
        hs,
    ) = saved
    if vc.numel() == 0:  # B200 fp32 + QK-norm: v as the forward's core read it, a column view of qkvg
        vc = qkvg[:, 2 * D : 3 * D]
    names = _names_for(qk, None if plain else v)
    at = torch.float32 if fp32 else BF
    W = _pack(dict(zip(names, params, strict=True)), qk, single.device, at, None if plain else v)
    Wn, Wraw, Wg, Wqkvg, Wo, Wab, Wsq, Wf = (
        W["Wn"],
        W["Wraw"],
        W["Wg"],
        W["Wqkvg"],
        W["Wo"],
        W["Wab"],
        W["Wsq"],
        W["Wf"],
    )
    nq, nk, bs1, bs2, bg1, bg2, w1, w2, wp, Wb = (W[k] for k in ("nq", "nk", "bs1", "bs2", "bg1", "bg2", "w1", "w2", "wp", "Wb"))
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    sdt, cdt, pdtype = single.dtype, cond.dtype, pair.dtype
    apb_pair = _pair_apb(h100, fp32, dpw, H, pdtype)  # the forward's route (its saved row statistics / unmasked bias are this backward's)
    native = _native90(dev)
    nrows = _nrows(dev)
    f8, f8p, f8w = _fp8_train(W, M, D, fp32, plain, h100, ("dqkvg", "dout", "dexpand", "dsqueeze"))
    e5 = bool(f8p) and f8.GRAD_E5M2
    c2, p2 = (
        (None if nrows else cond.reshape(M, cond.shape[-1]).contiguous()),
        pair.reshape(R, dpw),
    )  # native: cond_bwd90 reads cond as given
    dout = dout.reshape(M, D).to(sdt).contiguous()
    # the fused final-gradient kernels: the default options; B200 (2026-11-17) also a model with the sigmoid AdaLN and no q / k norm
    # (Boltz-2: the up-gate leaf's gradient is the third block of dWab) -- its torch tail was unfold_lnw + ~8 small kernels + the 24-leaf copy
    nfinal = nrows and (plain or (not h100 and not (opt & 1) and qkm == 0))
    # B200 (2026-11-17, part 3): the weight gradients straight into their leaves (``lf``), the rest of the final gradients beside the
    # chain (``_final_b200``): a captured step ended in a 25 us tail (the cond-LN partials, the partial-row reductions, one copy of
    # every weight gradient into its leaf) after the last row pass
    lf = (
        {n: torch.empty(p.shape, device=dev, dtype=p.dtype) for n, p in zip(names, params, strict=True)}
        if _FINAL_B200
        and not h100
        and (fp32 or _pair_fused(h100, fp32, dpw))  # (the pair backward runs beside the chain: ``early`` below)
        and all(p.dtype is torch.float32 and p.is_contiguous() for p in params)
        else None
    )
    # per-block column sums: bg2 bs2 bg1 bs1 bq | dwq dwk (per head); rows: the row kernels' block count (T.part_rows)
    # (the flat q / k LayerNorm: dwq dwk dbq dbk, [D] each)
    allp = torch.empty(T.part_rows(M, fp32), (9 if qkm == 2 else 7) * D, device=dev)
    part = [allp[:, i * D : (i + 1) * D] for i in range(5)]
    pwqk = allp[:, 5 * D :]
    dG = torch.empty(M, 4 * D, device=dev, dtype=at)
    dGg = torch.empty(M, 2 * D, device=dev, dtype=at)
    sd = _Side(dev, 3)  # B200 inside a capture: the weight-gradient GEMMs (streams 0, 2) and the pair backward (stream 1) beside the chain
    dz = torch.empty(M, D, device=dev, dtype=at)
    T.res_c_bwd(dout, z, Gg, bg2, dz, dGg, part[0])
    dh = f8.mm(*f8.q8(dz, e5), f8w["dsqueeze"]) if "dsqueeze" in f8p else mm(dz, Wsq)  # [M, n D]
    dab = torch.empty_like(ab)
    h = hs if hs.numel() else torch.empty_like(dh)  # hsave: the forward's h
    if "dexpand" in f8p:  # swiglu_bwd + dab as 8-bit with row scales, for the dxt dgrad
        dab8, sdab = f8.swiglu_bwd_q8(dh, ab, dab, h, e5)
    else:
        T.swiglu_bwd(dh, ab, dab, None if hs.numel() else h, up)
    if lf is not None:
        sd.run(0, lambda: _mm32(dz.t(), h, out=lf["transition.squeeze.weight"]))
        dWsq = None
    else:
        dWsq = sd.run(0, lambda: mm32(dz.t(), h))
    # fp32: the dxt / dxa GEMMs write straight into their column blocks of dG (the row kernels then copy nothing); the H100 owned
    # GEMMs return a fresh tensor, which the row kernels copy as before
    direct = fp32 and mm is torch.mm
    if "dexpand" in f8p:
        dxt = f8.mm(dab8, sdab, f8w["dexpand"])
    elif direct:
        dxt = torch.mm(dab, Wab, out=dG[:, 3 * D :])
    else:
        dxt = mm(dab, Wab)
    if lf is not None:  # expand_a | expand_b (| the up gate): one GEMM per leaf
        abn = ["transition.expand_a.weight", "transition.expand_b.weight", *([UP_LEAF[0]] if up else [])]
        sd.run(0, lambda: [_mm32(dab[:, i * nexp * D : (i + 1) * nexp * D].t(), xt, out=lf[n]) for i, n in enumerate(abn)])
        dWab = None
    else:
        dWab = sd.run(0, lambda: mm32(dab.t(), xt))
    dx1 = torch.empty(M, D, device=dev)
    dy = torch.empty(M, D, device=dev, dtype=at)
    xin = single.reshape(M, D).contiguous() if x.numel() == 0 else x  # nox: the block input is the residual (see _fwd_body)
    if x1.numel() == 0:  # rx1: x1 was not kept; the kernel recomputes it from the input
        T.res_adaln_b_bwd(dout, dxt, xin, x1st, G, bs2, Gg, bg1, y, dx1, dy, dG, dGg, part[1], part[2], opt, True)
    else:
        T.res_adaln_b_bwd(
            dout, dxt, x1, x1st, G, bs2, Gg, bg1, y, dx1, dy, dG, dGg, part[1], part[2], opt
        )  # the parallel block: no AdaLN of x1 (dxt goes to adaln_a_bwd)
    dog = f8.mm(*f8.q8(dy, e5), f8w["dout"]) if "dout" in f8p else mm(dy, Wo)
    if lf is not None:
        sd.run(0, lambda: _mm32(dy.t(), og, out=lf["attention.to_out.weight"]))
        dWo = None
    else:
        dWo = sd.run(0, lambda: mm32(dy.t(), og))

    def g_leaves(k):  # both to_scale leaves (lf): dGg -- complete after res_adaln_b_bwd -- against the conditioning rows, side stream k
        leaves = lf
        assert leaves is not None
        cg = c2 if cbf.numel() == 0 else cbf
        gn = ["attention.to_scale.weight", "transition.to_scale.weight"]
        sd.run(k, lambda: [_mm32(dGg[:, i * D : (i + 1) * D].t(), cg, out=leaves[n]) for i, n in enumerate(gn)])

    g_early = lf is not None and _GEARLY_B200  # (part 10) on stream 0 after dWo, not behind the q | k | v | g GEMMs at stream 2's end
    if g_early:
        g_leaves(0)
    dob = torch.empty(M, D, device=dev, dtype=at)
    dd = torch.empty(A, H, L, device=dev)
    dqkvg = torch.empty(M, 4 * D, device=dev, dtype=at)
    T.gate_o_bwd(dog, og if h100 else O, qkvg, dob, dd, dqkvg, L, h100)  # H100: D and dg from og (dO o = dog og)
    if native:
        DQ, DK, DV, DB = _native_attn_bwd90(T, qn, kn, vc, dob, bias, mask, LSE, dd, A, L, fp32, apb_pair)
    elif h100 and fp32:
        DQ, DK, DV, DB = _tf32_attn_bwd(T, qn, kn, vc, dob, bias, mask, LSE, dd, A, L)
    elif h100:  # dq for the unscaled q, dbias natural units
        DQ, DK, DV, DB = sm90.backward(qn, kn, vc, dob, bias, _key_mask(mask, A, L), LSE, dd, A, L, H, DH)
    else:
        # B200 fp32: dQ / dK / dV land in their column blocks of dqkvg; qknorm_bwd then rewrites dq / dk in place (QK-norm) or only sums
        DQ, DK, DV, DB = (
            sm100.backward_tf32(
                qn,
                kn,
                vc,
                dob,
                bias,
                LSE,
                dd,
                A,
                L,
                H,
                DH,
                dq=dqkvg[:, :D],
                dk=dqkvg[:, D : 2 * D],
                dv=dqkvg[:, 2 * D : 3 * D],
                det=True,
            )
            if fp32
            else sm100.backward(
                qn,
                kn,
                vc,
                dob,
                bias,
                LSE,
                dd,
                A,
                L,
                H,
                DH,
                det=True,
                bias_nt=_BIASNT_B200 >= 2 or (_BIASNT_B200 == 1 and A == 1),
                side=(lambda fn: sd.run(2, fn), lambda: sd.wait(2)) if _DQB_SIDE_B200 else None,
            )
        )

    def pair_bwd():  # B200: dpair and dWf need dbias only -- AttentionPairBias's one-pass backward (bf16, pair 128), else the rows' pass
        dpair = torch.empty(R, dpw, device=dev, dtype=pdtype)
        dWf = _pair_bwd_apb(h100, fp32, DB, p2, W["Wf_at"], dpair, H, L)
        if dWf is None:
            pwf = torch.empty(T.pair_grid(R), H * dpw, device=dev)
            T.pair_bias_bwd(DB.reshape(H, R).contiguous(), p2, pst, Wf, dpair, pwf)  # LN(pair) recomputed; dpair + the dWf partials
            dWf = _colsum(pwf).view(H, dpw)
        return dpair, dWf

    early = sd.run(1, pair_bwd) if not h100 and (fp32 or _pair_fused(h100, fp32, dpw)) else None
    T.qknorm_bwd(DQ, DK, DV, qkvg, rqk, nq, nk, dqkvg, part[4], pwqk, qkm)
    if "dqkvg" in f8p:
        dxa = f8.mm(*f8.q8(dqkvg, e5), f8w["dqkvg"])
    else:
        dxa = torch.mm(dqkvg, Wqkvg, out=dG[:, D : 2 * D]) if direct else mm(dqkvg, Wqkvg)
    if lf is not None:
        qkvgn = ["attention.to_query.weight", "attention.to_key.weight", "attention.to_value.weight", "attention.to_gate.weight"]
        sd.run(2, lambda: [_mm32(dqkvg[:, i * D : (i + 1) * D].t(), xa, out=lf[n]) for i, n in enumerate(qkvgn)])
        dWqkvg = None
    else:
        dWqkvg = sd.run(0, lambda: mm32(dqkvg.t(), xa))
    dx = torch.empty(M, D, device=dev, dtype=sdt)
    if par:  # both AdaLNs of the block input: dxa and dxt into one LN backward (bs2's partials into part[1])
        T.adaln_a_bwd(dxa, xin, xst, G, bs1, dx1, dx, dG, part[3], opt, dxt, bs2, part[1])
    else:
        T.adaln_a_bwd(dxa, xin, xst, G, bs1, dx1, dx, dG, part[3], opt)
    if not nfinal:
        sums = (allp if qkm else allp[:, : 5 * D]).sum(0) if h100 else _colsum(allp if qkm else allp[:, : 5 * D])
        pb = sums[: 5 * D].view(5, D)
        dwqk = sums[5 * D :].view(2, H, DH).sum(1).reshape(-1) if qkm == 1 else torch.zeros(2 * DH, device=dev)
    Mc = L if _shared_cond(single, cond) else M
    if Mc != M:  # the conditioning is shared by the samples (H100 default-options block): the sum over the samples first, every product below on L rows
        assert native and plain
        # the sum as two operands hi + lo ([2 L, n]: the sample sum rounded once to the operand type would add an error of the size of the
        # per-row roundings it replaces); the GEMMs run on 2 L rows and their fp32 halves are added
        dGs = torch.empty(2 * L, 4 * D, device=dev, dtype=at)
        dGgs = torch.empty(2 * L, 2 * D, device=dev, dtype=at)
        T.sample_sum2(dG.view(A, L, 4 * D), dGs, dGg.view(A, L, 2 * D), dGgs)

        def halves(y):  # fp32: cond_bwd reads them as they are (no operand rounding between the GEMM and the LayerNorm backward)
            return y[:L] + y[L:]

        dchat = halves(mm32(dGs, Wn))
        dcg = halves(mm32(dGgs, Wg))
        dWn = mm32(dGs.t(), torch.cat((chat, chat)))
        dWg = mm32(dGgs.t(), torch.cat((cbf, cbf)))
        fin = None
    else:
        if lf is not None and not nfinal and opt & 1:  # the identity AdaLN: dWn is its four weight leaves as they are
            sd.run(0, lambda: [_mm32(dG[:, i * D : (i + 1) * D].t(), chat, out=lf[n]) for i, n in enumerate(_COND_W)])
            dWn = None
        else:
            dWn = sd.run(0, lambda: mm32(dG.t(), chat))
        if lf is not None:
            Pd = dict(zip(names, params, strict=True))
            if nfinal:
                fin, keep = _final_b200(T, sd, lf, names, qk, W, Pd, allp, dWn, early, D)
            else:
                fin, keep = _rest_b200(T, sd, lf, names, W, Pd, sums, dWn, early, qkm, D, H, DH)
            if not g_early:
                g_leaves(2)
            dWg = None
        else:
            dWg = sd.run(0, lambda: mm32(dGg.t(), c2 if cbf.numel() == 0 else cbf))
            fin = None
        if _dc_epi(h100, fp32, dG, Wn):
            from foldjet_engine.kernels.conditioned_transition.cuda import tdt_epi

            dchat = tdt_epi.mm(dG, Wn, 1, bn=64, ks=8)
        else:
            dchat = mm(dG, Wn)
        dcg = mm(dGg, Wg)
    dc = torch.empty(Mc, cond.shape[-1], device=dev, dtype=cdt)
    if fin is not None:  # B200: every leaf written beside the chain (``_final_b200`` / ``_rest_b200``)
        if opt & 1:  # the identity AdaLN: the raw conditioning (no LayerNorm of it)
            dc.copy_((dchat.float() + dcg.float()).view_as(dc))
        else:
            T.cond_bwd90(dchat, dcg, cond, cst, dc)
        assert early is not None
        dpair, _ = early
        sd.join()
        del keep  # (read by the side streams until the join)
        return [dx.view(A, 1, L, D), dc.view(cond.shape), dpair.view(1, L, L, dpw), *fin, *_xgrad(DB, extra, H, L)]
    # (the weight-gradient buffers of the paths below: ``lf`` is None there)
    assert dWsq is not None and dWab is not None and dWo is not None and dWqkvg is not None and dWg is not None and dWn is not None
    if nfinal:
        T.cond_bwd90(dchat, dcg, cond, cst, dc)
        if early is not None:  # B200: made beside the chain (pair_bwd above)
            dpair, dWf = early
        elif apb_pair:
            dpair = torch.empty(R, dpw, device=dev, dtype=pdtype)
            dWf = _apb_pair_bwd(p2, DB, W["Wf_l2"], pst, bu, L, H, dpw, dpair)
        elif _pair_fused(h100, fp32, dpw):
            dpair = torch.empty(R, dpw, device=dev, dtype=pdtype)
            part = torch.empty(T.pair_grid(R), H * dpw, device=dev)
            T.pair_bias_bwd(
                DB.reshape(H, R).contiguous(), p2, pst, W["Wf"], dpair, part
            )  # LN(pair) recomputed; dpair and the dWf partials in one pass
            dWf = part.sum(0).view(H, dpw)
        else:
            dbv = DB.reshape(H, R).to(at)  # B200 bf16: the fp32 dbias into the bf16 GEMMs (H100 fp32: no-op)
            dWf = mm32(dbv, ph)
            dph = mm32(dbv.t(), W["Wf_at"])
            dpair = torch.empty(R, dpw, device=dev, dtype=pdtype)
            T.pair_ln_bwd(dph, p2, pst, dpair)
        sd.join()
        outs = _native_final90(T, names, params, W, allp, dWn, dWf, dWqkvg, dWo, dWg, dWab[: 2 * nexp * D], dWsq, qk)
        if up:  # the up-gate leaf (after the canonical ones): its own tensor
            outs = [*outs, dWab[2 * nexp * D :].to(params[len(outs)].dtype, copy=True)]
        return [
            dx.view(A, 1, L, D),
            dc.view(cond.shape),
            dpair.view(1, L, L, dpw),
            *outs,
            *_xgrad(DB, extra, H, L, LOG2E if apb_pair else 1.0),
        ]
    sd.join()  # dWn (side stream 0) is read below; the pair backward is in too
    dw12 = torch.zeros(2, cond.shape[-1], device=dev)
    if opt & 1:
        # the identity AdaLN: its tables read the raw conditioning (no LayerNorm of it, no folded LN weight)
        dc.copy_((dchat.float() + dcg.float()).view_as(dc))
        dWu = dWn
    else:
        if nrows:
            T.cond_bwd90(dchat, dcg, cond, cst, dc)
        else:
            T.cond_bwd(dchat, dcg, c2, cst, dc)
        dWu = torch.empty_like(dWn)
        Ws1, Wb1, Ws2, Wb2 = (Wraw[i * D : (i + 1) * D] for i in range(4))
        T.unfold_lnw(dWn, Ws1, Wb1, Ws2, Wb2, w1, w2, dWu, dw12)
    dpair = torch.empty(0) if early is not None else torch.empty(R, dpw, device=dev, dtype=pdtype)
    if early is not None:  # B200: made beside the chain (pair_bwd above)
        dpair, dWf = early
    elif apb_pair:
        dWf = _apb_pair_bwd(p2, DB, W["Wf_l2"], pst, bu, L, H, dpw, dpair)
    elif _pair_fused(h100, fp32, dpw):
        part = torch.empty(T.pair_grid(R), H * dpw, device=dev)
        T.pair_bias_bwd(DB.reshape(H, R).contiguous(), p2, pst, Wf, dpair, part)
        dWf = part.sum(0).view(H, dpw)
    else:
        dbv = DB.reshape(H, R).to(at)  # masked key columns are 0 (P = 0); H100: may be a transposed view
        dWf = mm32(dbv, ph)  # [H, dp]
        dph = mm32(dbv.t(), Wf.to(at))  # [R, dp]
        T.pair_ln_bwd(dph, p2, pst, dpair)
    dWs1, dWb1, dWs2, dWb2 = dWu.split(D)
    grads = {
        "attention.ada_ln_in.ln_cond.weight": dw12[0],
        "attention.ada_ln_in.to_scale.weight": dWs1,
        "attention.ada_ln_in.to_scale.bias": pb[3],
        "attention.ada_ln_in.to_bias.weight": dWb1,
        "attention.to_scale.weight": dWg[:D],
        "attention.to_scale.bias": pb[2],
        "attention.to_query.weight": dWqkvg[:D],
        "attention.to_query.bias": pb[4],
        "attention.to_key.weight": dWqkvg[D : 2 * D],
        "attention.to_value.weight": dWqkvg[2 * D : 3 * D],
        "attention.to_gate.weight": dWqkvg[3 * D :],
        "attention.norm_query.weight": dwqk[:DH],
        "attention.norm_key.weight": dwqk[DH:],
        "attention.to_out.weight": dWo,
        "attention.ln_pair.weight": (dWf * Wb).sum(0),
        "attention.to_bias.weight": dWf * wp,
        "transition.ada_ln_in.ln_cond.weight": dw12[1],
        "transition.ada_ln_in.to_scale.weight": dWs2,
        "transition.ada_ln_in.to_scale.bias": pb[1],
        "transition.ada_ln_in.to_bias.weight": dWb2,
        "transition.to_scale.weight": dWg[D:],
        "transition.to_scale.bias": pb[0],
        "transition.expand_a.weight": dWab[: nexp * D],
        "transition.expand_b.weight": dWab[nexp * D : 2 * nexp * D],
        "transition.squeeze.weight": dWsq,
    }
    if up:
        grads[UP_LEAF[0]] = dWab[2 * nexp * D :]
    if qkm == 2:  # the flat q / k LayerNorm's affine
        kq = sums[5 * D : 9 * D].view(4, D)
        grads.update({"kq.wq": kq[0], "kq.wk": kq[1], "kq.bq": kq[2], "kq.bk": kq[3]})
    # custom-op outputs may not alias each other: the gradients above are views of a few shared buffers, so each leaves as
    # its own tensor in the parameter's dtype -- one multi-tensor copy kernel for all of them (23 copies were ~45 us)
    outs = [torch.empty(p.shape, device=dev, dtype=p.dtype) for p in params]
    torch._foreach_copy_(  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
        outs, [grads[n].reshape(p.shape) for n, p in zip(names, params, strict=False)]
    )
    return [dx.view(A, 1, L, D), dc.view(cond.shape), dpair.view(1, L, L, dpw), *outs, *_xgrad(DB, extra, H, L, LOG2E if apb_pair else 1.0)]


class _Block(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, cond, pair, mask, qk, eq, ek, fp32, heads, extra, variant, *params):
        out, *saved = _fwd(single, cond, pair, mask, list(params), qk, eq, ek, fp32, heads, extra, variant)
        ctx.save_for_backward(single, cond, pair, *params, *saved, extra)
        ctx.meta = (mask, qk, eq, ek, fp32, len(params), heads, extra is not None and ctx.needs_input_grad[9], variant)
        return out

    @staticmethod
    def backward(ctx, dout):
        mask, qk, eq, ek, fp32, npar, heads, xgrad, variant = ctx.meta
        vals = ctx.saved_tensors
        single, cond, pair = vals[:3]
        params, saved, extra = list(vals[3 : 3 + npar]), list(vals[3 + npar : -1]), vals[-1]
        xg = extra if xgrad else None
        dx, dc, dpair, *pg = _bwd(
            single,
            cond,
            pair,
            mask,
            params,
            saved,
            dout.contiguous(),
            qk,
            eq,
            ek,
            fp32,
            heads,
            xg,
            variant,
        )
        if _native90(single.device):
            _activate_native_caller(force=True)
        return (
            dx,
            dc,
            dpair,
            None,
            None,
            None,
            None,
            None,
            None,
            pg.pop() if xgrad else None,
            None,
            *(g if ctx.needs_input_grad[11 + i] else None for i, g in enumerate(pg)),
        )


def _pad(single, cond, pair, mask, Ap, Lp):
    """The inputs padded to Ap samples of Lp rows (``block``): zero single / cond rows and samples, zero pair rows and columns,
    the padded keys masked out of every softmax. The padded queries and samples are computed and dropped, so their output
    gradient is zero and they add nothing to any gradient; the masked keys get P = 0, so they add nothing either."""
    F = torch.nn.functional
    A, _, L, _ = single.shape
    rows = (0, 0, 0, Lp - L, 0, 0, 0, Ap - A)
    crows = rows if cond.shape[0] == A else (0, 0, 0, Lp - L, 0, 0, 0, 0)  # a shared [1, 1, L, dc] conditioning has no samples to pad
    if Lp == L:  # samples only: the pair and the mask as they are (F.pad copied the pair, and its backward the pair gradient)
        return F.pad(single, rows), F.pad(cond, crows), pair, mask
    if mask is None:
        mask = torch.ones(1, L, device=single.device, dtype=torch.bool)
    return (
        F.pad(single, rows),
        F.pad(cond, crows),
        F.pad(pair, (0, 0, 0, Lp - L, 0, Lp - L)),
        F.pad(mask.to(torch.bool), (0, Lp - L), value=False),
    )


def block(module, single, cond, pair, mask=None, compute_dtype=None, extra=None):
    """One DiTBlock (attention + transition, both residuals) through the fused training path, in the precision
    ``operand_dtype`` picks. Call ``serves`` first. L is padded to a multiple of 128 and A to an even count where needed
    (``_pad``; a padded call returns the slice of the real samples and rows)."""
    a = module.attention
    qk = bool(a.use_qk_norm)
    eq = float(a.norm_query.effective_eps(torch.float32)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(torch.float32)) if qk else 0.0
    fp32 = operand_dtype(single, compute_dtype) is torch.float32
    A, _, L, _ = single.shape
    # the sm_100a forward pairs the samples; the H100 cores take any A (and the qualified odd-A buckets stay unpadded)
    # (B200 bf16 at A = 1: the -DQPAIR forward pairs query tiles instead, no padded sample -- sm100.forward)
    # (B200 fp32: the TF32 cores take an odd A themselves -- the last pair's missing partner is clamped and not stored)
    Ap = A if _sm90(single.device) or fp32 or A == 1 else A + A % 2
    Lp = -(-L // QM) * QM
    padded = (Ap, Lp) != (A, L)
    from foldjet_engine.kernels.conditioned_transition.cuda import VARIANT_DEFAULT
    from foldjet_engine.kernels.integrations import token_dit

    opts = token_dit.options(module)
    assert opts is not None  # serves() refused a block outside the algorithm
    cond0 = cond  # the caller's tensor: the native hooks attach to its autograd edge
    if SHARED_COND and _sm90(single.device) and A > 1 and cond.shape[0] == A and cond.stride(0) == 0 and opts == VARIANT_DEFAULT:
        cond = cond[:1]  # the conditioning shared by the samples (a stride-0 expand): every conditioning GEMM on L rows (_shared_cond)
    caller = (
        single,
        cond0,
        pair,
    )  # the caller's tensors: the native hooks attach to their autograd edges, not the pads'
    if padded:
        single, cond, pair, mask = _pad(single, cond, pair, mask, Ap, Lp)
        if extra is not None:
            extra = torch.nn.functional.pad(extra, (0, Lp - L, 0, Lp - L))
    params = _leaves(module, _names_for(qk, opts))
    out = _Block.apply(
        single,
        cond,
        pair.contiguous(),
        mask,
        qk,
        eq,
        ek,
        fp32,
        int(a.n_head),
        None if extra is None else extra.contiguous(),
        list(opts),
        *params,
    )
    if _native90(caller[0].device):
        # The result's edges now strongly own leaf AccumulateGrad nodes.
        # Installing before apply can attach hooks to a temporary weak owner.
        _native_caller_hooks(*caller, *module.parameters(), force=True)
    return out[:A, :, :L] if padded else out


__all__ = ["block", "serves"]
