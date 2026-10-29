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

``serves()`` is the whole gate: autograd on, the engine's kernel backend (implementation MINIWORLD), H100 or B200, a
bf16 or fp32 input (bf16 operands when the input or ``compute_dtype`` is bf16, else fp32), the token widths (cond 384 / pair
128 / transition n = 2) at a head layout of LAYOUTS (B200: 16 x 48, 24 x 32, 12 x 64 at d 768, 16 x 64 at d 1024; H100: 16 x 48
only; bf16 and fp32 alike), B == 1, a key mask of [B, L] or none (``DiTBlock`` hands an [A, B, L] mask shared by the samples
over as its [B, L] row), LayerNorm eps 1e-5. Anything else keeps the module path. settings.token_dit_train=0 turns it off.

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

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels._tf32 import tf32_matmul as _tf32

D, DC, DP, H, DH = 768, 384, 128, 16, 48  # the default layout; serves() takes LAYOUTS
#: (heads, d_single): 16 x 48, 24 x 32, 12 x 64 and 16 x 64 (d 1024), bf16 and fp32 (the TF32 cores are built per layout too)
LAYOUTS = ((16, 768), (24, 768), (12, 768), (16, 1024))
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
        and torch.cuda.get_device_capability(single.device) in ((9, 0), (10, 0))
        and extra.ndim == 2
        and tuple(extra.shape) == (single.shape[2], single.shape[2])
        and extra.dtype in (BF, torch.float32)
    ):
        return False   # the [L, L] logit term is added to the block's bias (H100 log2 units, B200 natural units); nowhere else
    if (not settings.current().token_dit_train) or not torch.is_grad_enabled():
        return False
    if settings.current().force_portable_kernels:
        return False
    a = module.attention
    # the engine's kernels (MINIWORLD resolve to them), as the sm_90 / sm_100 attention cores gate on
    from miniworld_engine.modules.dispatch import KernelBackend

    if a._backend != KernelBackend.PORTABLE:
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
    if not any(t.requires_grad for t in (single, cond, pair)) and not any(
        p.requires_grad for p in module.parameters()
    ):
        return False
    # any A and L >= 1: ``block`` pads L to a multiple of 128 and A to an even count (the kernels' tiles / sample pairs)
    if (
        single.ndim != 4
        or single.shape[1] != 1
        or single.shape[0] < 1
        or single.shape[2] < 1
    ):
        return False
    d = single.shape[-1]
    if tuple(cond.shape) != (*single.shape[:3], DC) or tuple(pair.shape) != (
        1,
        single.shape[2],
        single.shape[2],
        DP,
    ):
        return False
    if (a.n_head, d) not in LAYOUTS or module.transition.expand_a.weight.shape[
        0
    ] != 2 * d:
        return False
    if mask is not None and not (
        mask.ndim == 2 and tuple(mask.shape) == (1, single.shape[2])
    ):
        return False
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        a.ln_pair,
        module.transition.ada_ln_in.ln_in,
        module.transition.ada_ln_in.ln_cond,
    )
    return all(n.eps == EPS for n in norms)


def _f32(t):
    return t.detach().float().contiguous()


def operand_dtype(single, compute_dtype=None) -> torch.dtype:
    """The dtype of every GEMM operand / saved activation: bf16 when the input or ``compute_dtype`` is bf16, else fp32."""
    return BF if (single.dtype is BF or compute_dtype is BF) else torch.float32


def _mm32(a, b):
    """a @ b with an fp32 result (weight gradients): bf16 operands accumulate into fp32 output, fp32 ones already are.

    fp32 (TF32) weight gradients with few output tiles (dWo 768 x 768, dWg 1536 x 384, dWn 3072 x 384, dWsq 768 x 1536: under
    one wave of 128 x 128 tiles) and a long reduction (the A L rows) ran at 370-580 TF/s against ~740 for the wide ones: they are
    split over the rows into S <= 8 chunks as one batched GEMM, then summed."""
    if a.dtype is BF:
        return torch.mm(a, b, out_dtype=torch.float32)
    n, m = a.shape
    k = b.shape[1]
    tiles = -(-n // 128) * -(-k // 128)
    nsm = num_sms(a.device)
    s = 1
    if tiles < 128 and not _sm90(a.device) and b.is_contiguous() and a.t().is_contiguous():
        # (Hopper: the split costs ~440 us a step at A = 48, L768 -- measured dWo / dWg / dWn / dWsq 207 / 284 / 436 / 444 us as
        #  one GEMM against 319 / 321 / 548 / 627 us split in four)
        while s < 8 and tiles * s * 2 <= 2 * nsm and m % (2 * s * 128) == 0:
            s *= 2
    if s == 1:
        return torch.mm(a, b)
    return torch.bmm(
        a.t().reshape(s, m // s, n).transpose(1, 2), b.reshape(s, m // s, k)
    ).sum(0)


LOG2E = 1.4426950408889634
QS90 = (
    LOG2E / DH**0.5
)  # the sm_90a core's q scale: its logits are in log2 units (16 x 48 only)


def _sm90(dev) -> bool:
    """H100: the sm_90a attention cores and their operand conventions; B200 otherwise (``serves`` admits only these two)."""
    return torch.device(dev).type == "cuda" and torch.cuda.get_device_capability(
        dev
    ) == (9, 0)


def _key_mask(mask, A, L, perm=None):
    """The sm_90a cores' additive key mask, [A, L] fp32 (0 = attend), or None; ``perm``: its keys in that order (the TF32
    core's)."""
    if mask is None:
        return None
    m = mask.reshape(L) if perm is None else mask.reshape(L)[perm]
    return torch.where(m, 0.0, -1e30).to(torch.float32).expand(A, L).contiguous()


_PACKS: dict = {}


def _native_rpb():
    return 2


def _native90(dev):
    return _sm90(dev)


def _empty_uniform90(dev, A=None, L=None):
    return _native90(dev) and A == 5 and L in (384, 768)


def _owned_gemms90(single):
    """GEMMs whose captured scratch belongs to this graph, never a framework handle."""
    if not (
        _native90(single.device)
        and single.shape[0] == 5
        and single.shape[2] in (384, 768)
    ):
        return torch.mm, _mm32, torch.addmm
    from miniworld_engine.kernels.conditioned_transition.cuda.train import (
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


_NATIVE_CALLER_TASKS: set[int] = set()


def _activate_native_caller(*, force: bool = False):
    """Limit installed hooks to backward tasks that actually execute our block."""
    if not force or torch.compiler.is_compiling():
        return
    task = torch._C._current_graph_task_id()
    if task >= 0 and task not in _NATIVE_CALLER_TASKS:
        _NATIVE_CALLER_TASKS.add(task)
        torch.autograd.Variable._execution_engine.queue_callback(
            lambda: _NATIVE_CALLER_TASKS.discard(task)
        )


def _native_caller_output(output, *tensors, force: bool = False):
    """Bind accumulation after the native output node finishes backward."""
    if (
        torch.compiler.is_compiling()
        or not torch.is_grad_enabled()
        or not output.requires_grad
        or output.grad_fn is None
    ):
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
    if not force or torch.compiler.is_compiling():
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
                    if not overlap or (
                        dst.data_ptr() == src.data_ptr()
                        and dst.stride() == src.stride()
                    ):
                        if dst.is_contiguous() and src.is_contiguous():
                            kernels.caller_add90(dst, src)
                            return dst
                        if (
                            dst.ndim == 2
                            and src.ndim == 2
                            and dst.t().is_contiguous()
                            and src.t().is_contiguous()
                        ):
                            # Column-major matrix gradients need only metadata
                            # views; both operands retain identical element order.
                            kernels.caller_add90(dst.t(), src.t())
                            return dst
            elif func in (
                torch.ops.aten.zeros.default,
                torch.ops.aten.new_zeros.default,
            ):
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
            state.mode = (
                CallerMode()
                if torch._C._current_graph_task_id() in _NATIVE_CALLER_TASKS
                else None
            )
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
        elif (
            node.name() in ("ExpandBackward0", "AsStridedBackward0")
            and tensor.stride(0) == 0
        ):
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
    key = (native, *((t.data_ptr(), t._version) for t in P.values()))
    slot = _capture.scoped((next(iter(P.values())).data_ptr(), at))
    hit = None if slot is None else _PACKS.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    if native:
        from miniworld_engine.integrations.token_dit import QK_WEIGHTS, WEIGHTS
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

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
        W = dict(
            zip(keys, T.train_pack90(weights, qk, at is torch.float32), strict=True)
        )
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
    W = {
        "w1": w1,
        "w2": w2,
        "Wraw": Wraw,
        "Wn": torch.cat([Wraw[: 2 * D] * w1, Wraw[2 * D :] * w2]).to(
            at
        ),  # cond-LN weights folded
        "Wg": torch.cat(
            [g("attention.to_scale.weight"), g("transition.to_scale.weight")]
        )
        .detach()
        .to(at),
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
        "nq": _f32(g("attention.norm_query.weight"))
        if qk
        else torch.ones(DH, device=dev),
        "nk": _f32(g("attention.norm_key.weight"))
        if qk
        else torch.ones(DH, device=dev),
        "wp": wp,
        "Wb": Wb,
        "Wf": Wf,
        "Wf_at": Wf.to(at),
        "Wf_l2": (Wf * LOG2E).to(at),  # natural / log2 units
        "Wo": g("attention.to_out.weight").detach().to(at).contiguous(),
        "Wab": torch.cat(
            [g("transition.expand_a.weight"), g("transition.expand_b.weight")]
        )
        .detach()
        .to(at),
        "Wsq": g("transition.squeeze.weight").detach().to(at).contiguous(),
    }
    if slot is not None:
        _capture.prune(_PACKS)
        _PACKS[slot] = (key, W, tuple(P.values()))  # keeps the key tensors alive
    return W


_GSW: dict = {}


def _swiglu_gemm(xt, wab):
    """The sm_100a expand GEMM with the SwiGLU epilogue that also writes [a | b] (``gemm_swiglu2_sm100.cu -DSAVE_AB``) where
    it applies (bf16, B200, M >= its row threshold); None keeps cuBLAS + the SwiGLU rows."""
    from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

    if not gemm_swiglu.supported(xt, wab):
        return None
    idx = (
        xt.device.index if xt.device.index is not None else torch.cuda.current_device()
    )
    if idx not in _GSW:
        try:
            _GSW[idx] = gemm_swiglu.GemmSwiglu(
                idx, K=wab.shape[1], H=wab.shape[0] // 2, save_ab=True
            )
        except Exception as exc:  # a failed build keeps cuBLAS + the row pass
            import warnings

            warnings.warn(
                f"sm100 SwiGLU GEMM unavailable, keeping cuBLAS: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            _GSW[idx] = None
    return _GSW[idx]


def _names_for(qk):
    return ["attention." + n for n in ATT + (QKN if qk else ())] + [
        "transition." + n for n in TRN
    ]


def _own_cond(cond, fp32) -> bool:
    """fp32 path with an fp32 cond: the gate GEMM (and its weight gradient) reads cond itself, no operand copy is made. Not on
    the H100 native path, whose cond may be sample-expanded (stride 0): it reads cond through ``cond_prep90`` and keeps cbf."""
    return fp32 and cond.dtype is torch.float32 and not _native90(cond.device)


def _nd(params, qk, D):
    """(transition expansion n, pair channels of ln_pair) read off the block's parameters: expand_a is [n D, D], ln_pair [dp]."""
    P = dict(zip(_names_for(qk), params, strict=False))
    return P["transition.expand_a.weight"].shape[0] // D, P["attention.ln_pair.weight"].shape[0]


def _saved_like(single, cond, pair, fp32, qk=True, H=H, psb=False, n=2, dp=DP):
    """Shapes / dtypes of the forward's saved activations (the fake implementation and the contract of _fwd). The fp32 path
    saves no copy of cond (_own_cond), no copy of v (QK-norm: the core reads it from qkvg) and no LN(pair) (pair_bias_bwd
    recomputes it): those slots are empty [0, n] tensors. H100: q / k / v are always copies (the q pass scales q; the TF32 core's
    v^T is built from v) and no O slot (the cores write og), and fp32 keeps LN(pair) (the dWf / dpair GEMMs read it)."""
    A, _, L, D = single.shape
    M, R, e = A * L, L * L, single.new_empty
    f32, at = torch.float32, (torch.float32 if fp32 else BF)
    h100 = _sm90(single.device)
    copies = qk or h100
    return [
        e((M, D), dtype=f32),
        e((M, cond.shape[-1]), dtype=at),
        e((0 if _own_cond(cond, fp32) else M, cond.shape[-1]), dtype=at),
        e((M, 2), dtype=f32),
        e((M, 4 * D), dtype=at),
        e((M, 2 * D), dtype=at),
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
        e((0,), dtype=at) if psb else e((H, L, L), dtype=at),   # psb: the bias is an input, kept by the caller
        *([] if h100 else [e((M, D), dtype=f32)]),
        e((A, H, L), dtype=f32),
        e((M, D), dtype=at),
        e((M, D), dtype=at),
        e((M, D), dtype=f32),
        e((M, 2), dtype=f32),
        e((M, D), dtype=at),
        e((M, 2 * n * D), dtype=at),   # a | b of the SwiGLU: two [M, n D] halves
        e((M, D), dtype=at),
        e((0 if (psb or (fp32 and not h100) or _pair_fused(h100, fp32, psb, dp)) else R, dp), dtype=at),
        e((0 if psb else R, 2), dtype=f32),
    ]


def _pair_fused(h100, fp32, psb, dp) -> bool:
    """The H100 bf16 block builds its bias with the one-pass pair kernel (LN(pair) Wf^T on TF32 tensor cores, the extra term added in the same
    pass, no LN(pair) written) and differentiates it with its one-pass backward (dpair and the dWf partials): it keeps no LN(pair) either."""
    return h100 and not fp32 and not psb and dp == 128


def _fwd_fake(single, cond, pair, mask, params, qk, eq, ek, fp32, heads, psb=False, extra=None):
    n, dp = _nd(params, qk, single.shape[-1])
    return [
        torch.empty_like(single),
        *_saved_like(single, cond, pair, fp32, qk, heads, psb, n, dp),
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
    psb: bool = False,
    extra: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The block's forward: [out, *saved activations] (every output freshly allocated); ``fp32``: fp32 GEMM operands (on TF32
    tensor cores, whatever the caller's allow_tf32) and the TF32 attention kernels, else bf16. Autocast off: the precision is
    ``fp32``'s, and an enclosing bf16 autocast would otherwise turn the fp32 path's GEMM outputs bf16."""
    with _tf32(fp32), torch.autocast("cuda", enabled=False):
        return _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, heads, psb, extra)


def _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, H, psb=False, extra=None):
    mm, _mm32, addmm = _owned_gemms90(single)
    from miniworld_engine.kernels.augmented_attention import cuda as sm90
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

    A, _, L, D = single.shape
    DH = D // H
    nexp, dpw = _nd(params, qk, D)
    T = ext(
        D,
        DH,
        rpb=_native_rpb() if _native90(single.device) else 8,
        empty_uniform=_empty_uniform90(single.device, single.shape[0], single.shape[2]),
        dc=cond.shape[-1],
        n=nexp,
        dp=dpw,
    )
    names = _names_for(qk)
    P = dict(zip(names, params, strict=False))
    M, R, dev = A * L, L * L, single.device
    at = torch.float32 if fp32 else BF
    h100 = _sm90(dev)
    native = _native90(dev)
    copies = (
        qk or h100
    )  # q / k / v copies: QK-norm, or the sm_90a cores' pre-scaled q (see _saved_like)
    x = torch.empty(M, D, device=dev)  # the fp32 residual, written by adaln_a
    c2 = (
        None if native else cond.reshape(M, cond.shape[-1]).contiguous()
    )  # native: cond_prep90 reads cond as given
    own_c = _own_cond(cond, fp32)
    chat = torch.empty(M, cond.shape[-1], device=dev, dtype=at)
    cbf = None if own_c else torch.empty_like(chat)
    cst = torch.empty(M, 2, device=dev)
    if native:
        T.cond_prep90(cond, chat, cbf, cst, EPS)
    else:
        T.cond_prep(c2, chat, cbf, cst, EPS)
    W = _pack(P, qk, dev, at)
    Wn, Wg = W["Wn"], W["Wg"]
    G = mm(chat, Wn.t())  # [M, 4D] s1 | sh1 | s2 | sh2
    Gg = mm(c2 if own_c else cbf, Wg.t())  # [M, 2D] gate1 | gate2
    bs1, bs2, bg1, bg2 = W["bs1"], W["bs2"], W["bg1"], W["bg2"]
    xa = torch.empty(M, D, device=dev, dtype=at)
    xst = torch.empty(M, 2, device=dev)
    T.adaln_a(
        single.reshape(M, D).contiguous(), G, bs1, xa, xst, EPS, x
    )  # reads the input, writes x on the way
    Wqkvg, bqkvg = W["Wqkvg"], W["bqkvg"]
    qkvg = addmm(bqkvg, xa, Wqkvg.t())
    rqk = torch.empty(M, 2 * H, device=dev)
    nq, nk = W["nq"], W["nk"]
    if copies:  # B200 fp32: v is not copied, the core reads it as a column view of qkvg
        qn = torch.empty(M, D, device=dev, dtype=at)
        kn = torch.empty_like(qn)
        vc = None if (fp32 and not h100) else torch.empty_like(qn)
        T.qknorm(
            qkvg, nq, nk, qn, kn, vc, rqk, eq, ek, qk, (LOG2E / DH**0.5) if h100 else 1.0
        )  # H100: q scaled for the exp2 core
        vsave = vc if vc is not None else qkvg.new_empty(0, D)
        if vc is None:
            vc = qkvg[:, 2 * D : 3 * D]
    else:  # the core reads q / k / v as column views of qkvg: no copies
        qn, kn, vc = (qkvg[:, i * D : (i + 1) * D] for i in range(3))
    O = None
    if psb:
        # one bias per sample, produced by the caller (H100: head-major [H, A, L, L], log2 units; B200: sample-major [A, H, L, L],
        # natural units, the key mask folded in): there is no pair LayerNorm and no bias GEMM in the block, and nothing of either is
        # kept for the backward
        assert not fp32 and pair.shape == ((H, A, L, L) if h100 else (A, H, L, L)), "per-sample bias: the bf16 block"
        ph = pair.new_empty(0, dpw, dtype=at)
        pst = pair.new_empty(0, 2, dtype=torch.float32)
        bias = pair
    else:
        p2 = pair.reshape(R, dpw)
        pst = torch.empty(R, 2, device=dev)
    if psb:
        pass
    elif _pair_fused(h100, fp32, psb, dpw):
        ph = pair.new_empty(0, dpw, dtype=at)
        bias = torch.empty(H, L, L, device=dev, dtype=at)
        T.pair_bias(
            p2, (W["Wf"].float() * LOG2E).contiguous(), None, bias, pst, L, EPS,
            None if extra is None else extra.reshape(-1).float().contiguous(), LOG2E,
        )   # log2 units, rounded once to bf16
    elif h100 or not fp32:
        ph = torch.empty(R, dpw, device=dev, dtype=at)
        T.pair_ln(p2, ph, pst, EPS)
        # head-major bias, Wf = Wb diag(wp): natural units for sm_100a, log2 units (and key-permuted for the TF32 core) for sm_90a
        bias = mm(W["Wf_l2"] if h100 else W["Wf_at"], ph.t()).view(H, L, L)
        if extra is not None:   # an [L, L] logit term for every head, in the bias's units (H100 log2, B200 natural): one add
            bias.add_(extra.view(1, L, L), alpha=LOG2E if h100 else 1.0)
    else:  # B200 fp32: LN(pair) folded straight into the bias, fp32 FMAs, key mask as -inf; LN(pair) not kept
        ph = pair.new_empty(0, dpw, dtype=at)
        bias = torch.empty(H, L, L, device=dev)
        T.pair_bias(
            p2,
            W["Wf"],
            None if mask is None else mask.reshape(L).to(torch.bool).contiguous(),
            bias,
            pst,
            L,
            EPS,
        )
        if extra is not None:   # natural units; the masked key columns stay -inf
            bias.add_(extra.view(1, L, L))
    if psb and h100:
        km = T.train_keymask90(mask, qn, A, L, False)[0]
        og, LSE = sm90.forward_psb(qn, kn, vc, bias, km, A, L, qkvg, H, DH)  # sigmoid(g) o, no O
    elif psb:
        # B200: the AttentionPairBias cores (attn_fwd2 -DQPAIR, every sample its own bias planes), unsplit: A samples fill the SMs
        O, LSE = sm100.apb_forward(qn, kn, vc, bias.view(A * H, L, L), L, H, D)
        LSE = LSE.view(A, H, L)
        og = torch.empty(M, D, device=dev, dtype=at)
        T.gate_o(O, qkvg, og)
    elif h100:
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
            og, LSE = sm90.forward(
                qn, kn, vc, bias, km if native else _key_mask(mask, A, L), A, L, qkvg, H, DH
            )  # sigmoid(g) o, no O
    else:
        if not fp32 and mask is not None:
            bias.masked_fill_(~mask.reshape(1, 1, L), float("-inf"))
        O, LSE = (
            sm100.forward_tf32(qn, kn, vc, bias, A, L, H, DH)
            if fp32
            else sm100.forward(qn, kn, vc, bias, A, L, H, DH)
        )
        og = torch.empty(M, D, device=dev, dtype=at)
        T.gate_o(O, qkvg, og)
    Wo = W["Wo"]
    y = mm(og, Wo.t())
    x1 = torch.empty_like(x)
    xt = torch.empty(M, D, device=dev, dtype=at)
    x1st = torch.empty(M, 2, device=dev)
    T.res_adaln_b(x, y, Gg, bg1, G, bs2, x1, xt, x1st, EPS)
    Wab = W["Wab"]
    h = torch.empty(M, nexp * D, device=dev, dtype=at)
    gsw = _swiglu_gemm(xt, Wab)
    if gsw is not None:  # expand GEMM + SwiGLU, a | b saved on the way
        ab = torch.empty(M, 2 * nexp * D, device=dev, dtype=at)
        gsw(xt, Wab, h, ab)
    else:
        ab = mm(xt, Wab.t())  # [M, 2 * n D] a | b
        rows.swiglu_rows(ab, h)
    Wsq = W["Wsq"]
    z = mm(h, Wsq.t())
    out = torch.empty(M, D, device=dev, dtype=single.dtype)
    T.res_c(x1, z, Gg, bg2, out)
    # custom-op outputs may not alias each other: without QK-norm q / k / v are views of qkvg -- not outputs, recovered in _bwd
    qkv_out = [qn, kn, vsave] if copies else []
    if cbf is None:
        assert (
            c2 is not None
        )  # cbf is None only when _own_cond (never native), which built c2
        cbf = c2.new_empty(0, cond.shape[-1], dtype=at)
    # the sm_90a cores write og and the backward needs no O: H100 keeps none (not even a placeholder, which AOT mis-strides)
    return [
        out.view(single.shape),
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
        *qkv_out,
        bias.new_empty(0) if psb else bias,
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
    ]


def _tf32_attn_bwd(T, qs, k, v, dob, bias_p, mask, lse, dd, A, L):
    """The H100 fp32 block's attention backward on the TF32 CUDA core: dQ (for the unscaled q), dK, dV [A L, 768] fp32 and dbias
    [16, L, L] fp32 in natural key order. qs: q in exp2 units; bias_p: the forward's key-permuted bias (log2 units). The
    core takes q, k, dO transposed as well (tiled transposes, ``T.transpose``), and its bias / LSE / D in the permuted orders."""
    from miniworld_engine.kernels.augmented_attention import cuda as sm90
    from miniworld_engine.kernels.conditioned_transition import cuda as rows

    perm = rows.key_perm(L, qs.device)
    inv = torch.argsort(perm)
    nh_ = bias_p.shape[0]
    dh_ = qs.shape[1] // nh_

    def tr(t):
        out = t.new_empty(t.shape[1], t.shape[0])
        T.transpose(t, out)
        return out

    bias_tp = (
        bias_p[:, perm][..., inv].transpose(1, 2).contiguous()
    )  # [h, key, q position]
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


def _native_attn_bwd90(T, qs, k, v, dob, bias, mask, lse, dd, A, L, fp32):
    from miniworld_engine.kernels.augmented_attention import cuda as sm90

    H = bias.shape[0]  # this layout's heads (the module-level H is the default 16)
    DH = qs.shape[1] // H
    tf32_defs = sm90.tf32_defs(H, DH)
    bf_defs = sm90.fwd_defs(H, DH)  # attn_fwd / dq / dkv / dqb build defines of this layout (none at 16 x 48)

    dq, dk, dv = (torch.empty_like(qs, dtype=torch.float32) for _ in range(3))
    db = (
        None
        if (not fp32 and A % 3 == 0)
        else torch.empty((A, H, L, L), device=qs.device, dtype=torch.float32)
    )
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
        sm90._ext("attn_dkv", ("DBIAS=0", *bf_defs, *empty_defs)).attn_dkv(
            qs, k, v, dob, bias, km, lse, dd, dk, dv, dbn
        )
        return dq, dk, dv, dbn.to(torch.bfloat16)
    else:
        sm90._ext(
            "attn_dq", (*(() if L % 192 == 0 else ("NWG=2",)), *sm90.dq_defs(H, DH), *empty_defs)
        ).attn_dq(qs, k, v, dob, bias, km, lse, dd, dq)
        sm90._ext("attn_dkv", ("DB_PER_SAMPLE=1", *bf_defs, *empty_defs)).attn_dkv(
            qs, k, v, dob, bias, km, lse, dd, dk, dv, db
        )
    return dq, dk, dv, T.train_db_reduce90(db, fp32)


def _native_final90(
    T, names, params, W, allp, dWn, dWf, dWqkvg, dWo, dWg, dWab, dWsq, qk
):
    from miniworld_engine.integrations.token_dit import QK_WEIGHTS, WEIGHTS

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
    return [result[n] for n in names]


def _bwd_fake(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads, psb=False, extra=None):
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
    psb: bool = False,
    extra: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The block's backward: [d single, d cond, d pair, *d params] in the inputs' dtypes (autocast off, as ``_fwd``); with
    ``psb`` the third is d bias [H, A, L, L]."""
    with _tf32(fp32), torch.autocast("cuda", enabled=False):
        return _bwd_body(
            single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads, psb, extra
        )


def _xgrad(DB, extra, H, L):
    """d extra [L, L] (the extra term enters every head's logits once, in natural units: the sum of dbias over the heads) as a
    one-element list, or none when the caller does not differentiate it."""
    if extra is None:
        return []
    return [DB.reshape(H, L, L).sum(0, dtype=torch.float32).to(extra.dtype)]


def _psb_attn_bwd(T, qs, k, v, dob, mask, bias, lse, dd, A, L, H, DH):
    """The per-sample-bias attention backward: dQ, dK, dV [A L, 768] fp32 and dbias [H, A, L(query), L(key)] bf16, the layout
    and units of the bias it belongs to. Every sample's dbias has one writer (the dQ pass), so no reduction and no atomics."""
    from miniworld_engine.kernels.augmented_attention import cuda as sm90

    km = T.train_keymask90(mask, qs, A, L, False)[0]
    return sm90.backward_psb(qs, k, v, dob, bias, km, lse, dd, A, L, H, DH)


def _bwd_body(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, H, psb=False, extra=None):
    mm, mm32, _addmm = _owned_gemms90(single)
    from miniworld_engine.kernels.augmented_attention import cuda as sm90
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

    D = single.shape[-1]
    DH = D // H
    nexp, dpw = _nd(params, qk, D)
    T = ext(
        D,
        DH,
        rpb=_native_rpb() if _native90(single.device) else 8,
        empty_uniform=_empty_uniform90(single.device, single.shape[0], single.shape[2]),
        dc=cond.shape[-1],
        n=nexp,
        dp=dpw,
    )
    h100 = _sm90(single.device)
    saved = list(saved)
    if not (
        qk or h100
    ):  # the forward's q / k / v: column views of qkvg (see _fwd_body)
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
    ) = saved
    if (
        vc.numel() == 0
    ):  # B200 fp32 + QK-norm: v as the forward's core read it, a column view of qkvg
        vc = qkvg[:, 2 * D : 3 * D]
    names = _names_for(qk)
    at = torch.float32 if fp32 else BF
    W = _pack(dict(zip(names, params, strict=False)), qk, single.device, at)
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
    nq, nk, bs1, bs2, bg1, bg2, w1, w2, wp, Wb = (
        W[k] for k in ("nq", "nk", "bs1", "bs2", "bg1", "bg2", "w1", "w2", "wp", "Wb")
    )
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    sdt, cdt, pdtype = single.dtype, cond.dtype, pair.dtype
    native = _native90(dev)
    c2, p2 = (
        (None if native else cond.reshape(M, cond.shape[-1]).contiguous()),
        None if psb else pair.reshape(R, dpw),
    )  # native: cond_bwd90 reads cond as given
    if psb:
        bias = pair  # the bias is an input here, not a saved activation
    dout = dout.reshape(M, D).to(sdt).contiguous()
    # per-block column sums: bg2 bs2 bg1 bs1 bq | dwq dwk (per head); rows: the row kernels' block count (T.part_rows)
    allp = torch.empty(T.part_rows(M, fp32), 7 * D, device=dev)
    part = [allp[:, i * D : (i + 1) * D] for i in range(5)]
    pwqk = allp[:, 5 * D :]
    dG = torch.empty(M, 4 * D, device=dev, dtype=at)
    dGg = torch.empty(M, 2 * D, device=dev, dtype=at)
    dz = torch.empty(M, D, device=dev, dtype=at)
    T.res_c_bwd(dout, z, Gg, bg2, dz, dGg, part[0])
    dh = mm(dz, Wsq)  # [M, n D]
    dab = torch.empty_like(ab)
    h = torch.empty_like(dh)
    T.swiglu_bwd(dh, ab, dab, h)
    dWsq = mm32(dz.t(), h)
    # fp32: the dxt / dxa GEMMs write straight into their column blocks of dG (the row kernels then copy nothing); the H100 owned
    # GEMMs return a fresh tensor, which the row kernels copy as before
    direct = fp32 and mm is torch.mm
    dxt = torch.mm(dab, Wab, out=dG[:, 3 * D :]) if direct else mm(dab, Wab)
    dWab = mm32(dab.t(), xt)
    dx1 = torch.empty(M, D, device=dev)
    dy = torch.empty(M, D, device=dev, dtype=at)
    T.res_adaln_b_bwd(
        dout, dxt, x1, x1st, G, bs2, Gg, bg1, y, dx1, dy, dG, dGg, part[1], part[2]
    )
    dog = mm(dy, Wo)
    dWo = mm32(dy.t(), og)
    dob = torch.empty(M, D, device=dev, dtype=at)
    dd = torch.empty(A, H, L, device=dev)
    dqkvg = torch.empty(M, 4 * D, device=dev, dtype=at)
    T.gate_o_bwd(
        dog, og if h100 else O, qkvg, dob, dd, dqkvg, L, h100
    )  # H100: D and dg from og (dO o = dog og)
    if psb and h100:
        DQ, DK, DV, DB = _psb_attn_bwd(T, qn, kn, vc, dob, mask, bias, LSE, dd, A, L, H, DH)
    elif psb:
        # B200: attn_dkv / attn_dqb -DBPS (dQ per key chunk, dK / dV per query chunk: summed here in chunk order), dbias
        # [A H, L, L] fp32 natural units, the bias's own layout
        DQ, DK, DV, DB = sm100.apb_backward(
            qn, kn, vc, dob, bias.view(A * H, L, L), LSE.reshape(A * H, L), dd.view(A * H, L), L, H, D
        )
        DQ, DK, DV = DQ.sum(0), DK.sum(0), DV.sum(0)
    elif native:
        DQ, DK, DV, DB = _native_attn_bwd90(
            T, qn, kn, vc, dob, bias, mask, LSE, dd, A, L, fp32
        )
    elif h100 and fp32:
        DQ, DK, DV, DB = _tf32_attn_bwd(T, qn, kn, vc, dob, bias, mask, LSE, dd, A, L)
    elif h100:  # dq for the unscaled q, dbias natural units
        DQ, DK, DV, DB = sm90.backward(
            qn, kn, vc, dob, bias, _key_mask(mask, A, L), LSE, dd, A, L, H, DH
        )
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
            )
            if fp32
            else sm100.backward(qn, kn, vc, dob, bias, LSE, dd, A, L, H, DH)
        )
    T.qknorm_bwd(DQ, DK, DV, qkvg, rqk, nq, nk, dqkvg, part[4], pwqk, qk)
    dxa = torch.mm(dqkvg, Wqkvg, out=dG[:, D : 2 * D]) if direct else mm(dqkvg, Wqkvg)
    dWqkvg = mm32(dqkvg.t(), xa)
    dx = torch.empty(M, D, device=dev, dtype=sdt)
    T.adaln_a_bwd(dxa, x, xst, G, bs1, dx1, dx, dG, part[3])
    if not native:
        sums = allp.sum(0) if qk else allp[:, : 5 * D].sum(0)
        pb = sums[: 5 * D].view(5, D)
        dwqk = (
            sums[5 * D :].view(2, H, DH).sum(1).reshape(-1)
            if qk
            else torch.zeros(2 * DH, device=dev)
        )
    dchat = mm(dG, Wn)
    dWn = mm32(dG.t(), chat)
    dcg = mm(dGg, Wg)
    dWg = mm32(dGg.t(), c2 if cbf.numel() == 0 else cbf)
    dc = torch.empty(M, cond.shape[-1], device=dev, dtype=cdt)
    if native:
        T.cond_bwd90(dchat, dcg, cond, cst, dc)
        if psb:
            # the bias projection's weights belong to the producer of the biases, which differentiates them itself
            dWf = torch.zeros(H, dpw, device=dev)
        elif _pair_fused(h100, fp32, psb, dpw):
            dpair = torch.empty(R, dpw, device=dev, dtype=pdtype)
            part = torch.empty(T.pair_grid(R), H * dpw, device=dev)
            T.pair_bias_bwd(DB.reshape(H, R).contiguous(), p2, pst, W["Wf"], dpair, part)   # LN(pair) recomputed; dpair and the dWf partials in one pass
            dWf = part.sum(0).view(H, dpw)
        else:
            dbv = DB.view(H, R)
            dWf = mm32(dbv, ph)
            dph = mm32(dbv.t(), W["Wf_at"])
            dpair = torch.empty(R, dpw, device=dev, dtype=pdtype)
            T.pair_ln_bwd(dph, p2, pst, dpair)
        outs = _native_final90(
            T, names, params, W, allp, dWn, dWf, dWqkvg, dWo, dWg, dWab, dWsq, qk
        )
        return [
            dx.view(A, 1, L, D),
            dc.view(cond.shape),
            DB.view(pair.shape) if psb else dpair.view(1, L, L, dpw),
            *outs,
            *_xgrad(DB, extra, H, L),
        ]
    T.cond_bwd(dchat, dcg, c2, cst, dc)
    dWu = torch.empty_like(dWn)
    dw12 = torch.zeros(2, cond.shape[-1], device=dev)
    Ws1, Wb1, Ws2, Wb2 = (Wraw[i * D : (i + 1) * D] for i in range(4))
    T.unfold_lnw(dWn, Ws1, Wb1, Ws2, Wb2, w1, w2, dWu, dw12)
    dpair = torch.empty(R, dpw, device=dev, dtype=pdtype)
    if psb:
        # B200: the bias gradient in the bias's layout; ln_pair / to_bias belong to the bias's producer (zero here)
        dpair = DB.to(pdtype)
        dWf = torch.zeros(H, dpw, device=dev)
    elif (
        fp32 and not h100
    ):  # B200 fp32: LN(pair) recomputed: dpair and the dWf partials in one pass
        pwf = torch.empty(T.pair_grid(R), H * dpw, device=dev)
        T.pair_bias_bwd(DB.view(H, R), p2, pst, Wf, dpair, pwf)
        dWf = pwf.sum(0).view(H, dpw)
    elif _pair_fused(h100, fp32, psb, dpw):
        part = torch.empty(T.pair_grid(R), H * dpw, device=dev)
        T.pair_bias_bwd(DB.reshape(H, R).contiguous(), p2, pst, Wf, dpair, part)
        dWf = part.sum(0).view(H, dpw)
    else:
        dbv = DB.reshape(H, R).to(
            at
        )  # masked key columns are 0 (P = 0); H100: may be a transposed view
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
        "transition.expand_b.weight": dWab[nexp * D :],
        "transition.squeeze.weight": dWsq,
    }
    # custom-op outputs may not alias each other: the gradients above are views of a few shared buffers, so each leaves as
    # its own tensor in the parameter's dtype -- one multi-tensor copy kernel for all of them (23 copies were ~45 us)
    outs = [torch.empty(p.shape, device=dev, dtype=p.dtype) for p in params]
    torch._foreach_copy_(
        outs, [grads[n].reshape(p.shape) for n, p in zip(names, params, strict=False)]
    )
    return [
        dx.view(A, 1, L, D),
        dc.view(cond.shape),
        dpair.view(pair.shape) if psb else dpair.view(1, L, L, dpw),
        *outs,
        *_xgrad(DB, extra, H, L),
    ]


class _Block(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, cond, pair, mask, qk, eq, ek, fp32, heads, psb, extra, *params):
        out, *saved = _fwd(
            single, cond, pair, mask, list(params), qk, eq, ek, fp32, heads, psb, extra
        )
        ctx.save_for_backward(single, cond, pair, *params, *saved, extra)
        ctx.meta = (mask, qk, eq, ek, fp32, len(params), heads, psb, extra is not None and ctx.needs_input_grad[10])
        return out

    @staticmethod
    def backward(ctx, dout):
        mask, qk, eq, ek, fp32, npar, heads, psb, xgrad = ctx.meta
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
            psb,
            xg,
        )
        if _native90(single.device) and not psb:
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
            None,
            pg.pop() if xgrad else None,
            *(g if ctx.needs_input_grad[11 + i] else None for i, g in enumerate(pg)),
        )


def _pad(single, cond, pair, mask, Ap, Lp):
    """The inputs padded to Ap samples of Lp rows (``block``): zero single / cond rows and samples, zero pair rows and columns,
    the padded keys masked out of every softmax. The padded queries and samples are computed and dropped, so their output
    gradient is zero and they add nothing to any gradient; the masked keys get P = 0, so they add nothing either."""
    F = torch.nn.functional
    A, _, L, _ = single.shape
    rows = (0, 0, 0, Lp - L, 0, 0, 0, Ap - A)
    if mask is None:
        mask = torch.ones(1, L, device=single.device, dtype=torch.bool)
    return (
        F.pad(single, rows),
        F.pad(cond, rows),
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
    Ap = A if _sm90(single.device) else A + A % 2
    Lp = -(-L // QM) * QM
    padded = (Ap, Lp) != (A, L)
    caller = (
        single,
        cond,
        pair,
    )  # the caller's tensors: the native hooks attach to their autograd edges, not the pads'
    if padded:
        single, cond, pair, mask = _pad(single, cond, pair, mask, Ap, Lp)
        if extra is not None:
            extra = torch.nn.functional.pad(extra, (0, Lp - L, 0, Lp - L))
    params = [module.get_parameter(n) for n in _names_for(qk)]
    out = _Block.apply(
        single, cond, pair.contiguous(), mask, qk, eq, ek, fp32, int(a.n_head), False,
        None if extra is None else extra.contiguous(), *params
    )
    if _native90(caller[0].device):
        # The result's edges now strongly own leaf AccumulateGrad nodes.
        # Installing before apply can attach hooks to a temporary weak owner.
        _native_caller_hooks(*caller, *module.parameters(), force=True)
    return out[:A, :, :L] if padded else out


PSB_COND = (128, 256, 384, 768)  # conditioning widths the row kernels build for (a multiple of 128)


def _serves_psb_core(module, single, cond, mask, compute_dtype=None) -> bool:
    """Everything ``serves_psb`` asks except the bias tensor itself: the settings, the H100 or B200 bf16 block, one batch row, the
    widths (H100: cond in ``PSB_COND``, transition n = 2 or 4, pair channels of ``ln_pair`` 128 or 256; B200: the token widths, cond
    384, n = 2, ``ln_pair`` 128; a head layout of LAYOUTS), LayerNorm eps, and a key mask that is one [1, L] row or a row per sample
    [A, L]."""
    if not settings.current().token_dit_train:
        return False
    if settings.current().force_portable_kernels:
        return False
    a = module.attention
    from miniworld_engine.modules.dispatch import KernelBackend

    if a._backend != KernelBackend.PORTABLE or not single.is_cuda:
        return False
    cap = torch.cuda.get_device_capability(single.device)
    if cap not in ((9, 0), (10, 0)):
        return False
    if single.dtype not in (BF, torch.float32) or operand_dtype(single, compute_dtype) is torch.float32:
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[2] == 0:
        return False
    A, _, L, d = single.shape
    b200 = cap == (10, 0)
    if tuple(cond.shape[:3]) != (A, 1, L) or cond.shape[3] not in ((DC,) if b200 else PSB_COND):
        return False
    if (a.n_head, d) not in LAYOUTS or module.transition.expand_a.weight.shape[0] not in ((2 * d,) if b200 else (2 * d, 4 * d)):
        return False
    if a.ln_pair.weight.numel() not in ((DP,) if b200 else (128, 256)):
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) in ((1, L), (A, L))):
        return False
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        module.transition.ada_ln_in.ln_in,
        module.transition.ada_ln_in.ln_cond,
        *((a.ln_pair,) if b200 else ()),   # B200 batched: the pair's CUDA LayerNorm runs at eps 1e-5
    )
    return all(n.eps == EPS for n in norms)


def serves_psb(module, single, cond, bias, mask, compute_dtype=None) -> bool:
    """Can ``block_psb`` run this call: the H100 bf16 block with an attention bias of its own per sample, bias [H, A, L, L]
    (log2 units, as the block's bias GEMM leaves it). Any L (``block_psb`` pads to the kernels' tile), one batch row. Without
    autograd it runs the same forward (the saved activations are freed on return)."""
    if not _serves_psb_core(module, single, cond, mask, compute_dtype):
        return False
    A, _, L, _ = single.shape
    return tuple(bias.shape) == (module.attention.n_head, A, L, L) and bias.dtype is BF and bias.is_contiguous()


def block_psb(module, single, cond, bias, mask=None, compute_dtype=None):
    """One DiTBlock whose attention bias is its own per sample: bias [H, A, L, L] bf16 in log2 units (what a stacked bias GEMM
    leaves), the key mask folded in as an additive row, not into the bias. The block differentiates everything but the bias's
    producer: its gradient comes back in the bias's layout, and the block's ``ln_pair`` / ``to_bias`` weights get none from
    here (they are the producer's). Any L: a length that is not a multiple of the kernels' 128-row tile is padded here (zero
    rows, zero bias, the padded keys masked out of every softmax) and the result sliced back, as ``block`` does. Call
    ``serves_psb`` first. B200: the bias is re-laid sample-major in natural units by differentiable ops (``_block_psb_b200``)."""
    if not _sm90(single.device):
        return _block_psb_b200(module, single, cond, bias.permute(1, 0, 2, 3) * (1.0 / LOG2E), mask)
    a = module.attention
    qk = bool(a.use_qk_norm)
    eq = float(a.norm_query.effective_eps(torch.float32)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(torch.float32)) if qk else 0.0
    params = [module.get_parameter(n) for n in _names_for(qk)]
    _, _, L, _ = single.shape
    Lp = -(-L // QM) * QM
    if mask is None:
        mask = torch.ones(1, L, device=single.device, dtype=torch.bool)
    mask = mask.to(torch.bool).contiguous()
    if Lp != L:
        F = torch.nn.functional
        rows = (0, 0, 0, Lp - L)
        single, cond = F.pad(single, rows), F.pad(cond, rows)
        bias = F.pad(bias, (0, Lp - L, 0, Lp - L))
        mask = F.pad(mask.to(torch.bool), (0, Lp - L), value=False)
    out = _Block.apply(
        single, cond, bias.contiguous(), mask, qk, eq, ek, False, int(a.n_head), True, None, *params
    )
    return out[:, :, :L] if Lp != L else out


#: B200 masked keys in the per-sample bias (natural units): far enough below any real logit that exp gives exactly 0 in fp32, small
#: enough that attn_fwd2's one-FMA exponent ((t - m) log2 e) stays exact to well under one unit when every key of a sample is
#: masked (at -1e30 the FMA's rounding error alone is ~1e23 and such a sample's output is NaN): that sample then attends uniformly
#: and stays finite
NEG = -1e4


def _block_psb_b200(module, single, cond, bias, mask=None):
    """The B200 per-sample-bias block: ``bias`` [A, H, L, L] in natural units (any float dtype, differentiable: its producer's ops
    get the gradient), ``mask`` [1, L] / [A, L] or None. L is padded to the kernels' 128-row tile (zero rows, zero bias, the padded
    keys masked); the key mask is folded into the bf16 bias (``NEG``, a zero gradient there) that the AttentionPairBias cores read,
    sample-major (attn_fwd2 -DQPAIR, attn_dkv / attn_dqb -DBPS)."""
    F = torch.nn.functional
    a = module.attention
    qk = bool(a.use_qk_norm)
    eq = float(a.norm_query.effective_eps(torch.float32)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(torch.float32)) if qk else 0.0
    params = [module.get_parameter(n) for n in _names_for(qk)]
    A, _, L, _ = single.shape
    Lp = -(-L // QM) * QM
    if Lp != L:
        rows = (0, 0, 0, Lp - L)
        single, cond = F.pad(single, rows), F.pad(cond, rows)
        bias = F.pad(bias, (0, Lp - L, 0, Lp - L))
        if mask is None:
            mask = torch.ones(1, L, device=single.device, dtype=torch.bool)
        mask = F.pad(mask.to(torch.bool), (0, Lp - L), value=False)
    if mask is not None:
        bias = bias.masked_fill(~mask.to(torch.bool).reshape(-1, 1, 1, Lp), NEG)
    out = _Block.apply(
        single, cond, bias.to(BF).contiguous(), None, qk, eq, ek, False, int(a.n_head), True, None, *params
    )
    return out[:, :, :L] if Lp != L else out


class _SampleBias(torch.autograd.Function):
    """B200 batched proteins: every sample's bias from its own pair, sample-major [A, H, L, L] bf16 in natural units, as the
    AttentionPairBias cores read it -- LN(pair) by the CUDA row kernel (``pair_ln``, its statistics kept), Wf = to_bias.weight
    diag(ln_pair.weight) applied by one batched GEMM, the masked keys (``keep`` [A | 1, L] False) set to ``NEG`` in place (their
    gradient is zero: P = 0 there). Backward: dLN(pair) and dWf as batched GEMMs with fp32 results, the LayerNorm backward
    (``pair_ln_bwd``), the weights' gradients from dWf."""

    @staticmethod
    def forward(ctx, pair, ln_w, wb, keep):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

        A, L, dp = pair.shape[0], pair.shape[1], pair.shape[3]
        T = ext(D, DH)
        R = A * L * L
        ph = torch.empty(R, dp, device=pair.device, dtype=BF)
        pst = torch.empty(R, 2, device=pair.device, dtype=torch.float32)
        T.pair_ln(pair.view(R, dp), ph, pst, EPS)
        wf = (wb.float() * ln_w.float()[None, :]).to(BF)
        bias = torch.matmul(wf, ph.view(A, L * L, dp).transpose(1, 2)).view(A, -1, L, L)   # [A, H, L, L]
        if keep is not None:
            bias.masked_fill_(~keep.to(torch.bool).reshape(-1, 1, 1, L), NEG)
        ctx.save_for_backward(pair, ph, pst, wf, ln_w, wb)
        return bias

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

        pair, ph, pst, wf, ln_w, wb = ctx.saved_tensors
        A, L, dp = pair.shape[0], pair.shape[1], pair.shape[3]
        H, R = wf.shape[0], A * L * L
        g3 = g.reshape(A, H, L * L).to(BF)
        dph = torch.bmm(g3.transpose(1, 2), wf.expand(A, H, dp), out_dtype=torch.float32)   # [A, L L, dp]
        # dWf = sum_a g_a LN(pair)_a: a [H, dp] result over A L^2 rows; as A bmm's of K = L^2 cuBLAS ran it at one tile per sample
        # (172 us at A4 L384, measured), so the rows are cut into s chunks per sample -- A s small GEMMs, summed in order
        sc = next(c for c in (64, 32, 16, 8, 4, 2, 1) if (L * L) % c == 0)
        k = L * L // sc
        gs = g3.view(A, H, sc, k).transpose(1, 2).reshape(A * sc, H, k)
        dwf = torch.bmm(gs, ph.view(A * sc, k, dp), out_dtype=torch.float32).sum(0)     # [H, dp]
        dpair = torch.empty_like(pair)
        ext(D, DH).pair_ln_bwd(dph.view(R, dp), pair.view(R, dp), pst, dpair.view(R, dp))
        d_lnw = (dwf * wb.float()).sum(0).to(ln_w.dtype)
        d_wb = (dwf * ln_w.float()[None, :]).to(wb.dtype)
        return dpair, d_lnw, d_wb, None


def _batched_mask(mask, B, L):
    """A [B, L] / [A=1, B, L] key mask of B independent proteins as the [B, L] row per sample ``block_psb`` takes, or None."""
    if mask is None:
        return None
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return mask if mask.ndim == 2 and tuple(mask.shape) == (B, L) else False


def serves_batched(module, single, cond, pair, mask, compute_dtype=None) -> bool:
    """A DiTBlock call over B > 1 independent proteins, each with its own pair: ``single`` [1, B, L, d], ``cond`` [1, B, L, dc],
    ``pair`` [B, L, L, dp], a key mask [B, L] (or none). They run as B samples of ``block_psb``, each with the attention bias its own
    pair gives."""
    if single.ndim != 4 or single.shape[0] != 1 or single.shape[1] < 2:
        return False
    B, L = single.shape[1], single.shape[2]
    key = _batched_mask(mask, B, L)
    if key is False or cond.ndim != 4 or tuple(cond.shape[:3]) != (1, B, L):
        return False
    if pair.ndim != 4 or tuple(pair.shape[:3]) != (B, L, L) or pair.shape[3] != module.attention.ln_pair.weight.numel():
        return False
    return _serves_psb_core(module, single.reshape(B, 1, L, -1), cond.reshape(B, 1, L, -1), key, compute_dtype)


def block_batched(module, single, cond, pair, mask=None, compute_dtype=None):
    """The fused block over B independent proteins (see ``serves_batched``). Each protein's attention bias is read off its pair by
    differentiable torch ops -- LayerNorm, the head projection, log2 units, head-major bf16 -- so the pair, ``ln_pair`` and
    ``to_bias`` get their gradients from autograd, and everything else from the fused block."""
    import torch.nn.functional as F

    a = module.attention
    B, L = single.shape[1], single.shape[2]
    dp = pair.shape[-1]
    if not _sm90(single.device):
        # B200: the bias straight in the cores' layout (``_SampleBias``: the pair's CUDA LayerNorm, one batched GEMM, the key mask
        # filled in place); a length off the 128-row tile is padded / masked by ``_block_psb_b200``
        key = _batched_mask(mask, B, L)
        whole = L % QM == 0
        bias = _SampleBias.apply(pair.contiguous(), a.ln_pair.weight, a.to_bias.weight, key if whole else None)
        out = _block_psb_b200(
            module, single.reshape(B, 1, L, -1), cond.reshape(B, 1, L, -1), bias, None if whole else key
        )
        return out.reshape(1, B, L, -1)
    bias = F.linear(
        F.layer_norm(pair, (dp,), a.ln_pair.weight.to(pair.dtype), None, a.ln_pair.eps), a.to_bias.weight.to(pair.dtype)
    )
    bias = (bias * LOG2E).permute(3, 0, 1, 2).contiguous().to(BF)  # [H, B, L, L]
    out = block_psb(
        module, single.reshape(B, 1, L, -1), cond.reshape(B, 1, L, -1), bias, _batched_mask(mask, B, L), compute_dtype
    )
    return out.reshape(1, B, L, -1)


__all__ = ["block", "block_batched", "block_psb", "serves", "serves_batched", "serves_psb"]
