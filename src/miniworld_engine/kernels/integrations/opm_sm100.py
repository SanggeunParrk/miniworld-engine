"""OuterProductMean (groups 1) on the B200 (sm_100a): the ``cuda`` backend's route at capability (10, 0).

The stages of the H100 integration (`opm_train.py`) on tcgen05 / TMEM / TMA kernels (`csrc/sm100/opm_sm100.cu`, MiniWorld's
d_msa 64 / d_pair 128; `csrc/sm100/opm_wide_sm100.cu`, every other width and the projection biases):

Forward    prologue      LayerNorm + both projections (+ RoseTTAFold3's projection biases) x mask -> A2 [(i, c), s], BT [(j, e), s]
                         (+ the LayerNorm statistics [N, S, 2] in training, the mask as bit rows)
           cuBLAS        O = A2 . BT^T, the grouped outer product [(i, c), (j, e)]
           counts        the pair divisor table [N, N] fp32: max(n_ij, 1) off the bit rows (`opm_norm`), or n_ij + count_offset
                         (AlphaFold 3's exact divisor; one cuBLAS GEMM of the mask with fp32 accumulation: exact integer counts)
           epilogue      z = (O / n) Wo^T + b or (O Wo^T + b) / n (`after`, the division after the projection), + the pair residual,
                         the interchain mask (chain ids: a pair of two chains gets no update)
Backward   dgrad         dO = (dz / n) Wo in the grouped layout, dz / n, dbias (CTA pairs, `opm_wdgrad`, every c_z)
           cuBLAS        dA = BT^T dO^T, dB = A2^T dO; dWo beside them on a side stream (`opm_dwo` / `opm_wdwo`)
           prologue bwd  dm, dWa, dWb, dgamma, dbeta (+ the projection-bias gradients: the masked column sums of dA / dB)

A length that is not a multiple of 64 / a depth that is not a multiple of 128 is padded with masked rows (no count, no product);
the pair output is cut back. A batch of MSA stacks runs one stack at a time. bf16 activations, fp32 (master) or bf16 parameters.

The kernels and their measurements come from the B200 branch (``origin/b200/ladder-final``: ``docs/gpus/b200/opm/opm.md``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from miniworld_engine.kernels import _cast_cuda
from miniworld_engine.kernels._compile import opaque

CM, CH, CZ = 64, 32, 128
D_MSA, D_PAIR = (64, 128, 256), (128, 256, 384, 512)
_EXT: dict[str, Any] = {}


def refusal(msa: torch.Tensor, d_msa: int, d_hidden: int, d_pair: int) -> str | None:
    """None if the B200 kernels serve this call (shape / dtype / device), else why not."""
    if not msa.is_cuda or torch.cuda.get_device_capability(msa.device) != (10, 0):
        return "the sm_100a OuterProductMean kernels run on a B200"
    if msa.dtype != torch.bfloat16:
        return f"the kernels are bf16, got {msa.dtype}"
    if d_hidden != CH or d_msa not in D_MSA or d_pair not in D_PAIR:
        return f"the sm_100a kernels serve d_hidden {CH}, d_msa in {D_MSA}, d_pair in {D_PAIR}; got ({d_msa}, {d_hidden}, {d_pair})"
    if msa.dim() != 4 or msa.shape[1] == 0 or msa.shape[2] == 0:
        return f"msa must be a non-empty [B, S, N, d_msa], got {tuple(msa.shape)}"
    return None


def _ext():
    """`csrc/sm100/opm_sm100.cu` (built by `opm_train._ext(arch="sm100")`, the prebuild's entry)."""
    from miniworld_engine.kernels.integrations import opm_train

    return opm_train._ext(arch="sm100")


def _ext_wide():
    """`csrc/sm100/opm_wide_sm100.cu`: the other widths (d_msa 128 / 256, c_z 256 / 384 / 512), the projection biases, the CTA-pair
    dgrad of every c_z and the Chai-1 passes (`outer_product_mean.chai_sm100`)."""
    if "wide" not in _EXT:
        from miniworld_engine.kernels._nvcc import load_extension as load

        csrc = Path(__file__).with_name("csrc") / "sm100"
        _EXT["wide"] = load(
            name="miniworld_opm_wide_sm100",
            sources=[str(csrc / "opm_wide_sm100.cu")],
            extra_include_paths=[str(csrc)],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
            extra_cflags=["-O3"],
        )
    return _EXT["wide"]


# The training backward: dWo (an HBM-bound kernel) runs on a side stream on a few SMs while cuBLAS computes dA and dB on the others
# (an SM-count target for those two GEMMs, which are bound by the power cap rather than by their SM count). {c_z: CTA budget}
# where it measured faster on the B200 branch (docs/gpus/b200/opm/opm.md, L128 S1024 MiniWorld width 0.167 -> 0.154 ms).
_DWO_SIDE_CTAS: dict[int, int] = {128: 16, 256: 16}
_DWO_SIDE_MIN_N = 128
_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}
warnings.filterwarnings("ignore", message="Setting the SM carveout for matmuls is a temporary experimental mitigation")


def _dwo_side_ctas(cz: int, n: int) -> int:
    if n < _DWO_SIDE_MIN_N or not hasattr(torch._C, "_set_sm_carveout_experimental"):
        return 0
    return _DWO_SIDE_CTAS.get(cz, 0)


def _side_stream(device) -> torch.cuda.Stream:
    idx = torch.device(device).index
    idx = torch.cuda.current_device() if idx is None else idx
    if idx not in _SIDE_STREAMS:
        _SIDE_STREAMS[idx] = torch.cuda.Stream(device=idx)
    return _SIDE_STREAMS[idx]


class _sm_carveout:
    """cuBLAS leaves `n` SMs free for the GEMMs issued inside (CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, recorded into a CUDA graph)."""

    def __init__(self, n: int):
        self.n = n

    def __enter__(self):
        self.prev = torch._C._get_sm_carveout_experimental()  # pyright: ignore[reportAttributeAccessIssue]
        torch._C._set_sm_carveout_experimental(self.n)  # pyright: ignore[reportAttributeAccessIssue]

    def __exit__(self, *exc):
        torch._C._set_sm_carveout_experimental(self.prev)  # pyright: ignore[reportAttributeAccessIssue]
        return False


def _bf16_params(ps: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """bf16 forms of the parameters (Wa, Wb, Wo, the bias[, the projection biases]): fp32 master parameters in ONE launch into one flat
    buffer (`opm_cast_bf16`, segments at 64-element boundaries); returns the views and the buffer (None: every parameter bf16 already)."""
    if all(p.dtype == torch.bfloat16 for p in ps):
        return [p.detach().contiguous() for p in ps], None
    flat = _ext().opm_cast_bf16([p.detach().contiguous() for p in ps], -1)
    return _bf16_views(flat, ps), flat


#: Inference reads the bf16 parameter forms cast once per parameter version (`opm_cast_bf16`, kept on the module; no cast launch per call or
#: per graph replay), as PWA (`pwa_sm100._infer_weights`) and the H100 path (`opm_train` param_forms) do. A / B switch.
_EAGER_W16 = [True]


def _infer_flat(module, ps: list[torch.Tensor], mode: int) -> torch.Tensor | None:
    """The flat bf16 buffer `opm_cast_bf16(ps, mode)` for an inference call, cached on the module and keyed by (storage, version, dtype)
    of the parameters; None when every parameter is bf16, the switch is off, or a miss happens during a CUDA-graph capture (that call
    casts inside the graph as before: a buffer from the graph pool must not outlive it)."""
    if not _EAGER_W16[0] or all(p.dtype == torch.bfloat16 for p in ps):
        return None
    key = (mode, tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in ps))
    hit = module.__dict__.get("_b200_opm_w16")
    if hit is not None and hit[0] == key:
        return hit[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    with torch.no_grad():
        flat = _ext().opm_cast_bf16([p.detach().contiguous() for p in ps], mode)
    module.__dict__["_b200_opm_w16"] = (key, flat)
    return flat


def _bf16_views(flat: torch.Tensor, ps: list[torch.Tensor]) -> list[torch.Tensor]:
    views, off = [], 0
    for p in ps:
        views.append(flat[off : off + p.numel()].view(p.shape))
        off += (p.numel() + 63) // 64 * 64
    return views


def _seg_offsets(ps: list[torch.Tensor]) -> list[int]:
    """The start of the projection-bias segments (the parameters after Wa, Wb, Wo, the bias) in `opm_cast_bf16`'s flat buffer."""
    offs, off = [], 0
    for k, p in enumerate(ps):
        if k >= 4:
            offs.append(off)
        off += (p.numel() + 63) // 64 * 64
    return offs


def _flat_numel(ps: list[torch.Tensor]) -> int:
    if all(p.dtype == torch.bfloat16 for p in ps):
        return 0
    return sum((p.numel() + 63) // 64 * 64 for p in ps)


def _counts(bits: torch.Tensor, dmin: float, doff: float) -> torch.Tensor:
    """The pair divisor table [N, N] fp32 max(n, dmin) + doff off the prologue's bit rows (`opm_norm`: the exact popcount, then one fp32
    add of the offset): (1, 0) the clamp at 1, (0, ``count_offset``) AlphaFold 3's n + offset. (Until 2026-11-17 the count_offset table
    came from a cuBLAS GEMM of the mask + clamp / add: three launches more, the same fp32 values.)"""
    return _ext().opm_norm(bits, float(dmin), float(doff))


#: The count table runs on a side stream beside the cuBLAS product (both only read the prologue's outputs): its few CTAs take the SMs
#: the product's last wave leaves idle instead of a launch of their own between the product and the epilogue. A / B switch.
_NORM_SIDE = [True]


def _product_and_counts(a2: torch.Tensor, bt: torch.Tensor, bits: torch.Tensor, dmin: float, doff: float):
    """O = A2 . BT^T (cuBLAS NT) and the count table; the table on a side stream when `_NORM_SIDE`."""
    if not _NORM_SIDE[0]:
        return torch.matmul(a2, bt.t()), _counts(bits, dmin, doff)
    main = torch.cuda.current_stream(a2.device)
    aux = _side_stream(a2.device)
    aux.wait_stream(main)
    with torch.cuda.stream(aux):
        norm = _counts(bits, dmin, doff)
    bits.record_stream(aux)
    o = torch.matmul(a2, bt.t())
    main.wait_stream(aux)
    norm.record_stream(main)
    return o, norm


def _pb_arg(pb16: list[torch.Tensor], flat: torch.Tensor | None, offs: list[int] | None) -> tuple[torch.Tensor, int]:
    """The projection biases for `opm_wprologue` (bf16): (tensor, offset of the right ones). One tensor: [left | right] (the training
    form, one parameter cat); two (inference: the module's two bias parameters, no cat): their segments of the cast buffer, the right one
    at the next 64-element boundary (no copy); two bf16 parameters without a cast buffer: one cat."""
    if len(pb16) == 1:
        return pb16[0], CH
    if flat is not None and offs is not None:
        return flat[offs[0] : offs[0] + 64 + CH], offs[1] - offs[0]
    return torch.cat(pb16), CH


def _params(args: list[torch.Tensor]) -> list[torch.Tensor]:
    """Wa, Wb, Wo, the bias[, the projection biases] of the op arguments."""
    return [args[4], args[5], args[6], args[7], *((args[9],) if args[9].numel() else ()), *(a for a in args[11:12] if a.numel())]


def _forward(args: list[torch.Tensor], eps: float, after: bool, dmin: float, doff: float, save: bool):
    """args = [msa [1, S, N, d_msa], mask bool [1, S, N], lnw, lnb, wa, wb, wo, bo, residual | empty, pbias | empty, asym | empty]
    (S % 128, N % 64). Returns z [1, N, N, c_z] and what the backward reads."""
    msa, mask, lnw, lnb = args[:4]
    residual = args[8] if args[8].numel() else None
    has_pb = bool(args[9].numel())
    asym = args[10] if args[10].numel() else None
    m = msa[0].contiguous()
    mk = mask[0].contiguous()
    n, cm = m.shape[1], m.shape[2]
    cz = args[6].shape[0]
    ext = _ext()
    ps = _params(args)
    if len(args) > 12 and args[12].numel():  # inference: the cached bf16 forms (_infer_flat)
        w16 = args[12]
        wa16, wb16, wo16, bo16, *pb16 = _bf16_views(w16, ps)
    else:
        (wa16, wb16, wo16, bo16, *pb16), w16 = _bf16_params(ps)
    lnw_, lnb_ = lnw.detach().contiguous(), lnb.detach().contiguous()
    wext = None
    if cm != CM or cz != CZ or has_pb:
        wext = _ext_wide()
        # the biases in bf16, as the module's bf16 Linear adds them (its fp32 accumulation, one rounding of the sum)
        pb, roff = _pb_arg(pb16, w16, _seg_offsets(ps)) if has_pb else (m.new_empty((0,), dtype=torch.float32), CH)
        a2, bt, stats, bits = wext.opm_wprologue(m, mk, lnw_, lnb_, float(eps), wa16, wb16, pb, bool(save), roff)
    else:
        a2, bt, _, stats, bits = ext.opm_prologue(m, mk, lnw_, lnb_, float(eps), wa16, wb16, bool(save), False)
    o, norm = _product_and_counts(a2, bt, bits, dmin, doff)  # O [(i, c), (j, e)] bf16 (cuBLAS NT), the count table
    if cz == CZ:
        z = ext.opm_epilogue(o, norm, wo16, bo16, n, n, residual, int(after), asym)
    else:
        z = (wext or _ext_wide()).opm_wepilogue(o, norm, wo16, bo16, n, n, residual, int(after), asym)
    return z.view(1, n, n, cz), a2, bt, norm, stats, o, w16


#: Inference at c_z 128 (d_hidden 32): the outer product and the projection in ONE kernel (`opm_fused_infer`, `opm_fused_sm100<128>`:
#: the grouped product [(i, c), (j, e)] never reaches HBM, the counts are popcounts of the prologue's bit rows) where it measured faster
#: than cuBLAS + the count table + the epilogue (B200, GPU 1, CUDA graph, L2 flushed, 2026-11-17; module, fused / unfused, min of 3):
#: S384 0.77-0.89 at every N (the fused kernel pads the depth to 128, the unfused path to 256: a third of the product is padding), S512
#: 0.93-0.98 from N 512 (1.01 at N 256); S1024 / S2048 0.97-1.17, run to run unstable (the fused mainloop's long-K gap): not served.
#: Only the clamp-1 divisor (no ``count_offset``), no interchain mask.
_FUSED_OFF = [False]  # A / B switch for measurements
#: The depth every path pads to: 128 (every sm_100a kernel takes S % 128; 256 was the H100 integration's multiple, a third of the work
#: padding at depth 384). Measured against 256 at S 384 (2026-11-17): inference 0.88-0.91, training 0.84-0.98 (L 128-768), the same
#: outputs (bitwise in inference; gradients within 7e-7 relative: the fp32 sums over fewer padded rows). A / B switch.
_S_ALIGN = [128]


def _fused_serves(s: int, n: int, cz: int, dmin: float, doff: float, has_asym: bool) -> bool:
    """``s`` / ``n``: the depth padded to 128 and the length padded to 64."""
    if _FUSED_OFF[0] or cz != CZ or (dmin, doff) != (1.0, 0.0) or has_asym:
        return False
    return s <= 512 and (s % 256 != 0 or n >= 512)


def _fused_forward(args: list[torch.Tensor], eps: float, after: bool) -> torch.Tensor:
    """The inference forward on `opm_fused_infer` (S % 128, N % 64; c_z 128): prologue (+ projection biases: the wide prologue), the fused
    product + projection (+ bias, / count, + residual)."""
    msa, mask, lnw, lnb = args[:4]
    residual = args[8] if args[8].numel() else None
    has_pb = bool(args[9].numel())
    m = msa[0].contiguous()
    mk = mask[0].contiguous()
    n = m.shape[1]
    ext = _ext()
    ps = _params(args)
    if all(p.dtype == torch.bfloat16 for p in ps):
        cz = args[6].shape[0]
        wa16, wb16, bo16, *pb16 = (p.detach().contiguous() for k, p in enumerate(ps) if k != 2)
        wo16 = args[6].detach().reshape(cz, 4, 8, 4, 8).permute(0, 1, 3, 2, 4).reshape(cz, CH * CH).contiguous()
        flat = None
    else:  # one cast launch; Wo comes back in the fused kernel's k order (c / 8, e / 8, c % 8, e % 8)
        flat = args[12] if len(args) > 12 and args[12].numel() else ext.opm_cast_bf16([p.detach().contiguous() for p in ps], 2)
        wa16, wb16, wo16, bo16, *pb16 = _bf16_views(flat, ps)
    lnw_, lnb_ = lnw.detach().contiguous(), lnb.detach().contiguous()
    if has_pb:
        pb, roff = _pb_arg(pb16, flat, _seg_offsets(ps))
        a2, bt, _, bits = _ext_wide().opm_wprologue(m, mk, lnw_, lnb_, float(eps), wa16, wb16, pb, False, roff)
    else:
        a2, bt, _, _, bits = ext.opm_prologue(m, mk, lnw_, lnb_, float(eps), wa16, wb16, False, False)
    return ext.opm_fused_infer(a2, bt, bits, wo16, bo16, residual, int(after)).view(1, n, n, -1)


@opaque(fake=lambda args, eps, after: _infer_fake(args, eps, after, 1.0, 0.0), name="opm_sm100_fused_infer")
def _fused_op(args: list[torch.Tensor], eps: float, after: bool) -> torch.Tensor:
    return _fused_forward(args, eps, after)


def _fwd_fake(args, eps, after, dmin, doff):
    m = args[0]
    _, s, n, _ = m.shape
    dev, bf, f32 = m.device, torch.bfloat16, torch.float32
    return [
        torch.empty((1, n, n, args[6].shape[0]), device=dev, dtype=bf),
        torch.empty((n * CH, s), device=dev, dtype=bf),
        torch.empty((n * CH, s), device=dev, dtype=bf),
        torch.empty((n, n), device=dev, dtype=f32),
        torch.empty((n, s, 2), device=dev, dtype=f32),
        torch.empty((n * CH, n * CH), device=dev, dtype=bf),
        torch.empty((_flat_numel(_params(args)),), device=dev, dtype=bf),
    ]


@opaque(fake=_fwd_fake, name="opm_sm100_fwd")
def _fwd_op(args: list[torch.Tensor], eps: float, after: bool, dmin: float, doff: float) -> list[torch.Tensor]:
    """The training forward: [z, A2, BT, counts, stats, O, the parameters' flat bf16 forms | empty]."""
    z, a2, bt, norm, stats, o, w16 = _forward(args, eps, after, dmin, doff, True)
    return [z, a2, bt, norm, stats, o, w16 if w16 is not None else args[0].new_empty((0,), dtype=torch.bfloat16)]


def _infer_fake(args, eps, after, dmin, doff):
    m = args[0]
    return torch.empty((1, m.shape[2], m.shape[2], args[6].shape[0]), device=m.device, dtype=torch.bfloat16)


@opaque(fake=_infer_fake, name="opm_sm100_infer")
def _infer_op(args: list[torch.Tensor], eps: float, after: bool, dmin: float, doff: float) -> torch.Tensor:
    return _forward(args, eps, after, dmin, doff, False)[0]


def _bwd_fake(args, kept, dz, eps, after):
    return [torch.empty_like(args[i]) for i in (0, 2, 3, 4, 5, 6, 7, 9)]


@opaque(fake=_bwd_fake, name="opm_sm100_bwd")
def _bwd_op(args: list[torch.Tensor], kept: list[torch.Tensor], dz: torch.Tensor, eps: float, after: bool) -> list[torch.Tensor]:
    """[dmsa, dlnw, dlnb, dWa, dWb, dWo, dbias, dpbias | empty] from dz [1, N, N, c_z] and the forward's kept tensors."""
    bf = torch.bfloat16
    msa, mask, lnw, lnb, wa, wb, wo, bo = args[:8]
    has_pb = bool(args[9].numel())
    asym = args[10] if args[10].numel() else None
    a2, bt, norm, stats, o, w16 = kept
    m = msa[0]
    mk = mask[0].contiguous()
    s, n, cm = m.shape
    cz = wo.shape[0]
    ext, wext = _ext(), _ext_wide()
    if w16.numel():  # the forward's bf16 forms of Wa, Wb, Wo (no cast kernel here)
        wa16, wb16, wo16 = _bf16_views(w16, [wa, wb, wo])
    else:
        wa16, wb16, wo16 = (w.detach().to(bf).contiguous() for w in (wa, wb, wo))
    # the CTA-pair dgrad at every c_z: dO in the grouped layout, dz / n, dbias (sum dz / n with the division after the projection);
    # the interchain mask: a pair of two chains passes no gradient
    d_o, dzp, dbo = wext.opm_wdgrad(dz[0].contiguous(), norm, wo16, n, n, int(bo.dtype == bf), asym, int(after))
    dwo_fn = ext.opm_dwo if cz == CZ else wext.opm_wdwo
    side = _dwo_side_ctas(cz, n)
    if side:
        main = torch.cuda.current_stream(dz.device)
        aux = _side_stream(dz.device)
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            dwo = dwo_fn(dzp, o, n, n, int(wo.dtype == bf), side)
        dzp.record_stream(aux)
        o.record_stream(aux)
        with _sm_carveout(side):
            da = torch.mm(bt.t(), d_o.t())  # [s, (i, c)]
            db = torch.mm(a2.t(), d_o)  # [s, (j, e)]
        main.wait_stream(aux)
        dwo.record_stream(main)
    else:
        da = torch.mm(bt.t(), d_o.t())
        db = torch.mm(a2.t(), d_o)
        dwo = dwo_fn(dzp, o, n, n, int(wo.dtype == bf))
    del d_o
    lnw_, lnb_ = lnw.detach().contiguous(), lnb.detach().contiguous()
    if cm == CM:
        # ssa: the masked column sums of [dA dB] (the kernel's row 64 of R) = the projection-bias gradients
        dm, dwa, dwb, dgam, dbet, red = ext.opm_prologue_bwd(da, db, m, stats, mk, lnw_, lnb_, wa16, wb16)
        ssa = red.reshape(-1)[CM * CM : CM * CM + 2 * CH]
    else:
        mkf = mk.reshape(1, -1).to(bf)
        ssa = torch.cat(
            [
                torch.mm(mkf, da.view(-1, CH), out_dtype=torch.float32).view(-1),
                torch.mm(mkf, db.view(-1, CH), out_dtype=torch.float32).view(-1),
            ]
        )
        dm, dwa, dwb, dgam, dbet = wext.opm_wprologue_bwd(da, db, m, stats, mk, lnw_, lnb_, wa16, wb16, ssa)
    # the parameter gradients in their parameters' dtypes, as new tensors (a custom op's returns may not alias each other), ONE launch
    # (as the H100 path: _cast_cuda.cast(..., force=True))
    gs = [(dgam, lnw), (dbet, lnb), (dwa, wa), (dwb, wb), (dwo, wo), (dbo, bo), *(((ssa, args[9]),) if has_pb else ())]
    out = _cast_cuda.cast([g for g, _ in gs], [p.dtype for _, p in gs], force=True)
    return [dm.view(1, s, n, cm), *out[:6], out[6] if has_pb else args[9].new_empty((0,))]


class _OpmSm100Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = list(args[:11])
        ctx.eps, ctx.after, dmin, doff = args[11:]
        ctx.has_residual = bool(tensors[8].numel())
        ctx.has_pb = bool(tensors[9].numel())
        z, *kept = _fwd_op(tensors, ctx.eps, ctx.after, dmin, doff)
        ctx.save_for_backward(*tensors, *kept)
        return z

    @staticmethod
    def backward(ctx, dz):
        vals = ctx.saved_tensors
        g = _bwd_op(list(vals[:11]), list(vals[11:]), dz.to(torch.bfloat16).contiguous(), ctx.eps, ctx.after)
        return (
            g[0],
            None,
            *g[1:7],
            dz if ctx.has_residual else None,
            g[7] if ctx.has_pb else None,
            None,
            None,
            None,
            None,
            None,
        )


def outer_product_mean(
    module, msa: torch.Tensor, mask: torch.Tensor | None, residual: torch.Tensor | None = None, token_asym_id: torch.Tensor | None = None
) -> torch.Tensor:
    """The module's update (+ ``residual``) on the B200 kernels; ``token_asym_id`` only with the module's interchain mask."""
    if mask is None:
        mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
    if msa.shape[0] > 1:  # one MSA stack at a time (each its own autograd node)
        return torch.cat(
            [
                outer_product_mean(
                    module,
                    msa[b : b + 1],
                    mask[b : b + 1],
                    None if residual is None else residual[b : b + 1],
                    None if token_asym_id is None else token_asym_id[b : b + 1],
                )
                for b in range(msa.shape[0])
            ]
        )
    from miniworld_engine.kernels.integrations.opm_train import divisor_params

    eps = float(module.ln_msa.eps)
    after = not module.normalize_before_proj
    cz = module.to_out.weight.shape[0]
    dmin, doff = divisor_params(module, msa.shape[1])
    n0 = msa.shape[2]
    mask = mask if mask.dtype == torch.bool else mask != 0
    pn = -n0 % 64
    fused = not torch.is_grad_enabled() and _fused_serves(
        msa.shape[1] + -msa.shape[1] % 128, n0 + pn, cz, dmin, doff, token_asym_id is not None
    )
    ps = -msa.shape[1] % (128 if fused else _S_ALIGN[0])
    if pn or ps:  # masked rows: no count, no product, zero projections (biases included); the pad's backward drops their gradient
        msa = torch.nn.functional.pad(msa, (0, 0, 0, pn, 0, ps))
        mask = torch.nn.functional.pad(mask, (0, pn, 0, ps))
    fuse = residual is not None and not pn and residual.dtype == msa.dtype and tuple(residual.shape) == (1, n0, n0, cz)
    r = residual.contiguous() if fuse and residual is not None else msa.new_empty((0,))
    grad = torch.is_grad_enabled()
    lbias, rbias = module.to_left.bias, module.to_right.bias
    # the projection biases: training one parameter cat [left | right] (its gradient is one tensor); inference the two parameters as they
    # are (args[9], args[11]: the cast kernel lays them out, no cat launch)
    pbias = msa.new_empty((0,), dtype=torch.float32) if lbias is None else torch.cat([lbias, rbias]) if grad else lbias
    asym = msa.new_empty((0,), dtype=torch.int32)
    if token_asym_id is not None:  # chain ids int32 [N]; padded tokens -1 (their pairs are cut away)
        asym = torch.nn.functional.pad(token_asym_id[0].to(torch.int32), (0, pn), value=-1).contiguous()
    tensors = [
        msa.contiguous(),
        mask.contiguous(),
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_left.weight,
        module.to_right.weight,
        module.to_out.weight,
        module.to_out.bias,
        r,
        pbias,
        asym,
    ]
    if not grad:  # inference: [11] the right projection bias (or empty), [12] the cached bf16 parameter forms (or empty)
        tensors.append(rbias if lbias is not None else msa.new_empty((0,), dtype=torch.float32))
        w16 = _infer_flat(module, _params(tensors), 2 if fused else -1)
        tensors.append(w16 if w16 is not None else msa.new_empty((0,), dtype=torch.bfloat16))
    if grad:
        out = _OpmSm100Fn.apply(*tensors, eps, after, dmin, doff)
    elif fused:
        out = _fused_op(tensors, eps, after)
    else:
        out = _infer_op(tensors, eps, after, dmin, doff)
    if pn:
        out = out[:, :n0, :n0]
    return out + residual if residual is not None and not fuse else out
