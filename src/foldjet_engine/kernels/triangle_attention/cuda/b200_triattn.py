"""B200 (sm_100a) TriangleAttention, starting node, d_pair = 128, 4 heads x 32, bf16, optional broadcast dropout.

Forward:  tri_front (LN + q|k|v|g projections + head-major masked bias) -> triattn_fwd (attention + base-2 LSE)
          -> tri_tail (sigmoid gate + to_out + dropout scale + residual).
Backward: tri_gate_bwd (do, dg, delta, dWo) -> triattn_bwd (KV-owned dK/dV + query-owned dQ/dbias)
          -> tri_head_bwd (projection dgrad + LN backward + residual) -> tri_wgrad (parameter gradients on tcgen05).
Kernel bodies are tcgen05 / TMEM / TMA hand-CUDA in ``b200_sources/``, from the B200 research branch ``b200/msa``
(2026-09-27). Every call owns its buffers; nothing is cached across calls except the compiled extensions.
"""

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)
import functools
import os
from pathlib import Path

import torch

from foldjet_engine.kernels import _capture
from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).with_name("b200_sources")

#: Deterministic parameter gradients on the fused d_pair 128 path (per-CTA fp32 partials summed in CTA order instead of fp32 atomics;
#: the wide path is always deterministic). On by default since B200 2026-11-17 (training must give the same bits every run).
DETERMINISTIC_WGRAD = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_DETERMINISTIC_WGRAD", "1") not in ("", "0")

#: Small independent launches of the fused d_pair 128 training step as parallel branches of a captured CUDA graph (B200 2026-11-17,
#: ``bias_only_dit.cuda.side``: forks only while a graph is being captured): the dropout draw beside the parameter pack / front / core,
#: the backward's set-up beside the gate backward, head_bwd's constant upload and the dWo slot sum beside the attention backward.
SIDE_STREAMS = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_SIDE", "0") not in ("", "0")
#: (S90 training, cuda:b200_sm100 67) the core backward's key-side and query-side kernels are independent: while a CUDA graph is captured they are two branches (the query side with its slab sum on a side stream), so the SMs the key side leaves idle in its
#: second task round run query-side CTAs.  Step graph, off -> on (us): d_pair 64 x 2 L128 77.5 -> 73.5, L256 189.7 -> 184.8, L512 739.4 -> 722.4; 64 x 4 L128 90.4 -> 89.5, L512 1242 -> 1210; 256 x 8 L128 191.7 -> 186.9, L512 2588 -> 2534; 384 x 12 L512 4082 -> 4019.
#: FOLDJET_ENGINE_TRIATTN_B200_KVQ_BRANCH=0 turns it off (eager calls never branch).
#: (S90 training, cuda:b200_sm100 69) the wide backward's slab sum (`triattn_db_slabs_rows`) writes the head backward's dbias rows [R][64] (bf16, the H heads then zeros; extra blocks write the zero columns) directly: no `tri_db_rows` launch.
#: Starting-node bias orientation only (bt: the old kernel).  Step graph, off -> on (us): d_pair 64 x 2 L128 73.6 -> 71.2, L256 184.7 -> 178.0, L512 722.4 -> 689.5, L768 1879 -> 1801; 64 x 4 89.6 -> 88.4 (L128), 268 -> 261 (L256), 1209 -> 1180 (L512), 3331 -> 3257 (L768, ending); 256 x 8 187.7 -> 186.0 / 576 -> 569 / 2537 -> 2517 / 6870 -> 6809; 384 x 12 L512 4021 -> 4008.  Bitwise equal output, poison-fill tests equal.  FOLDJET_ENGINE_TRIATTN_B200_DBROWS=0 turns it off.
DBROWS_FUSED = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_DBROWS", "1") not in ("", "0")
KVQ_BRANCH = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_KVQ_BRANCH", "1") not in ("", "0")


#: drop tensor data_ptr -> the side stream its draw runs on (set by integrations.triattn_b200.forward, joined by fused_forward before the
#: tail reads it)
PENDING_DROP: dict = {}


def _side_mod():
    from foldjet_engine.kernels.bias_only_dit.cuda import side

    return side


def _on(side, stream):
    """``side.on(stream)``, or no context without the side module."""
    import contextlib

    return side.on(stream) if side is not None else contextlib.nullcontext()


C = 128  # d_pair
H = 4  # heads
D = C // H  # head dim (the kernels take 32 only)
TOK = 128  # query / key / token tile of every kernel; the kernels' length is a multiple of it


def padded_length(length: int) -> int:
    """The kernels' length for a caller's length: whole 128-token tiles."""
    return -(-length // TOK) * TOK


def ragged(length: int) -> int:
    """The ``lr`` argument of the kernels that touch the caller's pair-shaped tensors: the caller's length when it is not a whole number
    of tiles, else 0.  Those kernels (front, tail, gate / head backward, weight gradients) read and write the caller's unpadded
    [B, L, L, C] tensors through 4-D tensor maps whose bounds are L: rows and columns beyond L load as zeros (the zero padding) and
    their stores are dropped (the cut); keys beyond L are masked in the front's bias drain.  Every intermediate (q, k, v, g, bias, o,
    lse, the gradient rows) keeps the padded length, so the attention core is unchanged and no padded copy of the pair, the output,
    dy or dx is made."""
    return length if length % TOK else 0


def _flags():
    return [*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--use_fast_math", "-DNDEBUG", f"-I{_SRC}"]


# Per-length schedule of the attention core, from a sweep of every registered L (2026-09-29, triattn.md "Schedule per
# length"): two pair rows per forward task up to L = 384 (more tasks for the 148 SMs; -15 % at L256), four above.
# B200 2026-11-17: three rows (one a softmax group: `G3` in triattn_sm100.cu) from L = 384 on: L384 94.4 -> 90.3 us, L512 201.0 -> 193.0,
# L768 625 -> 583, L2048 10.90 -> 10.66 ms (4 heads, the same bits as the 2 / 4-row builds); L128 / 256 keep two (10.3 / 32.9 vs 12.4 / 34.9).
# The backward kernels are the same in both builds (query-side QR = 4 was best at every L).
# B200 2026-11-17 (round 3 part 3): at L256 the three-row build wins once there are enough tasks -- 12 heads 96.0 -> 91.4 us, 8 heads
# 65.0 / 65.8 (even), 2 / 4 heads 20.6 / 32.9 vs 22.7 / 34.9 (core, graph): `bh` = samples x heads of the launch.
def _rows_per_task(length: int, bh: int = 1) -> int:
    return 2 if length <= 128 or (length <= 256 and bh < 12) else 3


#: B200 2026-11-18 (round 3 part 9): inference takes the core build whose bias tile is bf16(bias * log2 e) -- the fronts write it so
#: (their bias rows pre-multiplied by log2 e in fp32: one rounding, as before) -- and adds it to the scaled score by one mixed fp32 + bf16
#: add an element (FHADD, no bf16 unpack: three instructions a key pair instead of four), in 16-key chunks (TA_CK=16: the 32-key chunks
#: spilled with this form at the 128-register cap). Core, 4 heads: L1024 1436 -> 1386 us, L2048 10.79 -> 10.45 ms, L512 -1 %.
#: Training keeps the unscaled build (its backward reads the bias).  FOLDJET_ENGINE_TRIATTN_B200_HB=0 turns it off.
HB_INFER = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HB", "1") != "0"
#: B200 2026-11-18 (round 3 part 10): inference with 16-channel heads (AF3 d_pair 64 x 4 x 16) on the d_pair 64 front: the front writes 1.0 into
#: v's padded channel 16 and the core (TA_D16) takes the row sum from the PV GEMM's channel 16 (no FADD2 a key pair, no per-tile overflow
#: guard), one QK^T MMA a sub-step.  FOLDJET_ENGINE_TRIATTN_B200_D16=0 turns it off.
D16_INFER = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_D16", "1") != "0"
#: B200 2026-11-18 (round 3 part 11): those heads stay compact (16 channels a head, the H100 fp16 core's layout) through the front's
#: q / k / v stores, the core's loads (3-D maps whose 32-channel box zero-fills channels 16 .. 31; the core writes v's 1.0 into the
#: landed V tile, TA_D16C) and the tail's o loads / out-projection (K = 64 instead of 128).  FOLDJET_ENGINE_TRIATTN_B200_D16C=0 turns it off.
D16C_INFER = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_D16C", "1") != "0"


D16QS_POLY = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_D16QS_POLY", "4"))   # exponential pairs of 16 on the FMA pipe, the QS build of 16-channel heads (three rows a task)
D16QS_POLY2 = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_D16QS_POLY2", "4"))   # (two rows)


def POLY_INFER(rows: int, qs: bool = False) -> int:
    """B200 2026-11-18 (round 3 part 10): the inference (HB) core computes this many of every 16 exponential pairs on the FMA pipe
    (Cody-Waite + degree-3 polynomial, TA_POLY; relative error 7.5e-5, under P's bf16 rounding): since the bias add lost its unpack
    (part 9) the MUFU ex2 bounds the softmax.  Core, 4 heads, graph, us (off -> on): head dim 32 L256 31.9 -> 30.9 (4 of 16), L768
    582 -> 561 (2), L2048 level (2); the 16-channel core (TA_D16) L256 30.8 -> 28.8 (4), L768 555 -> 529-535 (2), L2048 9502 ->
    9162-9202 (2); 6 of 16 is slower everywhere.  Two rows a task (L <= 256): 4, else 2.  FOLDJET_ENGINE_TRIATTN_B200_POLY=0 turns it off."""
    if os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_POLY", "1") == "0":
        return 0
    if qs:  # (the QS core has one FFMA2 a key pair less: 4 .. 7 of 16 are level at L1024 (1497-1503 us a module), 2 is 1560, 8 1558; the poly pair's -rr negations are gone too: TA_PSUB)
        return 4 if rows == 2 else 5
    return 4 if rows == 2 else 2
L2E = 1.4426950408889634
#: B200 2026-11-19: the d_pair 128 inference core takes q pre-scaled by scale * log2 e (the front's q weight rows scaled once, cached) and starts every
#: row at offset 0 (provisional; the overflow guard raises it): its logit is S + bias (two FHADD a key pair, no FFMA2).  FOLDJET_ENGINE_TRIATTN_B200_QS=0 turns it off.
QS_INFER = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_QS", "1") != "0"
#: ... above this padded length only: the first key tile of a row is the old loop (it fixes the offset), so with 3-4 tiles (L <= 512) the gain is gone
#: (L384 starting + ending 155.4 / 159.5 -> 157.7 / 161.7 us, L512 level) and it pays from 5 tiles on (L640 .. L2048 -3 .. -6 %).
QS_MIN_L = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_QS_MIN_L", "512"))
#: the 16-channel-head core (TA_D16: the row sum comes from the PV GEMM) takes it above this length (its loop has no FADD2, so the logit FFMA2 is a larger share of it)
QS_MIN_L16 = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_QS_MIN_L16", "256"))
QS_MIN_L64 = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_QS_MIN_L64", "512"))   # the head-dim-64 core (HB64_INFER)


def qs_for(hb: bool, head_dim: int, padded_len: int) -> bool:
    """Whether a call takes the QS core: the HB inference build of 32-channel heads above QS_MIN_L (the weight pack and the core decide alike)."""
    return bool(hb and QS_INFER and head_dim in (D, 16, 64) and padded_len > (QS_MIN_L16 if head_dim == 16 else QS_MIN_L64 if head_dim == 64 else QS_MIN_L))


@functools.lru_cache(maxsize=None)
def _core(rows: int = 4, hb: bool = False, d16: bool = False, poly: int | None = None, d16c: bool = False, qs: bool = False):
    """The head-dim-32 core with `rows` pair rows a task; hb: the inference build (bias pre-scaled by log2 e, TA_HB), d16: its
    16-channel-head form (TA_D16), poly: exponential pairs of 16 on the FMA pipe (TA_POLY; None: POLY_INFER(rows)), d16c: (with d16) compact 16-channel heads in global memory (TA_D16C), qs: (hb, not d16) q pre-scaled by scale * log2 e (TA_QS)."""
    ensure_cuda_home()
    p = ((D16QS_POLY if rows >= 3 else D16QS_POLY2) if qs and d16 else POLY_INFER(rows, bool(qs)) if poly is None else poly) if hb else 0
    return load_extension(
        name=(f"triattn_b200_core_r{rows}_hb_d16{'c' if d16c else ''}_p{p}{'_qs' if qs else ''}_v3_tt8" if d16 else f"triattn_b200_core_r{rows}_hb_p{p}{'_qs' if qs else ''}_v2_tt8") if hb else f"triattn_b200_core_r{rows}_v20_tt8",
        sources=[str(_SRC / "triattn_sm100.cu")],
        extra_cuda_cflags=[*_flags(), f"-DTA_R={rows}", *(("-DTA_HB=1", "-DTA_CK=16", f"-DTA_POLY={p}") if hb else ()), *(("-DTA_D16=1",) if hb and d16 else ()), *(("-DTA_D16C=1",) if hb and d16 and d16c else ()), *(("-DTA_QS=1",) if hb and qs else ())],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _pack():
    """The H100 pack source (``h100_sources/pack.cu``: plain CUDA, no sm_90 instructions) built for sm_100a, for its ``drop_scale``: the
    training dropout scale in one Philox launch, bitwise the torch draw (B200 2026-11-17: the draw was five torch launches a step). A draw
    captured in a graph registers its scalars with the ONE B200 graph RNG writer (``kernels/_graph_rng_sm100``, shared with TriMul):
    ``b200_sources/pack_sm100.cu`` includes pack.cu unchanged and routes its fill_ override and registration there."""
    from foldjet_engine.kernels import _graph_rng_sm100

    ensure_cuda_home()
    ext = load_extension(
        name="triattn_b200_pack_v2",
        sources=[str(_SRC / "pack_sm100.cu")],
        extra_cuda_cflags=[*_flags(), "-DMW_TRIATTN_KV_TILE=64", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
        verbose=False,
    )
    _graph_rng_sm100.connect(torch.ops.mw_pack_sm100.set_graph_rng_api)
    return ext


def drop_scale(pair: torch.Tensor, p: float) -> torch.Tensor | None:
    """``modules.functional.dropout_scale(pair, p, 1 or 2)`` as [B, L, C] (either broadcast axis of a square pair: the same tensor) in one
    launch, for a BF16 square pair and 0 < p < 1 (the H100 kernel's contract); None otherwise (the caller draws with torch)."""
    if not (pair.dtype == torch.bfloat16 and pair.ndim == 4 and pair.shape[1] == pair.shape[2] and pair.numel() > 0 and 0.0 < p < 1.0):
        return None
    from .h100_triattn import _drop_operands

    threshold, inverse = _drop_operands(float(p))
    return _pack().drop_scale(pair.detach(), threshold, inverse)


@functools.lru_cache(maxsize=1)
def _l128():
    """The L = 128 inference kernel (tri_l128_sm100.cu): front + core + tail of a pair row in one CTA, after a light bias launch."""
    ensure_cuda_home()
    return load_extension(
        name="triattn_b200_l128_v3_tt8", sources=[str(_SRC / "tri_l128_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False
    )


#: B200 2026-11-18 (round 3 part 7): inference at L = 128 (d_pair 128 x 4 heads, no gate / out biases, no dropout) runs the fused
#: one-CTA-a-pair-row kernel (tri_l128) instead of tri_front -> triattn_fwd -> tri_tail.  FOLDJET_ENGINE_TRIATTN_B200_L128=0 turns it off (A/B).
L128_FUSED = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_L128", "1") != "0"

#: B200 2026-11-18 (training tails round 8): the training forward at L = 128 (d_pair 128, no gate / out biases, AF3-orientation bias off) runs
#: the same fused kernel in its training form (tri_l128_train: also stores q | k | v | g | o, the base-2 LSE and the bias for the backward,
#: the dropout scale in its epilogue) instead of tri_front -> triattn_fwd -> tri_tail.  FOLDJET_ENGINE_TRIATTN_B200_L128_TRAIN=0: the chain.
L128_TRAIN = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_L128_TRAIN", "1") != "0"
#: (round 8) d_pair 128 is opt-in: its module step measured 113.8 -> 117.8 us with the fused training forward (the chain front -> core -> tail
#: overlaps by PDL; the cooperative launch does not) -- the wide path's d_pair 64 rows take it (83.4 -> 77.5 us, 99.8 -> 92.4)
L128_TRAIN_D128 = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_L128_TRAIN_D128", "0") != "0"
#: (round 8) the wide path's head-dim-32 core backward at L = 128 as programmatic dependent launches (triattn_bwd pdl=True)
CORE_PDL = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_CORE_PDL", "1") != "0"


@functools.lru_cache(maxsize=1)
def _mod():
    ensure_cuda_home()
    return load_extension(
        name="triattn_b200_mod_v23_tt11", sources=[str(_SRC / "triattn_mod_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False
    )


def supports(length: int) -> bool:
    """Lengths the kernels take: whole 128-token tiles."""
    return length > 0 and length % TOK == 0


class TriAttnCore(torch.autograd.Function):
    """out = softmax(scale q.k + bias) v on [B, N, S, H, 32] tensors, bias [B, H, S, S]."""

    @staticmethod
    def forward(ctx, q, k, v, bias, scale):
        out, lse = _core_forward(q, k, v, bias, scale, any(ctx.needs_input_grad))
        ctx.save_for_backward(q, k, v, bias, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, bias, out, lse = ctx.saved_tensors
        e = _core(_rows_per_task(q.shape[2]))
        dout = dout.contiguous()
        delta = e.triattn_delta(dout, out)
        E = q.new_empty(0)
        dq, dk, dv, db = e.triattn_bwd(q, k, v, bias, dout, lse, delta, ctx.scale, q.new_empty(0, dtype=torch.float32), E, E, E)
        return dq, dk, dv, db.to(bias.dtype), None


#: The attention core's forward in launches of a few samples (B200, 2026-10-20).  Its persistent kernel walks its tasks in a static
#: round robin, and in a long launch the CTAs drift apart and stop sharing K / V rows in L2: with four pair rows a task (L > 384),
#: module inference of 48 samples at L768 took 43.9 ms in one launch and 41.6 ms in launches of ~36 864 tasks (8 samples; L640 B48
#: 26.7 -> 25.8 ms, d_pair 256 L768 B16 29.5 -> 29.0 ms; training +0.5-1.7 %).  The forward with two rows a task (L <= 384) does not
#: drift.  Measured and not kept: a dynamic task counter in the forward (each CTA claiming its next task from a global counter, the
#: H100 fix) was 4-5 % slower per sample at every launch size on B200, and the same grouping of the two backward kernels changed the
#: module's training step by -0.5 to +0.1 % (isolated, with idle gaps between launches, they had looked 6-10 % faster in groups).
#: Each launch is the same kernel on a contiguous range of samples (the same bits).
GROUP_TASKS = 36864


def fwd_group(batch: int, heads: int, length: int) -> int:
    """Samples per launch of the core's forward: ~GROUP_TASKS tasks ((sample, head, 128-query tile) x 4 pair rows) at L > 384, all
    samples at L <= 384 (two rows a task: no drift)."""
    if length <= 384:
        return batch
    per = heads * (length // TOK) * ((length + 3) // 4)  # (counted in 4-row tasks: the same samples a launch with three rows)
    return max(1, min(batch, GROUP_TASKS // per))


def _core_forward(q, k, v, bias, scale, need, hb=False, d16=False, d16c=False, qs=False, kind=None, site=None):
    """The core's forward over [B, N, S, H, 32] tensors in launches of `fwd_group` samples -> (out [B, N, S, H, 32], lse or empty).
    hb: the bias is bf16(bias * log2 e) (inference: see HB_INFER); d16 (with hb): 16-channel heads whose v carries 1.0 in channel 16 (D16_INFER)."""
    B, N, S, H, _ = q.shape
    if kind is not None and hb and not need and not d16:  # opt-in e4m3 P / V core (docs/b200/fp8.md "TriAttn"; off unless fp8.enable_triattn set the switches)
        from . import h100_sm90 as _hs

        if _hs.FP8_PV:
            from . import b200_fp8

            o8 = b200_fp8.route(kind, q, k, v, bias, scale, qs, site)
            if o8 is not None:
                return o8, q.new_empty(0, dtype=torch.float32)
    e = _core(_rows_per_task(S, B * H), hb, d16, d16c=d16c, qs=qs)
    g = fwd_group(B, H, S)
    if g >= B:
        E = q.new_empty(0)  # (pybind: the optional outputs are passed explicitly)
        o, lse, _ = e.triattn_fwd(q, k, v, bias, scale, need, E, E)
        return o, lse
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty((B, N, H, S) if need else (0,), device=q.device, dtype=torch.float32)
    for i in range(0, B, g):
        j = min(B, i + g)
        e.triattn_fwd(q[i:j], k[i:j], v[i:j], bias[i:j], scale, need, o[i:j], lse[i:j] if need else lse)
    return o, lse


_WS_DWO, _WS_MT, _WS_VEC, _WS_DB = 0, C * C, 5 * C * C, 5 * C * C + 1056


def _empty_mask(device):
    return torch.empty(0, device=device, dtype=torch.bool)


#: S90: the bias-only inference update at L % 128 == 0, B = 1 as three launches around a cuBLAS PV: ``tri_front(vpl)`` (LN + the v
#: projection + the masked head-major bias, v written as a channel-major plane in the frame layout), softmax of the bias, ``bmm``
#: (the planes' PV, one launch for either node: the ending node is the starting node of the transposed frame), ``tri_tailg_pl``
#: (gate from x + out-projection + residual, o read as a plane). The H100 chain's LN pass, y stream, planes GEMM and g plane are gone.
BO_PLANES = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_BO_PLANES", "1") != "0"


def bias_only_planes(pair, lnw, lnb, wv, wg, wb, wo, mask, eps, starting, zero):
    B, L = pair.shape[0], pair.shape[1]
    x = pair.contiguous().view(-1, C)
    m = _mod()

    def pack():
        return m.tri_pack_params(
            zero.contiguous(), zero.contiguous(), wv.contiguous(), wg.contiguous(), wo.contiguous(), lnw.contiguous(), lnb.contiguous(), wb.contiguous()
        )

    w4, wo16, prm = _capture.param_forms("triattn_b200_pack", (zero, zero, wv, wg, wo, lnw, lnb, wb), pack, capture_reads_eager=True)
    lw, lb, wbf = prm[0], prm[1], prm[2:]
    trans = not starting
    _, _, vpl, _, bias = m.tri_front(x, lw, lb, eps, w4, wbf, mask if mask is not None else _empty_mask(pair.device), B, L, trans, False, 0, False, True)
    with torch.autocast("cuda", enabled=False):
        p = torch.softmax(bias.view(H, L, L), -1)
    o = torch.empty(H * D, L * L, device=pair.device, dtype=pair.dtype)
    with torch.autocast("cuda", enabled=False):
        torch.bmm(vpl.view(H, D * L, L), p.transpose(1, 2), out=o.view(H, D * L, L))
    return m.tri_tailg_pl(o, x, wo16, w4[3 * C :], lw, lb, eps, L, x.new_empty(0), trans, 0).view(B, L, L, C)


#: Inference (no grad, no dropout) on the fused path: the tail projects the gate from x (``tri_tailg``: g = LN(x) . Wg^T in fp32 inside the
#: tail, never stored) and the front writes q / k / v only -- one [B L^2, 128] bf16 stream less written and read (the H100 inference
#: tail, ``h100_sm90.INFERENCE_TAIL_GATE``).  RoseTTAFold3's gate / out-projection biases ride on it (fp32, before each rounding).
INFERENCE_TAIL_GATE = True


def _qk8_front(kind, site, S, H, dev):
    """The e4m3 q / k front's scales (qa0, ka0, qvi, guard) of this call's calibrated site, or None (FP8 route off, S below the measured gate, uncalibrated, ...): docs/b200/fp8.md "TriAttn e4m3 q / k"."""
    from . import h100_sm90 as _hs

    if not _hs.FP8_PV:
        return None
    from . import b200_fp8

    return b200_fp8.qk_front(kind, site, S, H, dev)


def fused_forward(
    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None, need=False, out=None, trans=False, bt=False, gb=None, ob=None
):
    """The d_pair 128 module forward (starting node, residual included) on B samples: -> (y [B, L, L, 128], saved).
    saved = (prm, w4, wo16, q, k, v, g, bias, o, lse): what `fused_backward` reads (lse empty unless `need`); every one a fresh
    allocation (none aliases an argument: they can be outputs of a custom op).  `out`: a contiguous bf16 tensor of the pair's size
    to write y into (a chunk's rows of a batch's output), or None.  Kernels: tri_pack_params -> tri_front -> triattn_fwd -> tri_tail.
    trans: the ending node -- pair, y (and the backward's dy, dx) in the caller's [B, I, J, C] layout, read and written through 4-D
    tensor maps by the kernels that touch them (the saved intermediates stay in the starting-node layout: no transposed copies).
    bt: the ending node on AlphaFold 3's bias orientation (``transposed_bias=False``): the bias of the pair as it is (inference; the
    backward reads the bias gradient in the other orientation, so training takes the wide path)."""
    B, L, _, _ = pair.shape
    Lp, lr = padded_length(L), ragged(L)
    x = pair.contiguous().view(-1, C)
    m = _mod()

    def pack():
        return m.tri_pack_params(
            wq.contiguous(),
            wk.contiguous(),
            wv.contiguous(),
            wg.contiguous(),
            wo.contiguous(),
            lnw.contiguous(),
            lnb.contiguous(),
            wb.contiguous(),
        )

    # inference: the packed weights are cached while the parameters are unchanged (kernels._capture.param_forms, as H100 caches its
    # pack on the module) -- one launch fewer per call; training keeps a fresh pack per call (it is saved for the backward, and a
    # custom op's outputs may not share storage with a cache)
    w4, wo16, prm = (
        pack() if need else _capture.param_forms("triattn_b200_pack", (wq, wk, wv, wg, wo, lnw, lnb, wb), pack, capture_reads_eager=True)
    )
    lw, lb, wbf = prm[0], prm[1], prm[2:]
    if L128_FUSED and Lp == TOK and not need and drop is None:
        E, E32 = x.new_empty(0), x.new_empty(0, dtype=torch.float32)
        y = _l128().tri_l128(
            x, lw, lb, eps, w4, wbf, wo16, gb if gb is not None else E32, ob if ob is not None else E32, gb is None and ob is None, C,
            mask if mask is not None else _empty_mask(pair.device), B, Lp, trans, trans, bt, lr, D**-0.5, out if out is not None else E,
        )
        return (out if out is not None else y.view(B, L, L, C)), (prm, w4, wo16, E, E, E, E, E, E, E)
    if L128_TRAIN and L128_TRAIN_D128 and Lp == TOK and need and not bt and gb is None and ob is None:
        if drop is not None and PENDING_DROP:
            s = PENDING_DROP.pop(drop.data_ptr(), None)
            if s is not None:
                _side_mod().join(x.device, s)
        y, q, k, v, g, o, lse, bias = _l128().tri_l128_train(
            x, lw, lb, eps, w4, wbf, wo16, C, mask if mask is not None else _empty_mask(pair.device), B, Lp, trans, lr,
            D**-0.5, drop if drop is not None else x.new_empty(0), out if out is not None else x.new_empty(0), x.new_empty(0, dtype=torch.float32),
        )
        return (out if out is not None else y.view(B, L, L, C)), (prm, w4, wo16, q, k, v, g, bias, o, lse)
    # inference: the tail projects the gate from x (tri_tailg, the H100 inference tail) and the front writes q / k / v only
    # (at L128 -- one tile a CTA -- the gate-from-x tail's longer per-tile chain costs more than the stream it saves: the 4-output front +
    # tri_tail stay there, unless the gate / out biases need the fused tail; B200 2026-11-17: d_pair 128 L128 eval 34.7 vs 36.7 us)
    gx = (gb is not None or ob is not None or (INFERENCE_TAIL_GATE and Lp > TOK)) and not need and drop is None
    hb = HB_INFER and not need  # inference: the bias as bf16(bias * log2 e) for the core's HB build (bias rows scaled in fp32)
    if hb:
        wbf = _capture.param_forms("triattn_b200_wb_l2e", (wb,), lambda: (wb.detach().float() * L2E).contiguous(), capture_reads_eager=True)
    qs = qs_for(hb, D, Lp)
    w4f = w4
    if qs:  # the core's QS build: q = LN(x) . (Wq * scale * log2 e)^T (one rounding of the scaled weight, as the unscaled one before)
        qsc = D**-0.5 * L2E

        def pack_qs():
            return m.tri_pack_params(
                (wq.detach().float() * qsc).to(wq.dtype).contiguous(), wk.contiguous(), wv.contiguous(), wg.contiguous(), wo.contiguous(),
                lnw.contiguous(), lnb.contiguous(), wb.contiguous(),
            )

        w4f = _capture.param_forms("triattn_b200_pack_qs", (wq, wk, wv, wg, wo, lnw, lnb, wb), pack_qs, capture_reads_eager=True)[0]
    qk8 = _qk8_front("fused", wv, Lp, H, x.device) if hb and gx else None  # e4m3 q / k (opt-in, off by default): the front writes them
    if qk8 is not None:
        q, k, v, g, bias = m.tri_front_q8(x, lw, lb, eps, w4f, wbf, mask if mask is not None else _empty_mask(pair.device), B, Lp, trans, bt, lr, *qk8)
    else:
        q, k, v, g, bias = m.tri_front(
            x, lw, lb, eps, w4f, wbf, mask if mask is not None else _empty_mask(pair.device), B, Lp, trans, bt, lr, gx
        )
    shp = (B, Lp, Lp, H, D)
    o, lse = _core_forward(q.view(shp), k.view(shp), v.view(shp), bias, D**-0.5, need, hb, qs=qs, kind="fused", site=wv)
    E32 = x.new_empty(0, dtype=torch.float32)
    if gx:
        y = m.tri_tailg(
            o.view(-1, C),
            x,
            wo16,
            w4[3 * C :],
            lw,
            lb,
            eps,
            Lp,
            out if out is not None else x.new_empty(0),
            trans,
            lr,
            gb if gb is not None else E32,
            ob if ob is not None else E32,
        )
    else:
        assert gb is None and ob is None, "the fused path's gate / out biases are inference-only"
        if drop is not None and PENDING_DROP:
            s = PENDING_DROP.pop(drop.data_ptr(), None)
            if s is not None:
                _side_mod().join(x.device, s)
        y = m.tri_tail(
            g, o.view(-1, C), x, wo16, drop if drop is not None else x.new_empty(0), Lp, out if out is not None else x.new_empty(0), trans, lr
        )
    return (out if out is not None else y.view(B, L, L, C)), (prm, w4, wo16, q, k, v, g, bias, o, lse)


def split_grads(fused, raw, dtypes):
    """(pair, lnw, lnb, wq, wk, wv, wg, wb, wo[, gate bias, out bias[, Wo_t, out_t bias]]) gradients from a backward's stacked outputs
    (views of them, cast to ``dtypes``, the parameters' dtypes, where they differ); the bias gradients when ``dtypes`` has ten entries,
    also the dual output's second out-projection (Chai-1's ``to_out_t``) when it has twelve (None for an absent parameter: an empty
    stacked gradient)."""
    dbias = dwo2 = None
    if fused:
        dpair, dgb, dw, dwo = raw  # dgb = [dgamma; dbeta; dWb (4 rows)], dw = [dWq; dWk; dWv; dWg]
        wb = dgb[2:]
    else:
        # dgb = [dgamma; dbeta] fp32; rest = [[dbg; dbo]] (the gate / out biases)[, dWo_t (empty unless the dual output)]
        dpair, dgb, dw, wb, dwo, *rest = raw
        dbias = rest[0] if rest else None
        dwo2 = rest[1] if len(rest) > 1 and rest[1].numel() else None
    grads = (
        dpair,
        dgb[0].to(dtypes[0]),
        dgb[1].to(dtypes[1]),
        *(dw[i].to(dtypes[2 + i]) for i in range(4)),
        wb.to(dtypes[6]),
        dwo.to(dtypes[7]),
    )
    hd = dw.shape[1]
    if len(dtypes) > 8:  # (gate bias, out bias) dtypes, None for an absent bias
        dbg = dbo = None
        if dbias is not None:
            dbg = dbias[:hd].to(dtypes[8]) if dtypes[8] is not None else None
            dbo = dbias[hd:].to(dtypes[9]) if dtypes[9] is not None else None
        grads += (dbg, dbo)
    if len(dtypes) > 10:  # (Wo_t, out_t bias): the second update's out bias sums the same dy_s rows as the first's
        dwt = dwo2.to(dtypes[10]) if dwo2 is not None and dtypes[10] is not None else None
        dbt = dbias[hd:].to(dtypes[11]) if dwo2 is not None and dbias is not None and dtypes[11] is not None else None
        grads += (dwt, dbt)
    return grads


def fused_backward(dy, pair, eps, drop, saved, dtypes, trans=False):
    """Stacked gradients (dpair, [dgamma; dbeta; dWb], [dWq; dWk; dWv; dWg], dWo) of `fused_forward` (`split_grads` takes them apart);
    dtypes = the eight parameters' dtypes (the weight gradients come back in fp32 when Wq is fp32, else bf16).
    Kernels: tri_gate_bwd (+ the core's bias^T on the CTAs beyond the tiles, or tri_bwd_pre) -> triattn_bwd (key side, query side:
    dbias in per-CTA slabs) -> tri_head_bwd (sums the slabs in order, writes dbias) -> tri_wgrad (+ slot sum, finish)."""
    prm, w4, wo16, q, k, v, g, bias, o, lse = saved
    B, L, _, _ = pair.shape
    Lp, lr = padded_length(L), ragged(L)
    x = pair.contiguous().view(-1, C)
    lw, lb, wbf = prm[0], prm[1], prm[2:]
    m, e = _mod(), _core(_rows_per_task(Lp))
    det = DETERMINISTIC_WGRAD
    fp32 = dtypes[2] == torch.float32
    E = q.new_empty(0)
    E32 = x.new_empty(0, dtype=torch.float32)
    if det:
        # the parameter gradients' partials go to per-CTA slots (no zeroing), summed in CTA order (tri_slot_sum); dbias: the core writes
        # every element (deterministic slabs)
        wdwo = wmt = E32
        dbws = None
    else:
        # one zeroed fp32 workspace for every gradient accumulator (one memset): [dWo C*C | wgrad M 4*C*C | wgrad vectors 8*C + 4,
        # padded | dbias B*4*L*L]; each piece starts on a 128-byte boundary
        ws = torch.zeros(_WS_DB + B * H * Lp * Lp, device=x.device, dtype=torch.float32)
        dbws, wdwo, wmt = ws[_WS_DB:], ws[_WS_DWO:_WS_MT], ws[_WS_MT:_WS_DB]
    dy = dy.contiguous().view(-1, C)
    drop = drop if drop is not None else x.new_empty(0)
    gbt = det and GATE_BIAST and bias.dtype == torch.bfloat16
    do, dg, delta, dwo, dpair, bT = m.tri_gate_bwd(dy, g, o.view(-1, C), wo16, B, Lp, wdwo, drop, det, trans, lr, bias if gbt else E)
    if det:
        if bT.numel() == 0:
            bT = None
            if BWD_PRE and bias.dtype == torch.bfloat16:
                _, bT, dbws = _wide().tri_bwd_pre(x[:0], x.new_empty(0), Lp, False, bias, False)
        if dbws is None:
            dbws = torch.empty(B * H * Lp * Lp, device=x.device, dtype=torch.float32)
    else:
        bT = None
    shp = (B, Lp, Lp, H, D)
    raw = False  # (the slabs summed by tri_head_bwd: measured and rejected, see docs/b200/triattn.md round 7)
    side = _side_mod() if KVQ_BRANCH and not raw else None
    s_ = side.fork(x.device) if side is not None else None
    if s_ is not None:  # (S90) the key-side and the query-side kernels as two branches of the captured graph
        with _on(side, s_):
            res_q = e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, D**-0.5, dbws.view(-1), E, E, E, bT if bT is not None else E, raw, False, 2, E)
        res_kv = e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, D**-0.5, dbws.view(-1), E, E, E, bT if bT is not None else E, raw, False, 1, E)
        side.join(x.device, s_)
        res = [res_q[0], res_kv[1], res_kv[2], res_q[3]]
    else:
        res = e.triattn_bwd(
        q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, D**-0.5, dbws.view(-1), E, E, E,
        bT if bT is not None else E, raw, False, 0, E,
    )
    dq, dk, dv = res[0].view(-1, C), res[1].view(-1, C), res[2].view(-1, C)
    db = dbws.view(B, H, Lp, Lp) if raw else res[3]
    # fp32 (master) parameters: the gate backward's dWo partials summed by head_bwd's CTAs beyond the tiles (fewer tiles than SMs),
    # else in tri_wgrad's slot sum
    hwo = False  # (the dWo slot sum on head_bwd's CTAs beyond the tiles: measured and rejected, docs/b200/triattn.md round 7)
    nt = B * Lp * Lp // TOK
    if det and HW_SPLIT and not raw and HW_MIN <= nt <= HW_TILES:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        nh = HW_NH or round(sms * (60 if nt <= 512 else 64) / 148)
        dw, dgb, dwo = m.tri_hw_bwd(dq, dk, dv, dg, db, x, w4, prm, eps, dpair, B, Lp, trans, lr, dwo, fp32, nh, HW_NW or sms - nh, HW_NG, WFOLD)
        return dpair.view(B, L, L, C), dgb, dw, dwo
    dwo_h = m.tri_head_bwd(
        dq, dk, dv, dg, db, x, w4, wbf, lw, lb, eps, dpair, B, Lp, trans, lr, False, res[3] if raw else E32, res[4] if raw else E,
        dwo if hwo else E32,
    )
    done = dwo_h.numel() > 0
    dw, dgb, dwo = m.tri_wgrad(dq, dk, dv, dg, db, x, eps, B, Lp, w4, wbf, lw, lb, True, E32 if done else dwo, wmt, fp32, det, trans, lr, WFOLD if det and not done else 0)
    if done:
        dwo = dwo_h
    return dpair.view(B, L, L, C), dgb, dw, dwo


#: The fused backward's set-up on the gate backward's idle CTAs: the core's bias^T when there are fewer row tiles than SMs (L128, one
#: sample; B200 2026-11-17: no tri_bwd_pre launch).
GATE_BIAST = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_GATE_BIAST", "1") != "0"

#: (training tails round 10) the gate backward's dWo partials (ready before tri_wgrad starts) summed in CTA order by tri_wgrad's row warps
#: after their xhat loop (idle while the MMAs and the flush run) and written in the parameters' dtype: tri_slot_sum loses its dWo
#: segment (8-9.5 MB of the 27 MB it read at L128) and tri_param_finish its 16 cast blocks.  Module training step (graph, d_pair 128,
#: off -> on, us): L128 113.5 -> 111.9 (starting), 114.0 -> 112.4 (ending), L256 286.4 -> 285.2, L384 634.0 -> 632.3, L512 1213.6 ->
#: 1212.1, L640 within noise.  WFOLD = 2 (default).  1: every tail folded into tri_wgrad (grid barrier, the M_t / vector slots summed
#: and the finish on the wgrad CTAs, dgamma / dbeta by the last CTA): measured slower (L128 +0.4 us, L256 / 384 +6 us: the slot sum
#: is ~16-27 MB of L2 reads, not a launch overhead) -- kept opt-in only.  0: the separate launches.
WFOLD = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_WFOLD", "2"))

#: (training tails round 9) tri_head_bwd and tri_wgrad as one CTA-split launch (tri_hw_bwd: the two read the same inputs and not each
#: other's outputs): head_bwd on ~60-64 of the 148 SMs, wgrad (2 tensor groups) on the rest -- one launch fewer and ~40 % fewer wgrad
#: slots for the slot sum.  Module training step (graph, d_pair 128, starting / ending, us): L256 294.7 -> 287.9 / 298.0 -> 290.6 (nh 60),
#: L384 645.0 -> 633.4 / 653.3 -> 639.9, L512 1229.8 -> 1218.4 / 1247.6 -> 1232.6 (nh 64); L640 / 768 within 0.7 % (not taken);
#: L128 (one tile a CTA) 111.6 -> 109.9-113.3 by the split, i.e. no gain: head_bwd's and wgrad's per-tile chains run in series per CTA,
#: so fewer CTAs a role just lengthen each role (globaltimer: the split kernel 25.8 us vs 11.3 + 15.7).  Row tiles HW_MIN .. HW_TILES.
HW_SPLIT = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HW", "1") != "0"
HW_MIN = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HW_MIN", "512"))
HW_TILES = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HW_TILES", "2048"))
HW_NH = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HW_NH", "0"))
HW_NW = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HW_NW", "0"))
HW_NG = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HW_NG", "0"))


class TriAttnFused(torch.autograd.Function):
    """The whole module (residual included): pair [B, L, L, 128] -> pair + TriangleAttention(pair).
    Parameters may be bf16 or fp32 (master weights): one pack kernel stages them, and the gradients come back in their dtype.
    drop: the module's broadcast dropout scale as [B, L, 128] bf16 (index = the row's last position), or None.
    (The module path goes through the custom ops of ``integrations.triattn_b200``, which call `fused_forward` / `fused_backward`.)"""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop):
        need = any(ctx.needs_input_grad)
        out, saved = fused_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, need)
        if need:
            ctx.save_for_backward(pair, drop if drop is not None else pair.new_empty(0), *saved)
            ctx.eps = eps
            ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, drop, *saved = ctx.saved_tensors
        raw = fused_backward(dy, pair, ctx.eps, drop if drop.numel() else None, saved, ctx.dtypes)
        return (*split_grads(True, raw, ctx.dtypes), None, None, None)


def triangle_attention(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None):
    """pair + drop o TriangleAttention(pair) for the starting node; the caller transposes for the ending node."""
    return TriAttnFused.apply(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop)


# ------------------------------------------------------------------------------------------------------------------------------
# Other widths: d_pair C in 64 .. 512 (a multiple of 64), heads of 16, 32, 64, 96 or 128 channels. Forward: tri_wfront (LN + the five
# projections as 2-CTA products, weights streamed in 64-column chunks; row statistics kept when training) -> the head-dim-32
# core above (a head dim of 16 is zero-padded to 32), the head-dim-64 core (triattn_dh64_sm100.cu, forward + backward) or the
# head-dim-128 core (triattn_dh128_sm100.cu, forward + backward; a head dim of 96 is zero-padded to 128) -> tri_wtail (sigmoid gate +
# out-projection + dropout + residual).
# The model options of FoldJet's families ride on the same kernels: the gate / out-projection biases (RoseTTAFold3; the front's
# drain adds bg to g, the tail's drain bo to the update), AlphaFold 3's ending-node bias orientation (``bt``: the front writes the
# bias of the pair as it is; the backward reads its gradient so), and Chai-1's outputs (``out_mode``): 1 = the ending node's update
# left as the transposed pair has it (the tail writes the starting-node layout), 2 = two out-projections, the second tail adding its
# update transposed into the first's output in place (starting: kept + other^T, ending: kept^T + other).  The dropout scale stays on
# the caller's broadcast axis, which for those tails lies across the kernel rows (``dfirst``); their backward runs the gate backward in
# the update's orientation (untransposed) or over both orientations at once (dual: du = dy_s . Wo + dy_s^T . Wo2, u kept in both).
# Backward (WideTrain): tri_scale_rows (dropout) -> tri_wgbwd (du = dy_s . Wo on tcgen05, the gate math and delta in its
# drain) -> the core's backward -> tri_whbwd ([dq | dk | dv | dg | db] . W on tcgen05, the LN backward and the residual in its
# drain, xhat | 1 written) -> the weight gradients from G = D^T [xhat | 1] (one cuBLAS GEMM, K = L^2) by the algebra of the
# d_pair 128 path, finished by tri_wfinish. All hand kernels in triattn_wide_sm100.cu.
WIDE_HEAD_DIMS = (16, 32, 64, 96, 128)
TRAIN_HEAD_DIMS = (16, 32, 64, 96, 128)  # every head dim has a backward (64: triattn_dh64_sm100.cu, 96 / 128: triattn_dh128_sm100.cu)
WIDE_MAX_HEADS = 16  # the packed bias block is 16 rows
WIDE_MAX_HDP = 512  # the front's q | k | v | g blocks and the tail's K
#: (d_pair, heads x slot) whose inference runs the tail with the gate projected from x (``tri_wtailg``; ``INFERENCE_TAIL_GATE``)
WIDE_TAIL_GATE = ((64, 64), (64, 128))
#: d_pair 64 with heads x 32 = 64 / 128 (<= 4 heads): the front of the fused path (``tri_front64``) instead of ``tri_wfront``
WIDE_FRONT64 = True


@functools.lru_cache(maxsize=1)
def _wide():
    ensure_cuda_home()
    return load_extension(
        name="triattn_b200_wide_v45_tt7", sources=[str(_SRC / "triattn_wide_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False
    )


#: the wide backward's two weight-gradient GEMMs (G = Dall^T [xhat | 1], dWo = dy_s^T u) in one tcgen05 launch, split-K over a CTA cluster
#: reduced in rank order (``tallk_sm100.cu``; deterministic) instead of two cuBLAS split-K GEMMs + their reduce kernels.
#: FOLDJET_ENGINE_TRIATTN_B200_TALLK=0 switches back to cuBLAS.
TALLK = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_TALLK", "1") != "0"
#: it wins while the reduction is short (one 128-row tile set a CTA cluster, latency bound): measurer, training, L128 d_pair 64 -5 to -7 %,
#: d_pair 256 -1 to -4 %; at L >= 256 (+2 to +3 %) and d_pair >= 384 (+2 to +8 %) cuBLAS's split-K GEMMs are faster (B200 2026-11-17)
TALLK_MAX_ROWS = 128 * 128

#: the backward's set-up passes (dy o drop, the head-dim-32 core's bias^T, its zeroed dbias accumulator) in one launch (tri_bwd_pre)
#: instead of three (B200 2026-11-17).  FOLDJET_ENGINE_TRIATTN_B200_BWD_PRE=0: the separate launches.
BWD_PRE = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_BWD_PRE", "1") != "0"


@functools.lru_cache(maxsize=1)
def _tallk():
    ensure_cuda_home()
    return load_extension(name="triattn_b200_tallk_v2", sources=[str(_SRC / "tallk_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False)


#: B200 2026-11-18 (round 3 part 14): inference with 64-channel heads takes the head-dim-64 core's HB build -- the head-dim-32 inference
#: core's softmax loop (bias pre-scaled by log2 e and added without unpack, the offset fixed at the first 32 keys with the overflow guard
#: after the tile instead of a per-chunk running maximum, POLY64_INFER of 16 exponential pairs on the FMA pipe, 16-key chunks).
#: FOLDJET_ENGINE_TRIATTN_B200_HB64=0 turns it off (the training build, as before).
HB64_INFER = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_HB64", "1") != "0"
POLY64_INFER = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_POLY64", "2"))
#: ... the QS build of the head-dim-64 core (qs_for: q pre-scaled by head_dim^-0.5 * log2 e; above QS_MIN_L64 padded tokens)
POLY64QS_INFER = int(os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_POLY64QS", "2"))


@functools.lru_cache(maxsize=None)
def _core64(hb: bool = False, poly: int | None = None, qs: bool = False):
    """The head-dim-64 core; hb: its inference build (TA_HB, bias pre-scaled by log2 e; no LSE), poly: TA_POLY (None: POLY64_INFER)."""
    ensure_cuda_home()
    if not hb:
        return load_extension(
            name="triattn_b200_dh64_v7_tt7", sources=[str(_SRC / "triattn_dh64_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False
        )
    p = (POLY64QS_INFER if qs else POLY64_INFER) if poly is None else poly
    return load_extension(
        name=f"triattn_b200_dh64_hb_p{p}{'_qs' if qs else ''}_v1", sources=[str(_SRC / "triattn_dh64_sm100.cu")],
        extra_cuda_cflags=[*_flags(), "-DTA_HB=1", "-DTA_CK=16", f"-DTA_POLY={p}", *(("-DTA_QS=1",) if qs else ())], verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _core128():
    ensure_cuda_home()
    return load_extension(
        name="triattn_b200_dh128_v4_tt7", sources=[str(_SRC / "triattn_dh128_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False
    )


def slot(head_dim: int) -> int:
    """Channels a head takes in the packed q | k | v | g layout: 32 (16 zero-padded), 64, or 128 (96 zero-padded: its padded q / k / v /
    g channels come out 0, meet zero columns of Wo and add nothing to the scores; the softmax scale stays 1/sqrt(96))."""
    return 32 if head_dim <= 32 else 64 if head_dim == 64 else 128


def wide_supports(d_pair: int, d_hidden: int, n_head: int, length: int, training: bool = False) -> bool:
    """Shapes the wide path takes (``training``: with a backward)."""
    if d_pair % 64 or not 64 <= d_pair <= 512 or d_hidden % n_head or not supports(length):
        return False
    hd = d_hidden // n_head
    return hd in (TRAIN_HEAD_DIMS if training else WIDE_HEAD_DIMS) and n_head <= WIDE_MAX_HEADS and n_head * slot(hd) <= WIDE_MAX_HDP


def pack_wide_weights(wq, wk, wv, wg, wb, wo, gb=None, ob=None, wo2=None, ob2=None, l2e=False, qs=False):
    """-> (wcat, wo_p, wo_t, wall_t, gb_p, ob, wo2_p, ob2, cpk), HDp = heads x slot(head dim) (a 16-channel head is zero-padded to 32, a
    96-channel head to 128: its padded q / k / v / g channels come out 0 and meet zero columns of Wo):
      wcat   [Wq; Wk; Wv; Wg; Wb (16 rows)]  [4 HDp + 16, C] bf16   the front's B
      wo_p   Wo                               [C, HDp] bf16          the tail's B
      wo_t   Wo^T                             [HDp, C] bf16          the gate backward's B ([Wo^T | Wo2^T] [HDp, 2C] with wo2)
      wall_t [Wq; Wk; Wv; Wg; Wb (64 rows)]^T [C, 4 HDp + 64] bf16   the head backward's B
      gb_p   the gate bias in the padded layout, fp32 [HDp], or None
      ob     the out-projection bias, fp32 [C], or None
      wo2_p, ob2  the dual output's second out-projection (Chai-1's ``to_out_t``) as wo_p / ob, or None
    l2e: the bias rows times log2 e (multiplied in fp32 before the one bf16 rounding): the inference pack of the HB core (HB_INFER);
    with 16-channel heads it also gives cpk = (compact [Wq; Wk; Wv; Wg; Wb] [4 H 16 + 16, C], compact Wo [C, H 16]) (D16C_INFER), else None"""
    n_head = wb.shape[0]
    if l2e:
        wb = wb.float() * L2E
    hd = wq.shape[0] // n_head
    S = slot(hd)
    bf = torch.bfloat16
    f32 = lambda t: None if t is None else t.float().contiguous()  # noqa: E731

    def pad_bias(b):  # the gate bias in the padded layout, fp32 [HDp]
        if b is None:
            return None
        out = b.new_zeros(n_head, S, dtype=torch.float32)
        out[:, :hd] = b.float().view(n_head, hd)
        return out.view(-1)

    # one launch for every pack (tri_wpack; the casts round as torch's). As torch statements (casts, zero fills, cats, transposes) the
    # pack was a dozen and a half launches, replayed by a training capture at every step: ~50 us of a 140-170 us step at L128
    # (B200 2026-11-17).
    if l2e and qs and hd in (D, 16, 64):  # the core QS build (32 / 16-channel heads, `qs_for`): Wq * scale * log2 e (one rounding; wide_forward runs the QS core)
        wq = (wq.detach().float() * (hd**-0.5 * L2E)).to(wq.dtype)
    ws = [wq, wk, wv, wg, wb, wo, wo2 if wo2 is not None else wq.new_empty(0)]
    if wq.dtype not in (torch.float32, bf) or any(w.dtype != wq.dtype for w in ws):
        ws = [w.float() for w in ws]  # mixed or other dtypes: through fp32 (exact for bf16)
    wcat, wo_p, wo_t, wall_t, wo2_p = _wide().tri_wpack(*(w.contiguous() for w in ws), S)
    cpk = None
    if l2e and hd == 16 and n_head * hd in (64, 128):
        # the compact 16-channel-head pack of the inference front / tail (D16C_INFER): [Wq; Wk; Wv; Wg] without the padded rows, then the
        # bias rows; Wo without the padded columns (the same bf16 values as the padded pack)
        C = wcat.shape[1]
        w4c = wcat[: 4 * n_head * S].view(4, n_head, S, C)[:, :, :hd].reshape(4 * n_head * hd, C)
        cpk = (torch.cat([w4c, wcat[4 * n_head * S :]]).contiguous(), wo_p.view(-1, n_head, S)[:, :, :hd].reshape(-1, n_head * hd).contiguous())
    return wcat, wo_p, wo_t, wall_t, pad_bias(gb), f32(ob), wo2_p if wo2 is not None else None, f32(ob2), cpk


def wide_inference(pair, lnw, lnb, packed, n_head, head_dim, mask=None, eps=1e-5, out=None, trans=False, bt=False, out_mode=0, hb=False):
    """pair + TriangleAttention(pair) for the starting node, no grad; packed = pack_wide_weights(...).
    Three launches (four for the dual output): tri_wfront (LN + the five projections, head-major masked bias) -> triattn_fwd /
    triattn_fwd64 -> tri_wtail (sigmoid gate + out-projection + residual).  `out`: a contiguous bf16 tensor like the pair to write
    into, or None.  hb: `packed` holds the bias rows times log2 e (`pack_wide_weights(.., l2e=True)`): the core's HB build (32-channel slots)."""
    y, _ = wide_forward(pair, lnw, lnb, packed, n_head, head_dim, mask, eps, None, False, out, trans, bt, out_mode, hb)
    return y


def _core64_forward(q, k, v, bias, scale, need=False, hb=False, qs=False, kind=None, site=None):
    """The head-dim-64 core over [B, N, S, H, 64] tensors -> (out [B, N, S, H, 64], base-2 lse [B, N, H, S] or empty unless `need`);
    hb: the bias is bf16(bias * log2 e) (inference: HB64_INFER)."""
    assert not (hb and need)
    if kind is not None and hb:  # opt-in e4m3 P / V core of the 64-channel heads (docs/b200/fp8.md "TriAttn"; the whitelist kind "d64")
        from . import h100_sm90 as _hs

        if _hs.FP8_PV and _hs.FP8_PV_D64:
            from . import b200_fp8

            o8 = b200_fp8.route(kind, q, k, v, bias, scale, qs, site)
            if o8 is not None:
                return o8, q.new_empty(0, dtype=torch.float32)
    o, lse, _ = _core64(hb, qs=qs).triattn_fwd64(q, k, v, bias, scale, q.new_empty(0), need, q.new_empty(0, dtype=torch.float32))
    return o, lse


def _core128_forward(q, k, v, bias, scale, need=False):
    """The head-dim-128 core over [B, N, S, H, 128] tensors -> (out [B, N, S, H, 128], base-2 lse [B, N, H, S] or empty unless `need`)."""
    o, lse, _ = _core128().triattn_fwd128(q, k, v, bias, scale, q.new_empty(0), need, q.new_empty(0, dtype=torch.float32))
    return o, lse


def wide_forward(
    pair, lnw, lnb, packed, n_head, head_dim, mask=None, eps=1e-5, drop=None, need=False, out=None, trans=False, bt=False, out_mode=0, hb=False
):
    """-> (y [B, L, L, C], saved = (q, k, v, g, bias, o, lse, stats)): lse / stats empty unless `need`; every saved tensor is a fresh
    allocation (none aliases an argument).  bt / out_mode: see the section header."""
    B, L, _, C = pair.shape
    Lp, lr = padded_length(L), ragged(L)
    wcat, wo_p, _, _, gb, ob, wo2, ob2 = packed[:8]
    cpk = packed[8] if len(packed) > 8 else None
    S = slot(head_dim)
    HDp = n_head * S
    w = _wide()
    E32 = pair.new_empty(0, dtype=torch.float32)
    x = pair.contiguous().view(-1, C)
    stats = torch.empty(B * Lp * Lp, 2, device=x.device, dtype=torch.float32) if need else E32
    # inference at d_pair 64 (heads x slot 64 / 128), standard output: the tail projects the gate from x (tri_wtailg), the front skips g
    nog = INFERENCE_TAIL_GATE and not need and drop is None and out_mode == 0 and (C, HDp) in WIDE_TAIL_GATE
    mk = mask if mask is not None else _empty_mask(pair.device)
    assert not (hb and (need or S not in (D, 64))), "the pre-scaled bias is for the head-dim-32 / 64 cores' inference builds"
    if L128_FUSED and not hb and Lp == TOK and C == 64 and S == D and HDp in (64, 128) and n_head <= 4 and not need and drop is None and out_mode in (0, 1):
        # L = 128: the whole update in one CTA a pair row (tri_l128; the gate in fp32 as tri_wtailg; Chai-1's untransposed output: out_mode 1)
        y = _l128().tri_l128(
            x, lnw, lnb, eps, wcat, E32, wo_p, gb if gb is not None else E32, ob if ob is not None else E32, False, HDp, mk, B, Lp, trans,
            trans and out_mode != 1, bt, lr, head_dim**-0.5, out if out is not None else x.new_empty(0),
        )
        return (out if out is not None else y.view(B, L, L, C)), (E32,) * 8
    if L128_TRAIN and Lp == TOK and C == 64 and S == D and HDp in (64, 128) and n_head <= 4 and need and out_mode == 0 and not bt and gb is None and ob is None:
        # training at L = 128 (round 8): the fused kernel's training form also stores q | k | v | g | o, the LSE, the bias and the LN statistics
        y, q, k, v, g, o, lse, bias = _l128().tri_l128_train(
            x, lnw, lnb, eps, wcat, E32, wo_p, HDp, mk, B, Lp, trans, lr, head_dim**-0.5, drop if drop is not None else x.new_empty(0),
            out if out is not None else x.new_empty(0), stats,
        )
        return (out if out is not None else y.view(B, L, L, C)), (q, k, v, g, bias, o, lse, stats)
    d16 = False
    if WIDE_FRONT64 and C == 64 and S == D and HDp in (64, 128) and n_head <= 4 and (gb is None or nog):
        # d_pair 64 with heads of 16 (padded) / 32 channels: the fused front's design (one CTA, the weights resident, the bias on the CUDA
        # cores of the LayerNorm warps) instead of the 2-CTA wide front, whose per-block weight stream and handoffs dominate at K = 64
        d16 = hb and D16_INFER and head_dim == 16
        if d16 and D16C_INFER and nog and cpk is not None:
            # compact 16-channel heads (D16C_INFER): the front at heads x 16, the compact core, the tail at K = heads x 16
            wc, woc = cpk
            HDc = n_head * head_dim
            q, k, v, g, bias = _mod().tri_front64(x, lnw, lnb, eps, wc, n_head, mk, B, Lp, stats, trans, bt, lr, True, False)
            shp = (B, Lp, Lp, n_head, head_dim)
            o, lse = _core_forward(q.view(shp), k.view(shp), v.view(shp), bias, head_dim**-0.5, False, hb=True, d16=True, d16c=True, qs=qs_for(hb, head_dim, Lp))
            y = w.tri_wtailg(
                o.view(-1, HDc), x, woc, wc[3 * HDc : 4 * HDc], lnw, lnb, eps, Lp, out if out is not None else x.new_empty(0), trans, lr, E32,
                ob if ob is not None else E32,
            )
            return (out if out is not None else y.view(B, L, L, C)), (q, k, v, g, bias, o, lse, stats)
        qk8 = _qk8_front("unfused", wcat, Lp, n_head, x.device) if hb and nog and not d16 and not need and S == D else None
        if qk8 is not None:  # e4m3 q / k (opt-in, off by default): the front writes them
            q, k, v, g, bias = _mod().tri_front64_q8(x, lnw, lnb, eps, wcat, n_head, mk, B, Lp, trans, bt, lr, *qk8)
        else:
            q, k, v, g, bias = _mod().tri_front64(x, lnw, lnb, eps, wcat, n_head, mk, B, Lp, stats, trans, bt, lr, nog, d16)
    else:
        q, k, v, g, bias = w.tri_wfront(
            x, lnw, lnb, eps, wcat, n_head, mk, B, Lp, stats, trans, gb if gb is not None and not nog else E32, bt, lr, nog
        )
    shp = (B, Lp, Lp, n_head, S)
    core = functools.partial(_core_forward, hb=hb, d16=d16, qs=qs_for(hb, head_dim, Lp), kind="unfused", site=wcat) if S == D else functools.partial(_core64_forward, hb=hb, qs=qs_for(hb, head_dim, Lp), kind="d64", site=wcat) if S == 64 else _core128_forward
    o, lse = core(q.view(shp), k.view(shp), v.view(shp), bias, head_dim**-0.5, need)
    o2 = o.view(-1, HDp)
    if nog:
        y = w.tri_wtailg(
            o2,
            x,
            wo_p,
            wcat[3 * HDp : 4 * HDp],
            lnw,
            lnb,
            eps,
            Lp,
            out if out is not None else x.new_empty(0),
            trans,
            lr,
            gb if gb is not None else E32,
            ob if ob is not None else E32,
        )
        return (out if out is not None else y.view(B, L, L, C)), (q, k, v, g, bias, o, lse, stats)
    dt = drop if drop is not None else x.new_empty(0)
    # the first (or only) out-projection: the ending node's update goes back to the pair's layout, except Chai-1's untransposed output;
    # the dropout scale is indexed on the caller's broadcast axis: the kernel row's first position where the tail's orientation is not
    # the node's (dfirst)
    y = w.tri_wtail(
        g,
        o2,
        x,
        wo_p,
        dt,
        Lp,
        out if out is not None else x.new_empty(0),
        trans and out_mode != 1,
        ob if ob is not None else E32,
        trans and out_mode == 1,
        lr,
    )
    if out_mode == 2:  # the second out-projection, transposed against the first, added in place
        y = w.tri_wtail(g, o2, y.view(-1, C), wo2, dt, Lp, y, not trans, ob2 if ob2 is not None else E32, True, lr)
    return (out if out is not None else y.view(B, L, L, C)), (q, k, v, g, bias, o, lse, stats)


def _mm32(a, b):
    """a @ b on cuBLAS with an fp32 result (bf16 operands, fp32 accumulation). cuBLAS 12.9's bf16 split-K kernels drop the last K % 64
    rows of a long reduction on B200 for some K (measured 2026-10-29), so a long reduction that is not a multiple of 64 (the caller's
    B L^2 rows at a ragged length) runs as its multiple-of-64 head plus the last K % 64 rows, both with fp32 results."""

    def mm(x, y):
        try:
            return torch.mm(x, y, out_dtype=torch.float32)
        except (TypeError, RuntimeError):  # a torch without mm(out_dtype=)
            return torch.mm(x.float(), y.float())

    k = a.shape[1]
    if k < 1024 or k % 64 == 0:
        return mm(a, b)
    h = k - k % 64
    out = mm(a[:, :h], b[:h])
    out += mm(a[:, h:], b[h:])
    return out


def wide_backward(dy, pair, lnw, lnb, packed, n_head, head_dim, eps, drop, saved, dtypes, trans=False, bt=False, out_mode=0):
    """Stacked gradients (dpair, [dgamma; dbeta] fp32, [dWq; dWk; dWv; dWg], dWb, dWo, [dbg; dbo], dWo_t) of `wide_forward`
    (`split_grads` takes them apart; the weight gradients in fp32 when Wq is fp32, else bf16; [dbg (d_hidden); dbo (C)] fp32, zeros for a
    bias the module does not have (unwritten when it has neither) -- one tensor: a custom op's outputs may not alias, and two empty ones would; dWo_t: the dual output's
    second out-projection (its bias's gradient is dbo: the same dy_s rows), else empty)."""
    q, k, v, g, bias, o, lse, stats = saved
    B, L, _, C = pair.shape
    Lp, lr = padded_length(L), ragged(L)
    x = pair.contiguous().view(-1, C)
    wcat, _, wo_t, wall_t, gb, ob = packed[:6]
    S = slot(head_dim)
    HDp = n_head * S
    R = B * Lp * Lp
    w = _wide()
    dy = dy.contiguous().view(-1, C)
    # dy_s in the caller's layout and length (as u below; elementwise over the caller's rows); the dropout scale on the node's
    # broadcast axis
    dr = drop is not None and drop.numel() > 0
    bT = dbz = None
    if BWD_PRE and S == D and bias.dtype == torch.bfloat16:
        # one launch: dy_s, the core's bias^T and its zeroed dbias accumulator (head dim 32; B200 2026-11-17)
        dys, bT, dbz = w.tri_bwd_pre(dy, drop if dr else x.new_empty(0), L, trans, bias, False)
    else:
        dys = w.tri_scale_rows(dy, drop, L, trans) if dr else dy
    Dall = x.new_empty(R, 4 * HDp + 64)  # [dq | dk | dv | dg | db rows], written in place by their producers
    cut = lambda i, n=HDp: Dall[:, i * HDp : i * HDp + n]
    # the gate backward in the update's orientation (Chai-1's untransposed output: the starting-node layout); the dual output's
    # wo_t is [Wo^T | Wo2^T] (both orientations in one K loop, u also stored in the other: u2)
    do, _, u, delta, u2 = w.tri_wgbwd(dys, g, o.view(-1, HDp), wo_t, cut(3), B, Lp, trans and out_mode != 1, lr)
    shp = (B, Lp, Lp, n_head, S)
    E32 = q.new_empty(0, dtype=torch.float32)
    if S == D:
        e = _core(_rows_per_task(Lp))
        dbw, btw = (dbz.view(-1), bT) if dbz is not None else (E32, E32)
        dbr = cut(4, 64) if (DBROWS_FUSED and not bt) else q.new_empty(0)   # (S90) the slab sum writes the head backward's dbias rows
        side = _side_mod() if KVQ_BRANCH else None
        s_ = side.fork(x.device) if side is not None else None
        if s_ is not None:  # (S90) the key-side and the query-side kernels are independent: two branches of the captured graph share the SMs
            with _on(side, s_):
                _, _, _, db = e.triattn_bwd(
                    q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, head_dim**-0.5, dbw, cut(0), cut(1), cut(2), btw, False, CORE_PDL, 2, dbr
                )
            e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, head_dim**-0.5, dbw, cut(0), cut(1), cut(2), btw, False, CORE_PDL, 1, q.new_empty(0))
            side.join(x.device, s_)
        else:
            _, _, _, db = e.triattn_bwd(
            q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, head_dim**-0.5, dbw, cut(0), cut(1), cut(2), btw, False,
            CORE_PDL, 0, dbr,
        )
    else:  # delta per 32-channel slice [B, L, S/32 x heads, L]: the core adds a head's slices where it reads them
        bwd = _core64().triattn_bwd64 if S == 64 else _core128().triattn_bwd128
        _, _, _, db = bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, head_dim**-0.5, cut(0), cut(1), cut(2))
    if not (S == D and DBROWS_FUSED and not bt):
        w.tri_db_rows(db, cut(4, 64), B, Lp, bt)
    dpair, xh = w.tri_whbwd(Dall, x, dy, lnw, stats, wall_t, Lp, trans, lr)
    # parameter gradients: G = D^T [xhat | 1] (one GEMM, K = B Lp^2: the pad rows of D are zero), then dW / dgamma / dbeta / dWb by the
    # d_pair 128 algebra (y = gamma o xhat + beta feeds every projection) and dWo without the padded channels, in tri_wfinish.
    # dWo = dy_s^T u over the caller's B L^2 rows (dy_s and u in the caller's layout and length)
    if TALLK and R <= TALLK_MAX_ROWS and C <= 256:
        G, dwo = _tallk().tallk(Dall, xh, dys, u)
    else:
        G = _mm32(Dall.t(), xh)
        dwo = _mm32(dys.t(), u)
    dt = dtypes
    fp32 = dt[2] == torch.float32
    dw, dwb, dwo, dgb = w.tri_wfinish(G, wcat, lnw, lnb, dwo, n_head, head_dim, fp32)
    # the biases: d bg = the column sums of dg (G's ones column, xh[:, C] = 1), d bo = the column sums of dy_s
    # (a module without these biases never reads it: split_grads returns None for both halves -- no zero-fill launch then)
    nb = n_head * head_dim + C
    dbias = (
        x.new_zeros(nb, dtype=torch.float32)
        if (gb is not None or ob is not None or out_mode == 2)
        else x.new_empty(nb, dtype=torch.float32)
    )
    if gb is not None:
        dbias[: n_head * head_dim] = G[3 * HDp : 4 * HDp, C].view(n_head, S)[:, :head_dim].reshape(-1)
    if ob is not None:
        dbias[n_head * head_dim :] = _mm32(dys.t(), xh[: dys.shape[0], C : C + 1]).view(-1)  # (xh's ones column: any B L^2 of its rows)
    dwo2 = x.new_empty(0)
    if out_mode == 2:  # dWo2 = dy_s^T u2 (the padded channels dropped), in Wo's dtype
        dwo2 = _mm32(dys.t(), u2).view(C, n_head, S)[:, :, :head_dim].reshape(C, n_head * head_dim).to(dwo.dtype)
    return dpair.view(B, L, L, C), dgb, dw, dwb, dwo, dbias, dwo2


class WideTrain(torch.autograd.Function):
    """The wide path with a backward: pair [B, L, L, C] -> pair + drop o TriangleAttention(pair) (starting node).
    packed = pack_wide_weights(...) of the same parameters (bf16 staging; the gradients come back in each parameter's dtype).
    (The module path goes through the custom ops of ``integrations.triattn_b200``, which call `wide_forward` / `wide_backward`.)"""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim, packed):
        need = any(ctx.needs_input_grad)
        out, saved = wide_forward(pair, lnw, lnb, packed, n_head, head_dim, mask, eps, drop, need)
        if need:
            ctx.save_for_backward(pair, lnw, lnb, drop if drop is not None else pair.new_empty(0), *saved)
            ctx.packed = packed
            ctx.dims = (n_head, head_dim, eps)
            ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, lnb, drop, *saved = ctx.saved_tensors
        n_head, head_dim, eps = ctx.dims
        raw = wide_backward(dy, pair, lnw, lnb, ctx.packed, n_head, head_dim, eps, drop if drop.numel() else None, saved, ctx.dtypes)[:6]
        return (*split_grads(False, raw, ctx.dtypes), None, None, None, None, None, None)


def wide_train(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, packed, n_head, head_dim, mask=None, eps=1e-5, drop=None):
    """pair + drop o TriangleAttention(pair) for the starting node with a backward (drop: [B, L, C] bf16 or None)."""
    return WideTrain.apply(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim, packed)
