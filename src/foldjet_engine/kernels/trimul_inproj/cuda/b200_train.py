"""B200 (sm_100a) TriMul training for every width D64-D512, either direction, bf16, B samples per launch (tokens b-major, M = B L L:
the kernels are token-flat but for the per-sample token mask and row-dropout scale; the contractions batch over channels x B).

D64 / D128 (``forward_small`` / ``backward_small``; L a multiple of 128): forward = the inference front (k1w,
cuBLAS contractions) and k3g's saving epilogue (x_n, LN_out statistics); backward = two fused kernels around the contraction
gradients:
  b1s / b1g   output side: gate / projection gradients, LN_out backward -> dG, dt; dWg, dWp, dg_o, db_o (front CTAs hand dP
              to LN-backward CTAs through an L2 ring). b1s: D64 (H = 64 / 128); b1g: D128 (H = 256 bidirectional / 128). Both fold
              LN_out's affine into the projection (no LayerNorm-forward pass; dWp / dg_o / db_o from A = dPr^T t on the tensor
              cores; one LayerNorm-backward pass with the per-token sums from the front epilogue: see b1g.cu, b1s.cu). At D64 the
              unified kernels replace the ring (b1u at H = 64, b1v at H = 128: one CTA per tile runs both halves, dPr in shared memory).
  cgb         contraction gradients -> plane gradient da [P, L, L], every product in one tcgen05 launch (L <= 256; cuBLAS above)
  b7m         D64 input side, one CTA per 128-token tile: pre-activations recomputed, gate gradients,
              dxn = [dpre | dG] [W1 ; Wg] (packed W1 resident in shared memory, dW1 in TMEM), LN_in backward + residual -> dx;
              dW1, dg_i, db_i
  b7g         D128 input side, over P / 64 plane chunks (8 bidirectional, 4 one direction) per source group (the packed
              W1, 128 / 256 KB, does not fit next to b7m's buffers)

D64 bidirectional at hidden 128 (8 D = 512 planes, an LN_out over 256 channels): the D64 / D128 forward (k1w, cuBLAS, k3g's saves);
the backward is the wide path's decomposition below on k3g's saved set (``_backward_hid_body``: q / g by cuBLAS, ``hid_gate_bwd``, lnout_bwd,
k1wb, ``hid_lnin_bwd``), since b1s / b7m do not fit 8 D planes.

D256 / 384 / 512:

Forward = the inference path (``b200_infer``: k1w, cuBLAS contractions, k3w) with k3w also storing p (output projection) and g
(gate logit), and the planes, the contraction output t and both LayerNorms' statistics kept for the backward.

Backward:
  gate_bwd    dp = dy ds sigmoid(g), dg = dp p (1 - sigmoid(g)) -> dpr = dp rs_o, dg; per token the LN_out backward's row sums
              S1 = dp . sp, S2 = dp . (p - ep) (fold identities: no pass over the H channels); per column r0, r1 for dWp
  cuBLAS      dor^T = Wp^T dpr^T = rs_o do^T              [H, M], channel-major like t
  lnout_bwd   dt = g_o dor - rs_o (S1 + xhat S2) / H, elementwise over [H, M]; dg_o, db_o
  cuBLAS      dWp = g_o (dpr^T t^T - r1) + b_o r0          (t read as stored: the LN_out fold again)
  cuBLAS      contraction gradients -> plane gradient da [P, L, L]
  k1wb        recomputes k1w's pre-activations and writes dpre = (dg_c, dp_c) token-major in the packed-row order
  cuBLAS      [dW1 ; dWg] = [dpre | dg]^T xn,  dxn = [dpre | dg] [W1 ; Wg]
  lnin_bwd    dx = dy + LN_in backward of dxn; dg_i, db_i
D64 / D128 weights: every kernel form of the parameters (bf16 row-major front matrices, the packed W1, k3g's permuted Wp, Wg, Wp)
comes from one ``trimul_forms`` launch, cached while the parameters are unchanged (``kernel_forms``: one launch per block per step,
shared by the no-grad recycles and the training forward). Parameter gradients: every kernel writes per-CTA partials (no atomics) and
``grad_reduce`` sums them in a fixed order straight into one fp32 buffer in parameter order, whose views are the gradients:
deterministic, and no unpack / copy kernels.

Sources: ``b200_sources/{k1w,k3g,b1s,b7m,b1g,b7g,wide_aux}.cu`` (D64 / D128), ``b200_sources/{k1w,k3w,wide_aux,k1wb,wide_bwd}.cu``.
Every call owns its buffers. The ``*_into`` ops are the same launches writing y (forward) / dx (backward) into a caller's buffer:
``trimul_train_chunked`` runs a batch in chunks of samples as one autograd node, each chunk writing its rows of y and dx in place.
"""

import functools
from pathlib import Path
from typing import NamedTuple

import torch

from foldjet_engine.kernels import _capture
from foldjet_engine.kernels._device import num_sms
from torch.autograd.function import once_differentiable

from foldjet_engine.kernels._compile import opaque

WIDTHS = (64, 128, 256, 384, 512)
EPS = 1e-5
TOK = 128
B1_RSF = 4  # b1s / b1g ring slots per front CTA
#: D64 one direction (contraction width 64): the unified B1 (``b1s.cu`` ``b1u_kernel``: each CTA runs the front epilogue AND the LN_out
#: backward of its tiles, dPr in shared memory; no L2 ring, no ring flags, no cooperative launch) instead of b1s's front / LN CTA split.
B1U = True
#: D64 at contraction width 128 (hidden 128 one direction, bidirectional): the unified B1 at H = 128 (``b1s.cu`` ``b1v_kernel``: A^T = tri dPr
#: on 128 TMEM lanes, dnr in a two-slot ring of 64-channel halves) instead of the ring b1s (2026-11-17, pass 2).
B1V = True
#: D128 one direction (contraction width 128): the unified B1 at C = 128 (``b1g.cu`` ``b1w_kernel``: one CTA per tile runs the front epilogue AND
#: the LN_out backward, dPr in shared memory, dnr in G's TMEM columns) instead of the ring b1g, up to this L (2026-11-17, part 3).
B1W = True
B1W_MAX_L = 256  # 128 -> 256 (2026-11-19, n1-trimul): D128 L256 one-direction step 0.1931 -> 0.1848 ms (graph median, 3 rounds); L384+ keeps the ring b1g (L384 +0.7 %)
#: the row-dropout scale of a D64 / D128 training call made by its forms' launch from the reference's draw (``forms_drop_op``) instead of a
#: separate ``drop_scale`` launch (2026-11-17, part 3).
DROP_IN_FORMS = True
#: the plane gradient of the D64 / D128 backward in one launch (``b7m.cu`` ``cgb_kernel``: every product of the contraction backward as
#: 128 x 128 tcgen05 tiles) up to this L; cuBLAS (one batched GEMM launch per product) above it (2026-11-17, pass 2).
CGB_MAX_L = 256
#: b7m with fewer tiles than SMs (L128 one sample): up to B7M_XG extra CTAs on the free SMs sum the B1 kernel's (output-side) parameter
#: gradient partials while the tiles run; grad_reduce then sums only b7m's partials (2026-11-17 part 6)
B7M_XR = True
B7M_XG = 32
#: the D64 / D128 training forward's contraction by ``ctr.cu`` (the inference route's small-L kernel, bitwise cuBLAS) in these
#: (L, bidirectional, contracted width) cells, where it measured faster in the training step (2026-11-17, round 3 training part 4;
#: graph, A / B in one process, three rounds): bidirectional L128 D64 72.8 -> 68.8 us, D128 141.0 -> 136.4; one direction L128 at 128
#: contracted channels (D128, D64 hidden 128) 95.3 -> 93.4 / 70.8 -> 69.3. Equal or slower: bidirectional D64 L256, one direction D64
#: L128 / L256, D64 hidden 128 L256, D128 L256, bidirectional D128 L256.
TRAIN_CTR = True
TRAIN_CTR_CELLS = frozenset({(128, True, 64), (128, True, 128), (128, False, 128)})
B7_RD = 12  # b7g ring slots per source group
B7_SG = 0  # b7g source groups (0: ``_b7_source_groups``' measured values)


#: Fixed-arity return types of the ``*_into`` ops (the saved sets of the wide / D64-D128 forwards, the ten parameter gradients).
_SAVED9 = tuple[(torch.Tensor,) * 9]
_SAVED6 = tuple[(torch.Tensor,) * 6]


def _ext():
    """The one B200 TriMul extension (shared with inference; see ``b200_sources/bind_b200.cpp``)."""
    from foldjet_engine.kernels.trimul_inproj.cuda.b200_infer import _ext as ext

    return ext()


@functools.lru_cache(maxsize=1)
def _k1wt():
    """The training front from the fp32 master weights (``b200_sources/k1wt.cu``, 2026-11-18 part 8): its own small extension."""
    from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    src = Path(__file__).with_name("b200_sources")
    ensure_cuda_home()
    return load_extension(
        name="trimul_b200_k1wt_v1",
        sources=[str(src / "k1wt.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--expt-relaxed-constexpr", "-DNDEBUG", f"-I{src}"],
        extra_cflags=["-std=c++17", f"-I{src}"],
        verbose=False,
    )


#: the D64 one-direction hidden-64 L128 one-sample training forward in ONE cluster launch (``b200_sources/tf_train_sm100.cu``, a copy
#: of the inference agent's ``tf_sm100.cu`` that also writes the backward's saved set; 2026-11-18 round 3 training part 11)
TRAIN_TF = True
#: extra nvcc defines of ``_tft_ext`` (experiments)
TFT_DEFS: tuple[str, ...] = ()


@functools.lru_cache(maxsize=1)
def _tft_ext():
    """``tf_train_sm100.cu``: its own small extension."""
    from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    src = Path(__file__).with_name("b200_sources")
    ensure_cuda_home()
    return load_extension(
        name="trimul_b200_tft_v1",
        sources=[str(src / "tf_train_sm100.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("100a"),
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            f"-I{src}",
            *TFT_DEFS,
        ],
        extra_cflags=["-std=c++17", f"-I{src}"],
        verbose=False,
    )


def _tft_ok(x, n, bsz, d, hid, direction, ws, mask, ds) -> bool:
    if not (TRAIN_TF and n == 128 and bsz == 1 and d == 64 and hid == 64 and direction in (1, 2)):
        return False
    if mask is not None and not (mask.numel() == n and mask.element_size() == 1 and mask.is_contiguous()):
        return False
    if not (ds.dtype == torch.bfloat16 and ds.numel() == n * d and ds.is_contiguous()):
        return False
    if not all(w.dtype == torch.bfloat16 and w.shape == (64, 64) and w.stride(1) == 1 for w in ws):
        return False
    return bool(_tft_ext().tft_supported())


def _philox_ext():
    """``_ext()`` connected to the B200 graph RNG writer (``kernels/_graph_rng_sm100``: the ONE fill_ override, shared with
    TriangleAttention) at the mode ``GRAPH_RNG_WRITER``: a draw captured in a graph registers its scalars there."""
    from foldjet_engine.kernels import _graph_rng_sm100 as rng

    e = _ext()
    if _CONNECTED.get(id(e)) is not e:
        rng.connect(e.trimul_graph_rng_api)
        _CONNECTED[id(e)] = e
    rng.set_mode(GRAPH_RNG_WRITER)
    return e


_CONNECTED: dict = {}


def supports(width: int, length: int, direction: int, hidden: int | None = None) -> bool:
    """direction 0 = bidirectional, 1 / 2 = one direction. k1w / k1wb / k3w tile 128 tokens and lnout_bwd 256, so L is a
    multiple of 16; the D64 / D128 path's k3g save and b1s / b1g group tiles by column (L a multiple of 128, at most 80 columns
    of tiles for the front CTAs). hidden: the contraction width (``b200_infer.hidden_ok``)."""
    if not hidden_ok(width, width if hidden is None else hidden, direction):
        return False
    if width <= 128:
        return width in WIDTHS and 0 < length <= 80 * TOK and length % TOK == 0
    return width in WIDTHS and length > 0 and length % 16 == 0


#: b1s / b1g front-CTA caps (see ``_front_ctas``); module values so a scan can set them
FRONT_CAP_D64 = 90
FRONT_CAP_D128_4 = 80
FRONT_CAP_D128_2 = 84


def _front_ctas(length: int, width: int, direction: int, planes: int | None = None) -> int:
    """b1s / b1g front CTAs: the largest multiple of L/128 <= 90 (D64, either plane count) / 80 (D128, 4 D planes) / 84 (D128,
    2 D planes), measured on L128-L768 (kernel alone, B200 2026-10-30; b1s at B4) -- per plane count, so a one-direction
    hidden-2 D block takes the bidirectional value (the kernels' per-token work is the same)."""
    rows = length // TOK
    planes = (4 if direction == 0 else 2) * width if planes is None else planes
    cap = {(64, True): FRONT_CAP_D64, (64, False): FRONT_CAP_D64, (128, True): FRONT_CAP_D128_4, (128, False): FRONT_CAP_D128_2}[width, planes == 4 * width]
    return (cap // rows) * rows


#: One-direction b7g source groups by length (S90 part 5, 2026-11-18, graph medians of L128-L1024 at 16-22 groups; the 18 of 2026-10-30 predates the
#: consumer-pair stream; L128-L256 flat, L384 -1.7 %, L512 -2.1 %, L768-L1024 -1.5 to -3 %). Bidirectional (8 chunks) stays at 12.
_B7_SG4 = {384: 21, 512: 20, 768: 22, 1024: 22}


def _b7_source_groups(length: int, nch: int) -> int:
    """b7g source groups of nch source CTAs each (the rest of the SMs consume, in pairs): 18 x 4 one direction; 12 x 8
    bidirectional, measured on L128-L768 (2026-10-30, after the hand-over / consumer-pair changes)."""
    if B7_SG:
        return B7_SG
    if nch == 4:
        return _B7_SG4.get(length, 18 if length < 384 else 22)
    return 12


def _pack_w1(wl, wlg, wr, wrg):
    """The packed W1 of b7m / b7g (chunk c = 64 gate rows, then 64 value rows, of the planes [left | right]); the kernels get it from
    ``trimul_forms`` -- this statement is the reference the tests hold it to."""
    p = wl.shape[0] + wr.shape[0]
    g = torch.cat((wlg, wrg))
    v = torch.cat((wl, wr))
    return torch.stack((g.reshape(p // 64, 64, -1), v.reshape(p // 64, 64, -1)), 1).reshape(2 * p, -1)


def _unpack_w1(d, p):
    c = d.reshape(p // 64, 2, 64, -1)
    g = c[:, 0].reshape(p, -1)
    v = c[:, 1].reshape(p, -1)
    h = p // 2
    return v[:h], g[:h], v[h:], g[h:]  # dWl, dWlg, dWr, dWrg


def _forms_op_fake(params):
    """The one bf16 buffer of forms_op: 8 R D + 3 D R + D D elements (see wide_aux.cu forms_kernel)."""
    r, d = params[0].shape
    return params[0].new_empty((8 * r * d + 3 * d * r + d * d,), dtype=torch.bfloat16)


@opaque(fake=_forms_op_fake, name="trimul_b200_forms")
def forms_op(params: list[torch.Tensor]) -> torch.Tensor:
    """wl, wlg, wr, wrg [R, D], wg [D, D], wp [D, R] (fp32 or bf16, any strides), go [R] fp32 -> one bf16 buffer of the kernels' weight
    forms (``wide_aux.cu`` ``forms_kernel``; the last, wpf = bf16(Wp g_o), from the stored Wp: one rounding)."""
    out = _forms_op_fake(params)
    with torch.cuda.device(out.device):
        _ext().trimul_forms(*params[:7], out)
    return out


def _forms_rf3_fake(params, mask, length):
    return [_forms_op_fake(params), params[0].new_empty((1,), dtype=torch.float32)]


@opaque(fake=_forms_rf3_fake, name="trimul_b200_forms_rf3")
def forms_rf3_op(params: list[torch.Tensor], mask: torch.Tensor | None, length: int) -> list[torch.Tensor]:
    """``forms_op`` for RoseTTAFold3: W_l (and its rows of the packed W1) times the sample's length factor f = bf16(1 / max(n, 1)), n the
    kept tokens of ``mask`` [L] (``length`` without one), as the reference's ``to_left.weight * f`` then cast -> [buffer, f (fp32 [1])].
    One launch for what the factor took ~7 (mask sum, casts, reciprocal, the weight product) plus an uncached forms launch (2026-11-17)."""
    out, f = _forms_rf3_fake(params, mask, length)
    with torch.cuda.device(out.device):
        _ext().trimul_forms(*params[:7], out, mask, int(length), f)
    return [out, f]


def _forms_drop_fake(params, r, p):
    return [_forms_op_fake(params), torch.empty_like(r)]


@opaque(fake=_forms_drop_fake, name="trimul_b200_forms_drop")
def forms_drop_op(params: list[torch.Tensor], r: torch.Tensor, p: float) -> list[torch.Tensor]:
    """``forms_op`` and the row-dropout scale of the reference's draw r (``dropout_draw``) in ONE launch -> [buffer, ds] (ds bitwise
    ``b200_infer.drop_scale_op(r, p)``; 2026-11-17 part 3: one launch fewer per training call)."""
    out, ds = _forms_drop_fake(params, r, p)
    with torch.cuda.device(out.device):
        _ext().trimul_forms(*params[:7], out, None, 0, None, r, ds, float(p))
    return [out, ds]


def _forms_rf3_drop_fake(params, mask, length, r, p):
    return [*_forms_rf3_fake(params, mask, length), torch.empty_like(r)]


@opaque(fake=_forms_rf3_drop_fake, name="trimul_b200_forms_rf3_drop")
def forms_rf3_drop_op(params: list[torch.Tensor], mask: torch.Tensor | None, length: int, r: torch.Tensor, p: float) -> list[torch.Tensor]:
    """``forms_rf3_op`` and the row-dropout scale of r in one launch -> [buffer, f, ds]."""
    out, f, ds = _forms_rf3_drop_fake(params, mask, length, r, p)
    with torch.cuda.device(out.device):
        _ext().trimul_forms(*params[:7], out, mask, int(length), f, r, ds, float(p))
    return [out, f, ds]


def _forms_philox_fake(params, rows, cols, p):
    return [_forms_op_fake(params), params[0].new_empty((rows, cols), dtype=torch.bfloat16)]


@opaque(fake=_forms_philox_fake, name="trimul_b200_forms_philox")
def forms_philox_op(params: list[torch.Tensor], rows: int, cols: int, p: float) -> list[torch.Tensor]:
    """``forms_drop_op`` with the draw made inside the same launch: the default generator's counter is reserved as
    ``torch.rand([rows * cols])`` reserves it and the kernel computes torch.rand's bits itself (``wide_aux.cu`` ``philox_rand_bf16``)
    -> [buffer, ds [rows, cols]], ds bitwise ``drop_scale_op(torch.rand(...), p)`` under the same generator state (part 6: the draw's
    own launch removed). Eager only (``dropout_draw`` hands traced code the torch draw)."""
    out, ds = _forms_philox_fake(params, rows, cols, p)
    with torch.cuda.device(out.device):
        _philox_ext().trimul_forms(*params[:7], out, None, 0, None, None, ds, float(p), True)
    return [out, ds]


def _forms_philox_gs_fake(params, rows, cols, p, blocks):
    return _forms_philox_fake(params, rows, cols, p)


@opaque(fake=_forms_philox_gs_fake, name="trimul_b200_forms_philox_gs")
def forms_philox_gs_op(params: list[torch.Tensor], rows: int, cols: int, p: float, blocks: int) -> list[torch.Tensor]:
    """``forms_philox_op`` on ``blocks`` 1024-thread blocks (``wide_aux.cu`` ``forms_gs_kernel``, the same values): beside ``k1wt``,
    which does not wait for it and takes the other SMs (2026-11-18 part 8). ``blocks`` < 0: on -blocks blocks, waiting for the work
    before it at entry and then triggering: BEFORE ``tft`` from the fp32 masters, which reads only ds of it (part 12)."""
    out, ds = _forms_philox_fake(params, rows, cols, p)
    with torch.cuda.device(out.device):
        _philox_ext().trimul_forms(*params[:7], out, None, 0, None, None, ds, float(p), True, abs(int(blocks)), 1 if int(blocks) < 0 else 0)
    return [out, ds]


def _forms_rf3_philox_fake(params, mask, length, rows, cols, p):
    return [*_forms_rf3_fake(params, mask, length), params[0].new_empty((rows, cols), dtype=torch.bfloat16)]


@opaque(fake=_forms_rf3_philox_fake, name="trimul_b200_forms_rf3_philox")
def forms_rf3_philox_op(
    params: list[torch.Tensor], mask: torch.Tensor | None, length: int, rows: int, cols: int, p: float
) -> list[torch.Tensor]:
    """``forms_rf3_op`` and the row-dropout scale of the draw made in the same launch (``forms_philox_op``) -> [buffer, f, ds]."""
    out, f, ds = _forms_rf3_philox_fake(params, mask, length, rows, cols, p)
    with torch.cuda.device(out.device):
        _philox_ext().trimul_forms(*params[:7], out, mask, int(length), f, None, ds, float(p), True)
    return [out, f, ds]


#: the training front from the fp32 master weights BEFORE the forms launch, which runs beside it (``k1wt`` + ``forms_philox_gs_op``;
#: 2026-11-18 part 8): bitwise, measured level (D64 L128 0.0555 -> 0.0553 ms, hidden 128 0.0676 -> 0.0676): k1wt takes 6.9 us (k1w
#: 6.3: the fp32 loads and the conversion), and the forms launch, which must complete after k1wt, puts ~0.9 us before the contraction.
#: Off (an experiment, ``K1WT = True`` to run it).
K1WT = False
#: k1wt only when at least this many SMs are left for the forms launch beside it
K1WT_MIN_FREE = 8
#: forms_gs_kernel blocks (1024 threads) per SM left free by k1wt
K1WT_GS_PER_SM = 1
#: (2026-11-18 part 12) k1wt AFTER a small wait-first forms launch (``TFTM_GS_BLOCKS`` blocks) that it waits for only at its exit
#: (``after_forms``): the forms launch beside the front, not before it, where ``tft`` does not serve the cell (hidden 128). Bitwise;
#: measured level (hidden 128 L128 0.0680-0.0682 -> 0.0679-0.0680 ms; 8 forms blocks: 0.0723, the forms launch gates k1wt's exit): off.
K1WT_AF = False

#: the draw inside the forms launch (``forms_philox_op``) instead of a torch.rand launch before it (2026-11-17 part 6)
PHILOX_IN_FORMS = True
#: a graph holding that draw: its RNG prologue (the generator scalars' two aten fills) served by one programmatic launch
#: (``kernels/_graph_rng_sm100.cu``, shared with TriangleAttention since part 7) instead of two ordinary ones (2026-11-17 part 6)
GRAPH_RNG_WRITER = 2  # 0: ATen's fills, 1: one ordinary launch, 2: one programmatic launch


class DeferredDraw(NamedTuple):
    """A torch.rand(shape, dtype, device) not drawn yet: the forms launch draws it (``forms_philox_op``), or ``realize_draw`` does
    where the forms come from the cache. Nothing between ``dropout_draw`` and that point uses the generator, so the bits and the
    generator's state afterwards are the reference's."""

    shape: tuple
    dtype: torch.dtype
    device: torch.device


def realize_draw(r):
    """The draw as a tensor (``DeferredDraw`` drawn now with torch.rand)."""
    return torch.rand(r.shape, device=r.device, dtype=r.dtype) if isinstance(r, DeferredDraw) else r


def draw_rows(r, rows: int, cols: int):
    """``r.view(rows, cols)`` for a drawn or a deferred draw."""
    return DeferredDraw((rows, cols), r.dtype, r.device) if isinstance(r, DeferredDraw) else r.view(rows, cols)


def dropout_draw(pair: torch.Tensor, p: float):
    """The reference's uniform draw of ``modules.functional.dropout_scale(pair, p, 1)`` (``torch.rand([B, 1, L, D])`` in the pair's dtype:
    the same generator state and bits) for the training kernels to turn into the scale inside their forms launch (``forms_drop_op``);
    None where ``b200_infer.dropout_scale`` would not use the one-kernel scale (it draws and scales with the torch statements then).
    Eager with ``PHILOX_IN_FORMS``: a ``DeferredDraw`` (drawn by the forms launch itself)."""
    if pair.dtype != torch.bfloat16 or (pair.shape[0] * pair.shape[2] * pair.shape[3]) % 8:
        return None
    shape = list(pair.shape)
    shape[1] = 1
    if PHILOX_IN_FORMS and pair.is_cuda and not torch.compiler.is_compiling():
        return DeferredDraw(tuple(shape), pair.dtype, pair.device)
    return torch.rand(shape, device=pair.device, dtype=pair.dtype)


def _unpack_forms(buf, params, vec):
    wl = params[0]
    r, d = wl.shape
    rd = r * d
    o = 8 * rd
    return (
        *buf[: 4 * rd].view(4, r, d).unbind(0),
        buf[o + d * r : o + d * r + d * d].view(d, d),
        buf[o + d * r + d * d : o + 2 * d * r + d * d].view(d, r),
        *vec,
        buf[4 * rd : o].view(4 * r, d),
        buf[o : o + d * r].view(d, r),
        buf[o + 2 * d * r + d * d :].view(d, r),
    )


def kernel_forms(
    params,
    *,
    eval_reads_eager: bool = False,
    rf3: tuple | None = None,
    drop: tuple | None = None,
    gs_blocks: int = 0,
):
    """[wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, w1, wpp, wpf]: the D64 / D128 kernels' operands from the module's ten parameters (fp32
    master or bf16, the front matrices in any layout) -- the four front matrices row-major bf16, Wg / Wp bf16, the LayerNorm vectors fp32,
    the packed W1 of b7m / b7g, k3g's permuted Wp and b1s / b1g's LN_out-folded Wp' = bf16(Wp g_o) -- from one ``trimul_forms`` launch,
    cached while the parameters are unchanged (``kernels._capture.param_forms``: inside a CUDA-graph capture once per capture, so a
    training step's no-grad recycles and its training forward share one launch per block). ``eval_reads_eager`` (an eval-mode no-grad
    call): a capture is served the eager entry its warm-up made (``capture_reads_eager``, the H100 inference forms' contract), so an
    inference graph replays no ``trimul_forms`` launch (2026-11-17). ``rf3`` = (token mask [L] or None, L): RoseTTAFold3's length factor
    folded into W_l in the same launch (``forms_rf3_op``; per call, not cached: it depends on the mask), f appended as a 14th entry.
    ``drop`` = (r, p): the row-dropout scale of the draw r (``dropout_draw``) appended last, made by the forms' launch when it runs
    (``forms_drop_op`` / ``forms_rf3_drop_op``), else by ``drop_scale_op`` (the forms came from the cache) -- 2026-11-17 part 3."""
    params = tuple(params)

    def vecs():
        return [v.detach().float().contiguous() for v in params[6:10]]

    deferred = drop is not None and isinstance(drop[0], DeferredDraw)
    if rf3 is not None:
        vec = vecs()
        if deferred:
            rows, cols = drop[0].shape
            buf, f, ds = forms_rf3_philox_op([*params[:6], vec[2]], rf3[0], int(rf3[1]), int(rows), int(cols), float(drop[1]))
            return [*_unpack_forms(buf, params, vec), f, ds]
        if drop is not None:
            buf, f, ds = forms_rf3_drop_op([*params[:6], vec[2]], rf3[0], int(rf3[1]), drop[0], float(drop[1]))
            return [*_unpack_forms(buf, params, vec), f, ds]
        buf, f = forms_rf3_op([*params[:6], vec[2]], rf3[0], int(rf3[1]))
        return [*_unpack_forms(buf, params, vec), f]

    made = []

    def build():
        vec = vecs()
        if drop is None:
            return _unpack_forms(forms_op([*params[:6], vec[2]]), params, vec)
        if deferred:
            rows, cols = drop[0].shape
            if gs_blocks:
                buf, ds = forms_philox_gs_op([*params[:6], vec[2]], int(rows), int(cols), float(drop[1]), int(gs_blocks))
            else:
                buf, ds = forms_philox_op([*params[:6], vec[2]], int(rows), int(cols), float(drop[1]))
        else:
            buf, ds = forms_drop_op([*params[:6], vec[2]], drop[0], float(drop[1]))
        made.append(ds)
        return _unpack_forms(buf, params, vec)

    if torch.compiler.is_compiling():
        out = list(build())  # traced: graph ops (the cache's capture-id query is a graph break)
    else:
        out = list(_capture.param_forms("trimul_b200_forms", params, build, capture_reads_eager=eval_reads_eager))
    if drop is not None:
        out.append(made[0] if made else _infer.drop_scale_op(realize_draw(drop[0]), float(drop[1])))
    return out


def _grad_total(leaves):
    """Elements of ``grad_reduce``'s output: the ten parameter gradients in parameter order."""
    return sum(int(t.numel()) for t in leaves[1:11])


def _split_grads(buf, leaves):
    """Views of ``grad_reduce``'s buffer shaped like the parameters (wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo)."""
    out, o = [], 0
    for t in leaves[1:11]:
        k = int(t.numel())
        out.append(buf[o : o + k].view(t.shape))
        o += k
    return out


#: int32 ring flags of one D64 / D128 backward (b1s / b1g, then b7g), zeroed by the training forward's k3g (its zero_buf): a fixed
#: size, the most any shape needs (front CTAs <= 112 x B1_RSF, b7g source groups <= 18 x B7_RD x 8 chunks), so that the saved set's
#: shape does not depend on L (torch.compile with dynamic shapes).
FLAG_WORDS = 112 * B1_RSF + 18 * B7_RD * 8


def _flag_words(n, d, direction, p):
    """The flags one backward uses (<= FLAG_WORDS)."""
    nf = _front_ctas(n, d, direction, p)
    nch = p // 64
    return nf * B1_RSF + (_b7_source_groups(n, nch) * B7_RD * nch if d == 128 else 0)


from foldjet_engine.kernels.trimul_inproj.cuda.b200_infer import _bmm, _contract, _fold, contracted, hidden_of, hidden_ok, planes_of
from foldjet_engine.kernels.trimul_inproj.cuda import b200_infer as _infer


def _contract_bwd(planes, dt, d, direction):
    """Outgoing t = a b^T: da = dt b, db = dt^T a; incoming t = a^T b: da = b dt^T, db = a dt. Planes [P, (B,) n, n]; the samples fold
    into cuBLAS's batch dim as in ``_contract`` (whose note on the ``out=`` buffer applies here too)."""
    n = planes.shape[-1]
    samples = planes.numel() // (planes.shape[0] * n * n)
    dpl = torch.empty((planes.shape[0] * samples, n, n), dtype=planes.dtype, device=planes.device)
    f = _fold
    h, h2, h3 = d * samples, 2 * d * samples, 3 * d * samples
    if direction == 0:
        a_o, a_i, b_o, b_i = f(planes[:d]), f(planes[d : 2 * d]), f(planes[2 * d : 3 * d]), f(planes[3 * d :])
        dt_o, dt_i = f(dt[:d]), f(dt[d:])
        _bmm(dt_o, b_o, out=dpl[:h])
        _bmm(dt_o.transpose(1, 2), a_o, out=dpl[h2:h3])
        _bmm(b_i, dt_i.transpose(1, 2), out=dpl[h:h2])
        _bmm(a_i, dt_i, out=dpl[h3:])
    elif direction == 1:
        _bmm(f(dt), f(planes[d:]), out=dpl[:h])
        _bmm(f(dt).transpose(1, 2), f(planes[:d]), out=dpl[h:])
    else:
        _bmm(f(planes[d:]), f(dt).transpose(1, 2), out=dpl[:h])
        _bmm(f(planes[:d]), f(dt), out=dpl[h:])
    return dpl.view(planes.shape)


def _sample_major(width: int, hidden: int, direction: int) -> bool:
    """D64 training keeps its channel-major activations (planes, t, dt, the plane gradient) sample-major: ``[P / 64, B, 64, L, L]``,
    each 64-channel chunk (one contraction operand) holding the B samples' planes one after another, instead of ``[P, B, L, L]``.
    k1w's plane stores then step one sample's L^2 tokens between channels whatever B is; with the channel stride B L^2, k1w ran at
    half its one-sample rate at B4 L768 (580 vs 4 x 72 us). The contractions fold (sample, channel) of a chunk into cuBLAS's batch
    exactly as before (the same per-sample products), and k3g / b1s / b7m read the same tiles through 4-D maps
    (``tmap::make_chmajor``). Every chunk must be one operand: the contracted width is 64."""
    return width == 64 and contracted(width, hidden, direction) == 64


def _contract_sm(planes, direction):
    """``_contract`` for sample-major planes [P / 64, B, 64, n, n] -> t [H / 64, B, 64, n, n] (chunk g of planes is operand g). As in
    ``_contract`` the ``out=`` buffers are slices of one 3-D tensor and the 5-D result is a view of it."""
    n, k = planes.shape[-1], planes.shape[1] * 64
    f = lambda a: a.view(-1, n, n)
    g = 2 if direction == 0 else 1
    t = planes.new_empty((g * k, n, n))
    if direction == 0:
        _bmm(f(planes[0]), f(planes[2]).transpose(1, 2), out=t[:k])
        _bmm(f(planes[1]).transpose(1, 2), f(planes[3]), out=t[k:])
    elif direction == 1:
        _bmm(f(planes[0]), f(planes[1]).transpose(1, 2), out=t)
    else:
        _bmm(f(planes[0]).transpose(1, 2), f(planes[1]), out=t)
    return t.view(g, *planes.shape[1:])


def _contract_bwd_sm(planes, dt, direction):
    """``_contract_bwd`` for sample-major planes / dt; returns the plane gradient in the planes' layout."""
    n, k = planes.shape[-1], planes.shape[1] * 64
    f = lambda a: a.view(-1, n, n)
    dpl = planes.new_empty((planes.shape[0] * k, n, n))
    if direction == 0:
        _bmm(f(dt[0]), f(planes[2]), out=dpl[:k])
        _bmm(f(planes[3]), f(dt[1]).transpose(1, 2), out=dpl[k : 2 * k])
        _bmm(f(dt[0]).transpose(1, 2), f(planes[0]), out=dpl[2 * k : 3 * k])
        _bmm(f(planes[1]), f(dt[1]), out=dpl[3 * k :])
    elif direction == 1:
        _bmm(f(dt[0]), f(planes[1]), out=dpl[:k])
        _bmm(f(dt[0]).transpose(1, 2), f(planes[0]), out=dpl[k:])
    else:
        _bmm(f(planes[1]), f(dt[0]).transpose(1, 2), out=dpl[:k])
        _bmm(f(planes[0]), f(dt[0]), out=dpl[k:])
    return dpl.view(planes.shape)


def _forward_fake(leaves, mask, ds, direction):
    """y and the saved set: planes, t, mean / rstd (LN_in), mean / rstd (LN_out), k3w vectors, p, g."""
    return [torch.empty_like(leaves[0]), *_forward_into_fake(leaves, mask, ds, direction, leaves[0])]


def _forward_into_fake(leaves, mask, ds, direction, y):
    """The saved set of ``forward`` (y is written into ``y``)."""
    x = leaves[0]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m, p = bsz * n * n, (4 if direction == 0 else 2) * d
    f32 = torch.float32
    return (
        x.new_empty((p, bsz, n, n)),
        x.new_empty((p // 2, bsz, n, n)),
        x.new_empty((m,), dtype=f32),
        x.new_empty((m,), dtype=f32),
        x.new_empty((m,), dtype=f32),
        x.new_empty((m,), dtype=f32),
        x.new_empty((4, d), dtype=f32),
        x.new_empty((m, d)),
        x.new_empty((m, d)),
    )


@opaque(fake=_forward_fake, name="trimul_b200_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int) -> list[torch.Tensor]:
    """leaves = x, wl, wlg, wr, wrg, wg, wp (bf16, [out, in]), gi, bi, go, bo (fp32); x [B, L, L, D] (samples b-major); mask [B * L]
    bool token mask or None; ds [B * L, D] bf16 row-dropout scale or None; direction 0 = bidirectional, 1 = outgoing, 2 = incoming."""
    return list(_forward_body(leaves, mask, ds, direction, None))


@opaque(fake=_forward_into_fake, name="trimul_b200_train_fwd_into", mutates_args=("y",))
def forward_into(
    leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int, y: torch.Tensor
) -> _SAVED9:
    """``forward`` writing y into the caller's ``y``; returns the saved set (a fixed-arity tuple: a mutating custom op returning
    ``Tensor[]`` cannot be functionalized)."""
    return tuple(_forward_body(leaves, mask, ds, direction, y)[1:])


def _forward_body(leaves, mask, ds, direction, y):
    """-> [y, *saved]; y None: allocated here (where the one-launch path always allocated it)."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves[:11]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        mean = x.new_empty((m,), dtype=torch.float32)
        rstd = torch.empty_like(mean)
        if d > _infer.K1W_INSTAT:  # else k1w computes and writes them
            E.k1w_stats(x2, mean, rstd, EPS)
        planes = x.new_empty(((4 if direction == 0 else 2) * d, bsz, n, n))
        E.k1w_forward(x2, wl, wlg, wr, wrg, mask, mean, rstd, gi, bi, planes, 0, EPS)
        t = _contract(planes, d, direction)
        mo = torch.empty_like(mean)
        ro = torch.empty_like(mean)
        ost = K3W_TRAIN_OST and n >= K3W_TRAIN_OST_MINL
        if not ost:
            E.wide_ln_stats(t, mo, ro, EPS)
        wpq = torch.empty_like(wp, dtype=torch.bfloat16)
        wgq = torch.empty_like(wg, dtype=torch.bfloat16)
        vec = x.new_empty((4, d), dtype=torch.float32)
        E.wide_fold_prep(wp, go, bo, wg, gi, bi, wpq, wgq, vec)
        if y is None:
            y = torch.empty_like(x)
        pv = torch.empty_like(x2)
        gv = torch.empty_like(x2)
        E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mo, ro, mean, rstd, ds, y.view(m, d), n, 0, pv, gv, (2 | 8) if ost else 0)
    return [y, planes, t, mean, rstd, mo, ro, vec, pv, gv]


#: S90: the training k3w computes the LN_out statistics in-kernel (the inference OST warps) and saves them: no wide_ln_stats launch
K3W_TRAIN_OST = True
#: ... from this length of a side (L256 measured slower: k3w 61.3 us in-kernel against 52.0 + the 6.5 us ln_stats launch)
K3W_TRAIN_OST_MINL = 384
#: S90: the wide backward's three partial-table column sums in ONE launch (``wide_colsum3``) with this many row splits per column tile
COLSUM3_SPLIT = 2


SIDE_BWD = True
_SIDE: dict = {}


def _side_stream(device):
    """One side stream per device for the wide backward's independent GEMMs."""
    key = (device.type, device.index)
    if key not in _SIDE:
        _SIDE[key] = torch.cuda.Stream(device)
    return _SIDE[key]


def _grads_fake(leaves):
    """The ten parameter gradients as ONE fp32 buffer in parameter order (``_split_grads``; the D64 / D128 ``grad_reduce`` layout)."""
    return leaves[0].new_empty((_grad_total(leaves),), dtype=torch.float32)


def _backward_fake(leaves, mask, ds, direction, saved, dy):
    """dx in x's dtype and the parameter-gradient buffer (fp32)."""
    return [torch.empty_like(leaves[0]), _grads_fake(leaves)]


def _backward_into_fake(leaves, mask, ds, direction, saved, dy, dx):
    """The parameter-gradient buffer (dx is written into ``dx``)."""
    return _grads_fake(leaves)


# Renamed (_g1) 2026-11-17 when the gradients became one buffer (wide_grad_pack) and the leaves gained the packed [W1; Wg] form: new
# custom-op names give new FX-graph-cache keys (see the note above the D64 / D128 ops).
@opaque(fake=_backward_fake, name="trimul_b200_train_bwd_g1")
def backward(
    leaves: list[torch.Tensor],
    mask: torch.Tensor | None,
    ds: torch.Tensor | None,
    direction: int,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
) -> list[torch.Tensor]:
    """gate_bwd -> do GEMM -> lnout_bwd -> dWp -> contraction gradients -> k1wb -> weight / input GEMMs -> lnin_bwd."""
    return _backward_body(leaves, mask, ds, direction, saved, dy, None)


@opaque(fake=_backward_into_fake, name="trimul_b200_train_bwd_into_g1", mutates_args=("dx",))
def backward_into(
    leaves: list[torch.Tensor],
    mask: torch.Tensor | None,
    ds: torch.Tensor | None,
    direction: int,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    dx: torch.Tensor,
) -> torch.Tensor:
    """``backward`` writing dx into the caller's ``dx``; returns the parameter-gradient buffer."""
    return _backward_body(leaves, mask, ds, direction, saved, dy, dx)[1]


def _backward_body(leaves, mask, ds, direction, saved, dy, dx):
    """-> [dx, *parameter gradients]; dx None: allocated here."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves[:11]
    w1g = leaves[11] if len(leaves) > 11 else torch.cat((_pack_w1(wl, wlg, wr, wrg), wg))  # [4R + D, D]: wide_forms' packed [W1; Wg]
    planes, t, mean, rstd, mo, ro, vec, pv, gv = saved
    n, d = x.shape[1], x.shape[-1]
    m = x.shape[0] * n * n
    p = planes.shape[0]
    h = p // 2
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        dy2 = dy.contiguous().view(m, d)
        dpr = torch.empty_like(x2)  # dp rs_o (dp itself is never needed)
        dcat = x.new_empty((m, 2 * p + d))  # [dpre | dg]
        # adjacent: wide_gate_bwd zeroes S1, S2 and colsum3's 256 arrival counters with one memset; the rest of s12 is colsum3's scratch
        s12 = mean.new_empty((2 * m + 256 + COLSUM3_SPLIT * 32 * ((4 * d + 2 * h) // 32),))
        s1, s2, cnt = s12[:m], s12[m : 2 * m], s12[2 * m :]
        acc = x.new_empty((4 * d + 2 * h,), dtype=torch.float32)  # r0, r1, dg_i, db_i, dg_o, db_o
        c3 = COLSUM3_SPLIT > 0
        pg = E.wide_gate_bwd(dy2, pv, gv, ds, vec, mo, ro, dpr, dcat[:, 2 * p :], s1, s2, acc[: 2 * d], n, c3, cnt if c3 else None)
        cur = torch.cuda.current_stream()
        side = _side_stream(x.device) if SIDE_BWD and not torch.compiler.is_compiling() else None
        if side is not None:  # S90: the weight-gradient GEMMs and ln_apply run beside the input-gradient chain (own stream, joined before the gradient pack)
            ev_g = torch.cuda.Event()
            ev_g.record(cur)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                xn = torch.empty_like(x2)
                E.wide_ln_apply(x2, mean, rstd, gi, bi, xn)
                side.wait_event(ev_g)
                dwp_mm = torch.mm(dpr.t(), t.view(h, m).t(), out_dtype=torch.float32)
        dt = torch.empty_like(t)
        # (dpr Wp)^T = rs_o do^T, channel-major like t and dt, so lnout_bwd transposes nothing
        po = E.wide_lnout_bwd(torch.mm(wp.t(), dpr.t()), t, mo, ro, s1, s2, go, dt, acc[4 * d :], c3)
        if side is None:
            dwp_mm = torch.mm(dpr.t(), t.view(h, m).t(), out_dtype=torch.float32)  # dWp = g_o (dwp_mm - r1) + r0 b_o in wide_grad_pack
        dpl = _contract_bwd(planes, dt, d, direction)
        del dt
        wl_, wlg_, wr_, wrg_ = (w if w.stride(1) == 1 else w.contiguous() for w in (wl, wlg, wr, wrg))
        E.k1wb_forward(x2, wl_, wlg_, wr_, wrg_, mask, mean, rstd, gi, bi, dpl, dcat, 0)
        del dpl
        if side is not None:
            ev_k = torch.cuda.Event()
            ev_k.record(cur)
            with torch.cuda.stream(side):
                side.wait_event(ev_k)
                dw = torch.mm(dcat.t(), xn, out_dtype=torch.float32)  # [2P + D, D]
        else:
            xn = torch.empty_like(x2)
            E.wide_ln_apply(x2, mean, rstd, gi, bi, xn)
            dw = torch.mm(dcat.t(), xn, out_dtype=torch.float32)  # [2P + D, D]
        dxn = dcat @ w1g
        if dx is None:
            dx = torch.empty_like(x)
        pi = E.wide_lnin_bwd(dxn, x2, dy2, mean, rstd, gi, dx.view(m, d), acc[2 * d : 4 * d], c3)
        if side is not None:
            cur.wait_stream(side)
        if c3:
            E.wide_colsum3(pg, po, pi, acc[: 2 * d], acc[4 * d :], acc[2 * d : 4 * d], cnt, COLSUM3_SPLIT)  # S90: the three column sums, one launch
        # the ten gradients in parameter order, one fp32 buffer (fp32: the parameters may be an fp32 master; the autograd function
        # splits it and hands each back in its parameter's dtype). One launch (2026-11-17; was ten copies + four elementwise launches).
        grads = acc.new_empty((_grad_total(leaves),))
        E.wide_grad_pack(dw, dwp_mm, acc, go, bo, grads)
    return [dx, grads]


# The D64 / D128 ops carry an _sm suffix since the D64 saved set became sample-major (5-D planes / t): inductor's FX graph cache
# keys a graph on its code and input metadata, not on a custom op's fake output, so a graph compiled when these ops returned 4-D
# planes would be reused under the old names (and fail its size asserts). New names give new keys. Renamed again (_smf) when the leaves became the
# kernel forms and the saved set / gradient outputs changed (cached weight forms, flags buffer, one gradient buffer).
def _forward_small_fake(leaves, mask, ds, direction):
    """y and the saved set: planes, t, x_n, LN_out mean / rstd."""
    return [torch.empty_like(leaves[0]), *_forward_small_into_fake(leaves, mask, ds, direction, leaves[0])]


def _forward_small_into_fake(leaves, mask, ds, direction, y):
    """The saved set of ``forward_small`` (y is written into ``y``): planes, t, x_n, LN_out mean / rstd, the backward's zeroed flags."""
    x = leaves[0]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    hid = hidden_of(leaves[1].shape[0], direction)
    m, p = bsz * n * n, planes_of(d, hid, direction)
    f32 = torch.float32
    if _sample_major(d, hid, direction):
        pl, tt = x.new_empty((p // 64, bsz, 64, n, n)), x.new_empty((p // 128, bsz, 64, n, n))
    else:
        pl, tt = x.new_empty((p, bsz, n, n)), x.new_empty((p // 2, bsz, n, n))
    return (pl, tt, x.new_empty((m, d)), x.new_empty((m,), dtype=f32), x.new_empty((m,), dtype=f32), x.new_empty((FLAG_WORDS,), dtype=f32))


@opaque(fake=_forward_small_fake, name="trimul_b200_train_small_fwd_smf")
def forward_small(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int) -> list[torch.Tensor]:
    """D64 / D128 training forward; arguments as ``forward`` except ds, which is required ([B * L, D] bf16, ones without dropout), and
    leaves, which are x and ``kernel_forms`` (the front matrices row-major, then w1 and wpp). One direction, the hidden width (wl's
    rows) may be 2 D (``b200_infer.hidden_ok``)."""
    return list(_forward_small_body(leaves, mask, ds, direction, None))


def _front_master_fake(x, wmaster, mask, gi, bi, direction, after_forms=False):
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    hid = hidden_of(wmaster[0].shape[0], direction)
    p = planes_of(d, hid, direction)
    return x.new_empty((p // 64, bsz, 64, n, n)) if _sample_major(d, hid, direction) else x.new_empty((p, bsz, n, n))


@opaque(fake=_front_master_fake, name="trimul_b200_train_front_master")
def front_master_op(
    x: torch.Tensor,
    wmaster: list[torch.Tensor],
    mask: torch.Tensor | None,
    gi: torch.Tensor,
    bi: torch.Tensor,
    direction: int,
    after_forms: bool = False,
) -> torch.Tensor:
    """The training front (k1w's planes, the same bits) by ``k1wt`` (k1wt.cu) from ``wmaster`` = the fp32 master W_l, W_lg, W_r, W_rg,
    BEFORE the forms launch (2026-11-18 part 8; ``k1wt_plan``): the forms launch after it runs beside it (``forms_philox_gs_op``).
    ``after_forms`` (part 12): launched right AFTER the wait-first forms launch, waiting for it only at its exit."""
    planes = _front_master_fake(x, wmaster, mask, gi, bi, direction)
    with torch.cuda.device(x.device):
        _k1wt().k1wt_forward(x.reshape(-1, x.shape[-1]), *wmaster, mask, gi, bi, planes, num_sms(), EPS, bool(after_forms))
    return planes


def _forward_small_master_fake(leaves, mask, ds, direction, planes):
    y, _pl, *rest = _forward_small_fake(leaves, mask, ds, direction)
    return [y, *rest]


@opaque(fake=_forward_small_master_fake, name="trimul_b200_train_small_fwd_master")
def forward_small_master(
    leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int, planes: torch.Tensor
) -> list[torch.Tensor]:
    """``forward_small`` with the front's planes already made (``front_master_op``) -> [y, *saved without the planes] (an output may not
    alias an input)."""
    y, _pl, *rest = _forward_small_body(leaves, mask, ds, direction, None, planes)
    return [y, *rest]


#: (2026-11-18 part 12) the one-launch training forward (``tft``) from the fp32 MASTER weights, launched right after a small wait-first
#: forms launch (``forms_philox_gs_op`` on ``TFTM_GS_BLOCKS`` 1024-thread blocks: the draw, ds and the backward's forms) that it does not
#: wait for until its epilogue (ds): the forms launch leaves the forward's critical path. Bitwise; measured level (D64 L128 out / in
#: 0.0512-0.0520 / 0.0533-0.0540 ms either way): tft starts ~0.5 us earlier, but the 96 KB of fp32 masters (cold after the L2 flush)
#: reach the front ~1 us later than the forms launch's L2-hot bf16 copies (tft_train_sm100.cu TFT_FM 1 / 2 / 3 all so): off.
TFT_MASTER = False
TFTM_GS_BLOCKS = 8


def tftm_plan(leaves, direction, mask, drop, rf3) -> bool:
    """True when ``_TrainingSmall`` runs ``forward_tft_master``: a deferred draw (the forms launch makes ds), no RoseTTAFold3 factor,
    the ``tft`` cell (D64 one direction hidden 64, L128, one sample) and the ten parameters fp32 ([64, 64] matrices with unit column
    stride and 16-byte rows, contiguous vectors)."""
    if not (TFT_MASTER and TRAIN_TF and drop is not None and isinstance(drop[0], DeferredDraw) and not rf3 and direction in (1, 2)):
        return False
    if torch.compiler.is_compiling():
        return False
    x = leaves[0]
    if not (
        x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 128 and x.shape[2] == 128 and x.shape[3] == 64 and x.dtype == torch.bfloat16
    ):
        return False
    w, v = leaves[1:7], leaves[7:11]
    if not all(
        t.dtype == torch.float32 and t.shape == (64, 64) and t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0 for t in w
    ):
        return False
    if not all(t.dtype == torch.float32 and t.numel() == 64 and t.is_contiguous() for t in v):
        return False
    if mask is not None and not (mask.numel() == 128 and mask.element_size() == 1 and mask.is_contiguous()):
        return False
    return bool(_tft_ext().tft_supported())


def _forward_tft_master_fake(leaves, masters, mask, ds, direction):
    return _forward_small_fake(leaves, mask, ds, direction)


@opaque(fake=_forward_tft_master_fake, name="trimul_b200_train_tftm")
def forward_tft_master(
    leaves: list[torch.Tensor], masters: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int
) -> list[torch.Tensor]:
    """``forward_small``'s one-launch path (``tft``) with the weights and LayerNorm vectors read from ``masters`` (the module's ten fp32
    parameters, ``tftm_plan``) instead of the forms in ``leaves`` -> [y, *saved] (the same bits)."""
    x = leaves[0]
    n, d = x.shape[1], x.shape[-1]
    m = n * n
    with torch.cuda.device(x.device):
        planes = x.new_empty((2, 1, 64, n, n))
        t = x.new_empty((1, 1, 64, n, n))
        y = torch.empty_like(x)
        x2 = x.reshape(m, d)
        xn = torch.empty_like(x2)
        mo = x.new_empty((m,), dtype=torch.float32)
        ro = torch.empty_like(mo)
        zb = x.new_empty((FLAG_WORDS,), dtype=torch.float32)
        blocked = x.new_empty((128, n, n))
        mw = [t_.detach() for t_ in masters]
        _tft_ext().tft_forward_master(x2, *mw[:6], mask, *mw[6:10], ds, blocked, planes, t, xn, mo, ro, zb, y.view(m, d), direction, EPS)
    return [y, planes, t, xn, mo, ro, zb]


#: (2026-11-18 part 12) the chain: ``tft`` from the masters as the step's FIRST kernel (it waits for the work before it at entry and
#: triggers at once), the forms launch after it as its programmatic dependent on ``TFTC_GS_BLOCKS`` blocks beside it (ds first, then the
#: backward's forms), which triggers the backward when every tft CTA is at its epilogue (counters shared by the two launches).
#: Bitwise; measured slower (D64 L128 0.0513 / 0.0534 -> 0.0573 / 0.0574 ms: tft 17-18 us with the master conversion, the forms launch
#: starts 2.7 us after tft, b1u starts ~2.5 us later through the counter hand-off): off (with TFT_MASTER).
TFT_CHAIN = False
TFTC_GS_BLOCKS = 8
_TFTC_CNT: dict = {}


def _tftc_counters(device):
    """The chain's two monotonic int32 counters on ``device`` (made once, outside any capture; None inside a capture before then)."""
    c = _TFTC_CNT.get(device)
    if c is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        c = _TFTC_CNT[device] = torch.zeros(4, dtype=torch.int32, device=device)
    return c


def _forward_tft_chain_fake(x, masters, mask, direction, rows, cols, p, nb):
    y, *saved = _forward_small_fake([x, *masters[:6]], mask, None, direction)
    buf, ds = _forms_philox_fake(masters, rows, cols, p)
    return [y, *saved, buf, ds]


@opaque(fake=_forward_tft_chain_fake, name="trimul_b200_train_tftc")
def forward_tft_chain(
    x: torch.Tensor, masters: list[torch.Tensor], mask: torch.Tensor | None, direction: int, rows: int, cols: int, p: float, nb: int
) -> list[torch.Tensor]:
    """The chain (``TFT_CHAIN``): ``tft`` from the ten fp32 ``masters`` as the first launch, then the forms launch drawing the row-dropout
    scale (``forms_philox_gs_op``'s values, mode 2) -> [y, *saved, forms buffer, ds]; the same bits as ``forward_tft_master`` after the
    wait-first forms launch."""
    n, d = x.shape[1], x.shape[-1]
    m = n * n
    with torch.cuda.device(x.device):
        cnt = _tftc_counters(x.device)
        fx = _philox_ext()  # connected to the graph RNG writer before tft launches: nothing may fail between the two launches
        # the forms kernel's module loaded before tft runs (CUDA lazy loading: a first launch that loads it waits for the running tft,
        # whose epilogue waits for it -- the first chain step deadlocked so, measured with bounded spins)
        fx.trimul_forms_gs_preload()
        planes = x.new_empty((2, 1, 64, n, n))
        t = x.new_empty((1, 1, 64, n, n))
        y = torch.empty_like(x)
        x2 = x.reshape(m, d)
        xn = torch.empty_like(x2)
        mo = x.new_empty((m,), dtype=torch.float32)
        ro = torch.empty_like(mo)
        zb = x.new_empty((FLAG_WORDS,), dtype=torch.float32)
        blocked = x.new_empty((128, n, n))
        buf, ds = _forms_philox_fake(masters, rows, cols, p)
        mw = [t_.detach() for t_ in masters]
        _tft_ext().tft_forward_chain(
            x2, *mw[:6], mask, *mw[6:10], ds, blocked, planes, t, xn, mo, ro, zb, y.view(m, d), direction, EPS, cnt, int(nb)
        )
        fx.trimul_forms(*mw[:6], mw[8], buf, None, 0, None, None, ds, float(p), True, int(nb), 2, cnt)
    return [y, planes, t, xn, mo, ro, zb, buf, ds]


@opaque(fake=_forward_small_into_fake, name="trimul_b200_train_small_fwd_into_smf", mutates_args=("y",))
def forward_small_into(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int, y: torch.Tensor) -> _SAVED6:
    """``forward_small`` writing y into the caller's ``y``; returns the saved set."""
    return tuple(_forward_small_body(leaves, mask, ds, direction, y)[1:])


def _forward_small_body(leaves, mask, ds, direction, y, front=None):
    """-> [y, *saved]; y None: allocated here. front: the planes made already (``front_master_op``)."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, _w1, wpp = leaves[:13]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        hid = hidden_of(wl.shape[0], direction)
        p = planes_of(d, hid, direction)
        sm = _sample_major(d, hid, direction)
        if front is None and _tft_ok(x, n, bsz, d, hid, direction, (wl, wlg, wr, wrg, wg, wp), mask, ds):
            # the whole forward in one cluster launch (tf_train_sm100.cu): the same saved set, the backward unchanged
            planes = x.new_empty((p // 64, bsz, 64, n, n))
            t = x.new_empty((1, bsz, 64, n, n))
            if y is None:
                y = torch.empty_like(x)
            xn = torch.empty_like(x2)
            mo = x.new_empty((m,), dtype=torch.float32)
            ro = torch.empty_like(mo)
            zb = x.new_empty((FLAG_WORDS,), dtype=torch.float32)
            blocked = x.new_empty((p, n, n))  # the in-kernel contraction's blocked planes (scratch)
            _tft_ext().tft_forward(
                x2, wl, wlg, wr, wrg, wg, wp, mask, gi, bi, go, bo, ds, blocked, planes, t, xn, mo, ro, zb, y.view(m, d), direction, EPS
            )
            return [y, planes, t, xn, mo, ro, zb]
        if front is not None:
            planes = front
        else:
            planes = x.new_empty((p // 64, bsz, 64, n, n) if sm else (p, bsz, n, n))
            E.k1w_forward(x2, wl, wlg, wr, wrg, mask, None, None, gi, bi, planes, 0, EPS)
        c = contracted(d, hid, direction)
        if TRAIN_CTR and (n, direction == 0, c) in TRAIN_CTR_CELLS and (not sm or bsz == 1):
            # ctr.cu (bitwise cuBLAS) as a programmatic dependent of k1w that triggers k3g at entry; one sample's sample-major planes
            # [P / 64, 1, 64, n, n] are the channel-major [P, 1, n, n] bytes
            t = _infer._contract_inf(E, planes.view(p, bsz, n, n) if sm else planes, c, direction)
            if sm:
                t = t.view(-1, bsz, 64, n, n)
        else:
            t = _contract_sm(planes, direction) if sm else _contract(planes, c, direction)
        if y is None:
            y = torch.empty_like(x)
        xn = torch.empty_like(x2)
        mo = x.new_empty((m,), dtype=torch.float32)
        ro = torch.empty_like(mo)
        zb = x.new_empty((FLAG_WORDS,), dtype=torch.float32)  # the backward's ring flags, zeroed by k3g
        E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 1, ds, xn, mo, ro, 0, zb)
    return [y, planes, t, xn, mo, ro, zb]


def _backward_small_fake(leaves, mask, ds, direction, saved, dy, wl_scale):
    """dx in x's dtype and the ten parameter gradients as one fp32 buffer in parameter order (``_split_grads``)."""
    return [torch.empty_like(leaves[0]), _backward_small_into_fake(leaves, mask, ds, direction, saved, dy, None, wl_scale)]


def _backward_small_into_fake(leaves, mask, ds, direction, saved, dy, dx, wl_scale):
    """The parameter gradients as one fp32 buffer (dx is written into ``dx``)."""
    return leaves[0].new_empty((_grad_total(leaves),), dtype=torch.float32)


# _smf2 (2026-11-17): wl_scale added (RoseTTAFold3's length factor: dW_l = f dW_l', in grad_reduce); new names, new FX-cache keys
@opaque(fake=_backward_small_fake, name="trimul_b200_train_small_bwd_smf2")
def backward_small(
    leaves: list[torch.Tensor],
    mask: torch.Tensor | None,
    ds: torch.Tensor,
    direction: int,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    wl_scale: torch.Tensor | None,
) -> list[torch.Tensor]:
    """b1s / b1g -> contraction gradients -> b7m / b7g -> grad_reduce. Every kernel writes its parameter gradients as per-CTA partials
    (no atomics); ``grad_reduce`` sums them in a fixed order into one fp32 buffer in parameter order -> [dx, buffer]: deterministic. The
    ring flags come zeroed from the forward (saved[5]). ``wl_scale`` (fp32 [1] or None): the forms' RoseTTAFold3 factor (``forms_rf3_op``),
    W_l's gradient is scaled by it."""
    return _backward_small_body(leaves, mask, ds, direction, saved, dy, None, wl_scale)


@opaque(fake=_backward_small_into_fake, name="trimul_b200_train_small_bwd_into_smf2", mutates_args=("dx",))
def backward_small_into(
    leaves: list[torch.Tensor],
    mask: torch.Tensor | None,
    ds: torch.Tensor,
    direction: int,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    dx: torch.Tensor,
    wl_scale: torch.Tensor | None,
) -> torch.Tensor:
    """``backward_small`` writing dx into the caller's ``dx``; returns the parameter-gradient buffer (one tensor: a mutating custom op
    returning ``Tensor[]`` cannot be functionalized)."""
    return _backward_small_body(leaves, mask, ds, direction, saved, dy, dx, wl_scale)[1]


def _eight_planes(width: int, rows: int, direction: int) -> bool:
    """The D64 bidirectional hidden-128 block (``b200_infer.hidden_ok``): 8 D = 512 planes, an LN_out over 256 channels. Its forward is
    ``forward_small``'s (k1w, cuBLAS, k3g); its backward is ``_backward_hid_body`` (b1s's double-buffered front and LN tiles do not fit
    at H = 256, b7m's packed W1 and dW1 do not fit at P = 512)."""
    return direction == 0 and rows == 4 * width


def _backward_hid_body(leaves, mask, ds, direction, saved, dy, dx):
    """``_backward_small_body`` for the 8-plane D64 block: the wide path's decomposition (cuBLAS for the GEMMs, hand CUDA between them).
      fold_prep      Wp' = bf16(Wp g_o), sp = rowsum(Wp'), ep = Wp b_o
      cuBLAS         q = t^T Wp'^T, g = xn Wg^T (fp32; k3g saved neither p nor g)
      hid_gate_bwd   p = rs_o (q - mu_o sp) + ep; dp, dg (into [dpre | dg]), dpr = dp rs_o; S1, S2; r0, r1
      cuBLAS         dor^T = Wp^T dpr^T;  lnout_bwd -> dt, dg_o, db_o;  dWp = g_o (dpr^T t^T - r1) + b_o r0
      cuBLAS         contraction gradients -> plane gradient da
      k1wb           da -> dpre (pre-activations recomputed from x)
      cuBLAS         [dW1 ; dWg] = [dpre | dg]^T xn,  dxn = [dpre | dg] [W1 ; Wg]
      hid_lnin_bwd   dx = dy + LN_in backward of dxn; dg_i, db_i
    -> [dx, the ten parameter gradients as one fp32 buffer in parameter order]."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, w1, _wpp = leaves[:13]
    planes, t, xn, mo, ro, _zb = saved
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    p = planes.shape[0]
    h = p // 2  # LN_out channels
    f32 = torch.float32
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        dy2 = dy.contiguous().view(m, d)
        t2 = t.view(h, m)
        # (2026-11-17: Wp' = trimul_forms' wpf, rounded once from the stored Wp, measured neither better nor worse here over four input
        # seeds at L384 / L640 -- unlike b1s / b1g -- so this path keeps wide_fold_prep on the bf16 forms)
        wpq = torch.empty_like(wp, dtype=torch.bfloat16)
        wgq = torch.empty_like(wg, dtype=torch.bfloat16)
        vec = x.new_empty((4, d), dtype=f32)
        E.wide_fold_prep(wp, go, bo, wg, gi, bi, wpq, wgq, vec)
        q = torch.mm(t2.t(), wpq.t(), out_dtype=f32)
        g = torch.mm(xn, wg.t(), out_dtype=f32)
        dpr = torch.empty_like(x2)  # dp rs_o
        dcat = x.new_empty((m, 2 * p + d))  # [dpre | dg]
        s1 = x.new_empty((m,), dtype=f32)
        s2 = torch.empty_like(s1)
        acc = x.new_empty((4 * d + 2 * h,), dtype=f32)  # r0, r1, dg_i, db_i, dg_o, db_o
        r0, r1 = acc[:d], acc[d : 2 * d]
        E.hid_gate_bwd(dy2, q, g, ds, vec, mo, ro, dpr, dcat[:, 2 * p :], s1, s2, acc[: 2 * d], n)
        del q, g
        dt = torch.empty_like(t)
        E.wide_lnout_bwd(torch.mm(wp.t(), dpr.t()), t2, mo, ro, s1, s2, go, dt.view(h, m), acc[4 * d :])
        dwp = go[None, :] * (torch.mm(dpr.t(), t2.t(), out_dtype=f32) - r1[:, None]) + r0[:, None] * bo[None, :]
        dpl = _contract_bwd(planes, dt, p // 4, direction)
        del dt
        E.k1wb_forward(x2, wl, wlg, wr, wrg, mask, mo, ro, gi, bi, dpl, dcat, 0)  # D <= 128: k1wb forms its own LN_in statistics
        del dpl
        dw = torch.mm(dcat.t(), xn, out_dtype=f32)  # [2P + D, D] = [packed dW1 ; dWg]
        dxn = torch.mm(dcat, torch.cat((w1, wg)))
        del dcat
        if dx is None:
            dx = torch.empty_like(x)
        E.hid_lnin_bwd(dxn, x2, dy2, gi, dx.view(m, d), acc[2 * d : 4 * d], EPS)
        grads = torch.cat([v.reshape(-1) for v in (*_unpack_w1(dw[: 2 * p], p), dw[2 * p :], dwp, acc[2 * d :])])
    return [dx, grads]


def _backward_small_body(leaves, mask, ds, direction, saved, dy, dx, wl_scale=None):
    """-> [dx, parameter-gradient buffer]; dx None: allocated here."""
    if _eight_planes(leaves[0].shape[-1], leaves[1].shape[0], direction):
        assert wl_scale is None, "RoseTTAFold3 is one direction"
        return _backward_hid_body(leaves, mask, ds, direction, saved, dy, dx)
    x, wl, wlg, wr, wrg, wg, wp, gi, _bi, go, bo, w1, _wpp, wpf = leaves
    planes, t, xn, mo, ro, zb = saved
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    sm = planes.dim() == 5  # sample-major (``_sample_major``)
    p = planes.shape[0] * (64 if sm else 1)
    h = p // 2
    # the unified B1 (no ring): b1s_backward (D64: b1u, b1v) / b1g_backward (D128: b1w) with nf = 0
    uni = (B1U and d == 64 and (h == 64 or (B1V and h == 128))) or (B1W and d == 128 and h == 128 and n <= B1W_MAX_L)
    nf = 0 if uni else _front_ctas(n, d, direction, p)
    nch = p // 64
    sg = _b7_source_groups(n, nch)
    nsm = num_sms(x.device)
    f32 = torch.float32
    E = _ext()
    assert _flag_words(n, d, direction, p) <= FLAG_WORDS
    with torch.cuda.device(x.device):
        flags = zb.view(torch.int32)
        x2 = x.reshape(m, d)
        dy2 = dy.contiguous().view(m, d)
        dg = torch.empty_like(x2)
        dt = torch.empty_like(t)
        ring = x.new_empty((nf * B1_RSF * (TOK + 1024 // (2 * d)), d))  # per slot: a dPr tile, then a 1 KB (S1 | S2) block
        # unified: one row per CTA (dWg | dWp, and dg_o | db_o) over min(tiles, SMs) CTAs; else front / LN CTAs
        rows_o = min(m // TOK, nsm) if uni else max(nf, nsm - nf)
        nl_o = rows_o if uni else nsm - nf
        part_o = x.new_empty((rows_o, d * (d + h)), dtype=f32)  # row r: front CTA r's dWg | LN CTA r's dWp
        lnp_o = x.new_empty((nl_o, 2 * h), dtype=f32)  # LN CTAs' (dg_o | db_o) rows
        (E.b1g_backward if d == 128 else E.b1s_backward)(
            dy2, xn, t, ds, mo, ro, wg, wp, wpf, go, bo, dg, dt, part_o, lnp_o, ring, flags[: nf * B1_RSF], n, nf
        )
        del ring
        if n <= CGB_MAX_L:
            dpl = torch.empty_like(planes)
            E.cgb_backward(planes, dt, dpl, n, direction)
        else:
            dpl = (
                _contract_bwd_sm(planes, dt, direction)
                if sm
                else _contract_bwd(planes, dt, contracted(d, hidden_of(wl.shape[0], direction), direction), direction)
            )
        del dt
        if dx is None:
            dx = torch.empty_like(x)
        grads = x.new_empty((_grad_total(leaves),), dtype=f32)
        xg = 0
        if d == 128:
            ns, nc = sg, nsm - nch * sg
            part_i = x.new_empty((ns, 2 * p * d), dtype=f32)  # source groups' dW1 partials
            lnp_i = x.new_empty((nc, 2 * d), dtype=f32)
            ring7 = x.new_empty((sg * B7_RD * nch * TOK, 128))
            E.b7g_backward(
                x2, xn, dy2, dpl.view(p, m), dg, mask, w1, wg, gi, dx.view(m, d), part_i, lnp_i, ring7, flags[nf * B1_RSF :], EPS, sg, n
            )
        else:
            ns = nc = min(m // TOK, nsm)
            part_i = x.new_empty((ns, 2 * p * d), dtype=f32)
            lnp_i = x.new_empty((nc, 2 * d), dtype=f32)
            # b7m tiles leave SMs free (one tile per CTA): extra CTAs there sum the output-side partials (B7M_XR)
            xg = min(nsm - ns, B7M_XG) if B7M_XR and ns < nsm else 0
            E.b7m_backward(
                x2,
                xn,
                dy2,
                dpl if sm else dpl.view(p, m),
                dg,
                mask,
                w1,
                wg,
                gi,
                dx.view(m, d),
                part_i,
                lnp_i,
                EPS,
                n,
                part_o,
                rows_o,
                lnp_o,
                nl_o,
                grads,
                xg,
            )
        E.grad_reduce(part_o, rows_o, lnp_o, nl_o, part_i, ns, lnp_i, nc, grads, wl_scale, xg > 0)
    return [dx, grads]


def _wide_forms_fake(params):
    r, d = params[0].shape
    return params[0].new_empty((8 * r * d + d * d + d * r,), dtype=torch.bfloat16)


@opaque(fake=_wide_forms_fake, name="trimul_b200_wide_forms")
def wide_forms_op(params: list[torch.Tensor]) -> torch.Tensor:
    """wl, wlg, wr, wrg [R, D], wg [D, D], wp [D, R] (fp32 or bf16, any strides) -> one bf16 buffer (``wide_aux.cu`` ``wide_forms``):
    the six casts and the packed [W1; Wg] operand of the input-gradient GEMM, one launch."""
    out = _wide_forms_fake(params)
    with torch.cuda.device(out.device):
        _ext().wide_forms(*params[:6], out)
    return out


def _wide_forms_philox_fake(params, rows, cols, p):
    return [_wide_forms_fake(params), params[0].new_empty((rows, cols), dtype=torch.bfloat16)]


@opaque(fake=_wide_forms_philox_fake, name="trimul_b200_wide_forms_philox")
def wide_forms_philox_op(params: list[torch.Tensor], rows: int, cols: int, p: float) -> list[torch.Tensor]:
    """``wide_forms_op`` and the row-dropout scale of torch.rand's draw made in the same launch (S90: ``forms_philox_op``'s draw for the
    D256-512 training call) -> [buffer, ds [rows, cols]], ds bitwise ``drop_scale_op(torch.rand(...), p)`` under the same generator state.
    Eager / graph capture (the graph RNG writer registers the captured scalars); ``dropout_draw`` hands traced code the torch draw."""
    out, ds = _wide_forms_philox_fake(params, rows, cols, p)
    with torch.cuda.device(out.device):
        _philox_ext().wide_forms(*params[:6], out, ds, float(p))
    return [out, ds]


def _kernel_leaves(leaves, drop=None):
    """The kernels' operands: x, the six projection weights bf16 row-major, the four LayerNorm vectors fp32, and the packed
    [W1; Wg] [4R + D, D] of the input-gradient GEMM (a 12th leaf, read by ``_backward_body``).
    The casts happen here, outside autograd, so the parameters can be an fp32 master and still receive unrounded fp32 gradients.
    (2026-11-17: folding Wp / Wg from the fp32 masters instead, as the D64 / D128 path now does, measured no better over four input
    seeds at D256 L384 / D512 L640 -- the D256+ gradients sum the folded-weight rounding over 4-8x more channels -- so the
    training fold stays on these casts.) The casts are one ``wide_forms`` launch (2026-11-17; round to nearest even, bitwise torch's
    ``.to(bf16)``; was six cast launches, plus four cat / two copy launches for [W1; Wg] in every backward)."""
    x = leaves[0]
    assert x.dtype == torch.bfloat16, "B200 wide TriMul training: bf16 activations"
    wl = leaves[1]
    r, d = wl.shape
    rd = r * d
    ds = None
    if drop is not None and isinstance(drop[0], DeferredDraw):   # S90: the draw and its scale in the forms launch
        rows, cols = drop[0].shape
        buf, ds = wide_forms_philox_op(list(leaves[1:7]), int(rows), int(cols), float(drop[1]))
    else:
        buf = wide_forms_op(list(leaves[1:7]))
        if drop is not None:
            ds = _infer.drop_scale_op(drop[0], float(drop[1]))
    o = 8 * rd
    out = [
        x,
        *buf[: 4 * rd].view(4, r, d).unbind(0),
        buf[o : o + d * d].view(d, d),
        buf[o + d * d :].view(d, r),
        *(v.float().contiguous() for v in leaves[7:11]),
        buf[4 * rd : o + d * d].view(4 * r + d, d),
    ]
    return out if drop is None else (out, ds)


def k1wt_plan(leaves, direction):
    """(fp32 master W_l, W_lg, W_r, W_rg, forms blocks) for ``forward_small_master`` (``k1wt``) and the forms launch beside it
    (``forms_philox_gs_op`` on the SMs k1wt leaves free), when the cell is D64 one direction with fp32 front masters and every weight
    K-block of a k1wt CTA is resident at one tile per CTA (L128); else None. The front runs first (``front_master_op``), then the forms
    launch, if this call makes one, beside it (it does not wait for k1wt at entry; ``_TrainingSmall``)."""
    if not ((K1WT or K1WT_AF) and direction != 0) or torch.compiler.is_compiling():
        return None
    x, w = leaves[0], list(leaves[1:5])
    d = x.shape[-1]
    if d != 64 or not all(t.dtype == torch.float32 and t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0 for t in w):
        return None
    if not all(t.dtype == torch.float32 for t in leaves[5:11]):  # the forms' vectors: no cast launch between k1wt and the forms launch
        return None
    p = planes_of(d, hidden_of(w[0].shape[0], direction), direction)
    m, nsm = x.shape[0] * x.shape[1] * x.shape[2], num_sms()
    tiles = m // 128
    if tiles > nsm or nsm - tiles < K1WT_MIN_FREE or not _k1wt().k1wt_fits(d, p, m, nsm):
        return None
    return [t.detach() for t in w], (nsm - tiles) * K1WT_GS_PER_SM


class _TrainingSmall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direction, mask, ds, rf3, drop, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves[1:]]
        tftm = tftm_plan(leaves, direction, mask, drop, rf3)
        masters = list(leaves[1:11])
        if tftm and TFT_CHAIN and _tftc_counters(leaves[0].device) is not None:
            rows, cols = drop[0].shape
            y, *saved, buf, ds = forward_tft_chain(
                leaves[0], masters, mask, direction, int(rows), int(cols), float(drop[1]), TFTC_GS_BLOCKS
            )
            vec = [v.detach().float().contiguous() for v in masters[6:10]]
            leaves = [leaves[0], *_unpack_forms(buf, masters, vec)]
            ctx.direction = direction
            ctx.save_for_backward(mask, ds, None, *leaves, *saved)
            return y
        plan = None
        # k1wt after the forms: the hidden 128 cells (the hidden 64 L128 cell is tft's, from the masters or the forms)
        af_cell = K1WT_AF and leaves[1].shape[0] != leaves[0].shape[-1]
        if not tftm and drop is not None and isinstance(drop[0], DeferredDraw) and not rf3 and (K1WT or af_cell):
            plan = k1wt_plan(leaves, direction)
        af = bool(plan) and af_cell
        if plan and not af:  # the front first (k1wt), the forms launch beside it
            front = front_master_op(leaves[0], plan[0], mask, leaves[7].detach(), leaves[8].detach(), direction)
        gs = -TFTM_GS_BLOCKS if tftm else (-plan[1] if af else (plan[1] if plan else 0))
        forms = kernel_forms(leaves[1:], rf3=(mask, rf3) if rf3 else None, drop=drop, gs_blocks=gs)
        if af:  # (part 12) the forms launch first (wait-first, a few blocks), then k1wt beside it
            front = front_master_op(leaves[0], plan[0], mask, leaves[7].detach(), leaves[8].detach(), direction, True)
        if drop is not None:
            ds = forms.pop()  # the row-dropout scale, made by the forms' launch
        wl_scale = forms.pop() if rf3 else None  # RoseTTAFold3: the length factor the forms folded into W_l
        leaves = [leaves[0], *forms]
        if tftm:  # tft from the masters, a programmatic dependent of the (wait-first) forms launch that it waits for at its epilogue
            y, *saved = forward_tft_master(leaves, masters, mask, ds, direction)
        elif plan:
            y, *rest = forward_small_master(leaves, mask, ds, direction, front)
            saved = [front, *rest]
        else:
            y, *saved = forward_small(leaves, mask, ds, direction)
        ctx.direction = direction
        ctx.save_for_backward(mask, ds, wl_scale, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        mask, ds, wl_scale, leaves, saved = vals[0], vals[1], vals[2], list(vals[3:17]), list(vals[17:])
        dx, grads = backward_small(leaves, mask, ds, ctx.direction, saved, dy, wl_scale)
        return (None, None, None, None, None, dx, *(g.to(dt) for g, dt in zip(_split_grads(grads, leaves), ctx.param_dtypes, strict=True)))


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direction, mask, ds, drop, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves[1:]]
        if drop is None:
            leaves = _kernel_leaves(leaves)
        else:  # S90: the row-dropout scale made by the forms launch (a drawn r: drop_scale_op) in place of ds
            leaves, ds = _kernel_leaves(leaves, drop)
        y, *saved = forward(leaves, mask, ds, direction)
        ctx.direction = direction
        ctx.save_for_backward(mask, ds, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        mask, ds, leaves, saved = vals[0], vals[1], list(vals[2:14]), list(vals[14:])
        dx, grads = backward(leaves, mask, ds, ctx.direction, saved, dy)
        return (None, None, None, None, dx, *(g.to(dt) for g, dt in zip(_split_grads(grads, leaves), ctx.param_dtypes, strict=True)))


def trimul_train(leaves, mask, ds, direction, rf3: int = 0, drop: tuple | None = None):
    """Training step's forward with autograd (x + ds * trimul(x)); see ``forward`` for the arguments. The parameter leaves may be
    in any float dtype (bf16, or an fp32 master): the kernels run bf16 and each parameter gets its gradient in its own dtype, fp32
    ones unrounded. ``rf3`` = L (D <= 128, one sample): RoseTTAFold3's length factor of ``mask`` folded into the forms of W_l
    (``forms_rf3_op``) and into its gradient; 0 = none. ``drop`` = (r [B * L, D], p) at D <= 128 in place of ds: the reference's draw
    (``dropout_draw``), turned into the scale by the forms' launch (``kernel_forms``)."""
    x = leaves[0]
    d = x.shape[-1]
    if d <= 128:
        if ds is None and drop is None:
            ds = x.new_ones((x.shape[0] * x.shape[1], d))
        return _TrainingSmall.apply(direction, mask, ds, int(rf3), drop, *leaves)
    assert not rf3, "RoseTTAFold3 length factor in the forms: D64 / D128 only"
    return _Training.apply(direction, mask, ds, drop, *leaves)


def _chunk_ops(width):
    return (forward_small_into, backward_small_into) if width <= 128 else (forward_into, backward_into)


class _Chunked(torch.autograd.Function):
    """A batch in chunks of samples (``sizes``, b-major, in order) as ONE autograd node. y and dx are allocated once and each chunk's
    launch writes its rows of them in place (``*_into``), so nothing is copied or concatenated. Plain chunks keep their saved tensors
    from the forward to the backward; checkpointed chunks (``checkpoint``) save nothing and run their forward again in the backward,
    so only the chunk being differentiated holds saved tensors. The parameter gradients are the chunks' fp32 gradients summed in chunk
    order (deterministic). Each chunk runs exactly the launches of a one-launch call of its samples."""

    @staticmethod
    def forward(ctx, direction, sizes, checkpoint, mask, ds, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves[1:]]
        small = leaves[0].shape[-1] <= 128
        leaves = [leaves[0], *kernel_forms(leaves[1:])] if small else _kernel_leaves(leaves)
        ctx.nleaves = len(leaves)
        x = leaves[0]
        n = x.shape[1]
        fwd, _ = _chunk_ops(x.shape[-1])
        y = torch.empty_like(x)
        saved, counts, start = [], [], 0
        for size in sizes:
            stop = start + size
            part = fwd(
                [x[start:stop], *leaves[1:]],
                None if mask is None else mask[start * n : stop * n],
                None if ds is None else ds[start * n : stop * n],
                direction,
                y[start:stop],
            )
            if checkpoint:
                del part
                counts.append(0)
            else:
                saved.extend(part)
                counts.append(len(part))
            start = stop
        ctx.direction, ctx.sizes, ctx.checkpoint, ctx.counts = direction, sizes, checkpoint, counts
        ctx.save_for_backward(mask, ds, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        nl = 2 + ctx.nleaves
        mask, ds, leaves, saved = vals[0], vals[1], list(vals[2:nl]), list(vals[nl:])
        x = leaves[0]
        n = x.shape[1]
        small = x.shape[-1] <= 128
        fwd, bwd = _chunk_ops(x.shape[-1])
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        total = None
        start, at = 0, 0
        for size, count in zip(ctx.sizes, ctx.counts, strict=True):
            stop = start + size
            sub = [x[start:stop], *leaves[1:]]
            m = None if mask is None else mask[start * n : stop * n]
            d = None if ds is None else ds[start * n : stop * n]
            if ctx.checkpoint:
                part = fwd(sub, m, d, ctx.direction, torch.empty_like(sub[0]))
            else:
                part = saved[at : at + count]
                at += count
            grads = bwd(sub, m, d, ctx.direction, part, dy[start:stop], dx[start:stop], *([None] if small else []))
            del part
            total = grads if total is None else total.add_(grads)  # one fp32 buffer either way (grad_reduce / wide_grad_pack)
            start = stop
        total = _split_grads(total, leaves)
        return (None, None, None, None, None, dx, *(g.to(dt) for g, dt in zip(total, ctx.param_dtypes, strict=True)))


def trimul_train_chunked(leaves, mask, ds, direction, sizes, checkpoint):
    """``trimul_train`` for a batch run as chunks of ``sizes`` samples (``integrations.trimul_b200._plan``), one autograd node. mask
    [B * L] or None, ds [B * L, D] or None."""
    x = leaves[0]
    d = x.shape[-1]
    if d <= 128 and ds is None:
        ds = x.new_ones((x.shape[0] * x.shape[1], d))
    return _Chunked.apply(direction, tuple(sizes), bool(checkpoint), mask, ds, *leaves)
