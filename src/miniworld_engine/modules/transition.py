"""Transitions: ``Transition`` (LayerNorm + SwiGLU expand / squeeze + residual), ``ConditionedTransition`` (AlphaFold 3 Algorithm 25:
AdaLN + SwiGLU + sigmoid conditioning gate + residual) and Boltz-2's ``UpGatedConditionedTransition`` (a third, up-gate projection).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from miniworld_engine.modules.functional import swish_gate
from miniworld_engine.modules.norm import (
    AdaptiveLayerNorm,
    LayerNorm,
    _activation_dtype_reason,
    _as_rows3,
    _wants_grad,
    a100,
    b200,
    h100_unsupported,
    sm120,
    triton_unsupported,
)
from miniworld_engine.modules.primitives import BackendModule, Linear, MasterWeights, master_linear, random_input, stream_shape

#: The OpenDDE kit's transition composition (``fpf_transition_odde``), a release directory for ``modules/_anthropic.load``.
_ODDE_ARMT = "opendde/opt/forward/fast_inference/levers/ARMT/third_party"


class Transition(BackendModule, MasterWeights):
    """Transition layer with SwiGLU activation::

        h = LN(x)                                  # ln_in
        y = x + W_s (silu(W_a h) * (W_b h))        # expand_a, expand_b, squeeze (no biases)

    Parameters
    ----------
    d_hidden : int
        Dimension of the input and output features.
    n : int
        Expansion factor.
    master_weights : bool
        Keep fp32 master parameters under a bf16 cast (:class:`MasterWeights`).
    backend : str
        See :class:`BackendModule`. ``"anthropic"``: the shared core's transition provider (``opt_core.kernels.transition``, the af3_torch
        kit's lever ``transition``) by its ``fast`` tier word -- LayerNorm + SwiGLU + output projection in its fused rows (the residual
        added after, as the reference does), bfloat16 / float32 cells, inference only, where the provider's measured row for the cell is one of its kernels.
        ``"cuda"`` (H100): the engine's sm_90a Transition kernels (``kernels/transition/cuda.transition_sm90a``: LayerNorm + SwiGLU expand +
        squeeze with the residual in its epilogue, forward and backward): ``fused_sm90a`` (d 128, n 2 / 4), ``fused_wide_sm90a`` (d 64 /
        256 / 384 / 512 with n 2 / 4, d 768 with n 2; whole 128-row tiles) and ``fused_generic_sm90a`` (d a multiple of 64 up to 1024,
        hidden a multiple of 128, any row count; fp32 activations on TF32 tensor cores, so only where torch allows TF32); fp32 with TF32 off (IEEE products) runs the engine's IEEE fp32 chain (``kernels/ieee_f32``: FFMA GEMMs with fused SwiGLU / residual epilogues and LayerNorm row kernels, forward and backward, d a multiple of 4 up to 1024); fp32 at d 64 / 128
        (n 2 / 4), 256 / 512 (n 2) inference and d 128 / 256 / 512 (n 2) training runs ``fused_wide_sm90a.transition_f32_sm90a`` instead
        (the bf16 builds' fused forwards with fp16 products at TF32's rounding; the generic TF32 backward). bf16 / fp32
        parameters (fp32 masters cast inside, their gradients fp32). ``"triton"``: the Triton split Transition
        (``kernels/transition/triton/residual.transition_residual``: the LayerNorm that also hands back the residual, the expand-SwiGLU,
        the squeeze with the residual in its epilogue, and their backward), any width, bf16 / fp16 / fp32 (fp32 dots IEEE unless torch
        allows TF32), the parameters cast to the activation dtype.

    """

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 3,
        "anthropic:a100_sm80": 4,  # 2026-11-13: the A100 kit stack; the pair_c256_n4 exclusion is capability 9.0 only (A100: the provider cell serves)
        "cuda:h100_sm90": 8,  # 2026-11-21 (8): fp32 with TF32 off (training and inference) runs on the engine's IEEE fp32 CUDA chain (kernels/ieee_f32 built for sm_90a, as the A100 / RTX 5090 7; was refused to Triton / PyTorch); TF32-allowed fp32 and bf16 unchanged
        "cuda:a100_sm80": 8,  # 2026-11-20 (8): bf16 one-kernel forward (fused_fwd_sm80 / fused_sm80, D 64 / 128) keeps its f16 squeeze finite for |x| up to 1e6 (residual and Ws carried x 2^-4; was inf at |x| >= 65504: AF3 5I28 msa_transition); results otherwise within 1 bf16 ulp of 7; 2026-11-19 (7): fp32 with TF32 off runs on the engine's IEEE fp32 CUDA chain (kernels/ieee_f32; was refused to Triton / PyTorch); TF32 and bf16 unchanged
        "triton:a100_sm80": 2,
        "cuda:5090_sm120": 10,  # 2026-11-20 (10): the same bf16 one-kernel forward fix as cuda:a100_sm80 8 (inf at |x| >= 65504 in the D 64 / 128 f16 squeeze); (9): 2026-11-20 (9): fp32 inference (d 128, n 2, float32 TF32-allowed) squeezes in f16 accumulation with 32 rows a warp and up to two half tiles a CTA (kernels/transition/cuda/sm80/tr_fwd_sm80.cuh F32H, 0.97-1.03 -> 0.94-0.96 of anthropic; rms vs float64 5.9e-4, PyTorch TF32 3.4e-4); (8): 2026-11-19 (8): bf16 inference at D >= 256 and >= 32768 rows folds the LayerNorm into the dual GEMM (statistics kernel + gamma-folded weights, xn never written; kernels/transition/cuda/fused_wide_sm80._dual_ln), -80 us of the L 512 d 256 n 2 call, outputs differ from 7 within the accuracy standard; (7): see the comment below; (6): bf16 training on the one-kernel backward does not save xn (rebuilt by ln_apply from the saved (mean, rstd), -36 MiB at L 384; gradients bitwise unchanged); (5): the same IEEE fp32 chain (fp32 TF32-off training at 2048 / 4096 / 8192 rows and the 384-row no-grad calls fell to Triton)
        "cuda:b200_sm100": 67,  # 2026-11-22 (67): fp32 with TF32 off (training and inference) runs on the engine's IEEE fp32 CUDA chain (kernels/ieee_f32 built for sm_100a, as the H100 8 / A100 7; was refused to Triton: the AF3 / Protenix diffusion-head conditioning transitions); TF32-allowed fp32 and bf16 unchanged
        "triton:b200_sm100": 22,
    }
    # 2026-11-19 (B200, N9, inference): cuda:b200_sm100 66 = the bf16 expand epilogue (tsx_expand.cu: the D 384 / D 768 small-M chains, the fold, the K-split cluster) on packed f32x2 pairs with the one-MUFU tanh sigmoid (0.5 tanh(a / 2) + 0.5, the TR_SIG 1 / SIGT 1 form of the other bf16 kernels) in place of ex2 + rcp (4 MUFU ops a pair -> 2): row 36 / 37 chains -0.3 - -0.6 us device time; end-to-end error vs fp64 unchanged to 3 digits (rms), fp32 / fp16 expands untouched
    # 2026-11-19 (B200, N1, inference): cuda:b200_sm100 62 = (1) the small-M D 384 chain (4 - 24 tiles of n 2 as far as the chain takes them (<= 16), 5 - 12 of n 4) with the LayerNorm folded into the expand (tsx_expand.cu -DFOLD: raw x as A, Wp = bf16(W gamma) as B and one K16 correction MMA, row statistics from the x tile in shared memory by the epilogue warps; Wp and the u / v images are a weight form made once per weights): no LayerNorm launch, device time -0.2 - -3 us (the 128-unit chunk, with its own weight ring beside the resident x tile, also covers the cells that used fused_small's chains: n 4 8 - 12 tiles keep its CTA-pair squeeze, n 2 13 - 16 tiles; a 7-tile n 4 and 13 - 15-tile n 2 hole of 2 - 3 us is gone); the squeeze of 7 - 12 tiles of n 2 on the 64-column CTA pair (the old rule fell back to (32, 1) there: -0.2 - -0.8 us); (2) the fused D 384 inference kernel's fold prep (Wp, u / v images: a 4.5 us launch plus a zero fill in front of every call) cached as a weight form, so the fold now serves n 2 from 16 384 rows (was 65 536) and n 4 from 32 768: -4 - -7 %; (3) the D 512 inference fold prep (Wp, u, v: 32 blocks of the statistics launch) cached as a weight form: the fold now serves n 2 and n 4 from 16 384 rows (-1 - -4 % device time); (4) route_sm100a: the zero-row pad of a ragged row count only where a tuned build runs (it ran, 2 kernels / ~4 us in a graph, for the small-M chains and the generic kernels that need none: row 36 L 2976 26.5 -> 23.1 us)
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 60 = the D 512 n 2 inference squeeze GEMM (tgemm_nd.cu -DPARTS) with the K loop per N part and a double-buffered 256-column accumulator: the epilogue of one part overlaps the
    # MMAs of the next (the single whole-row accumulator left the tensor pipe idle ~68 us of 244 at row 33 L 512): squeeze 244 -> 235 us, row 33 master -2 to -2.5 %
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 58 = the fused D 384 inference kernel (rows 31 / 32 from 65 536 rows) with the LayerNorm folded into the expand: raw x as the A operand (no apply pass), W gamma as B (prep launch),
    # the row statistics in the SwiGLU warps and a K16 CORRECTION MMA per chunk ([-mean q] x [u v] bf16 hi / lo pairs) for the rank-1 terms; the SwiGLU epilogue is a = rstd acc: -3 to -9 % at 65 536 rows and up
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 56 = the D 512 fold's statistics pass at 16 lanes a row and 8 blocks a SM (68 -> 47 us at row 33 L 512, 88 % of the HBM peak)
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 54 = D 512 n 2 inference (row 33, 65 536 .. 524 288 rows) with the LayerNorm folded into the expand GEMM (raw x as the A operand, W gamma as B,
    # a = rstd (x . Wp) - c1 u + v; one statistics + weight-fold launch: no xn round trip): row 33 master -3 to -5 %
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 52 = the fp32 inference forward tg_fused_h16 at D 128 (row 27 float32, from 98 304 rows) with the y epilogue split over both warp groups, no staging
    # (EP3: residual re-read from global in 128-B segments, y by row-segment stores): [262 144, 128] n 2 103.1 -> 91.2 us, [589 824] 223.2 -> 196.5; same bits
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 50 = the D 256 inference forward (tfwd_d256.cu -DX2=1) with TWO xn buffers and no x buffer (the LayerNorm of tile i + 1 overlaps the
    # expands of tile i; the residual is re-read from global memory, y written in 128-B row segments): [589 824, 256] n 2 13.5 -> 11.9 us a tile, n 4 24.6 -> 21.5; same bits
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 48 = the D 256 inference forward (tfwd_d256.cu) prefetches the next tile's x into L2 a tile ahead (PF2; same bits,
    # [589 824, 256] n 2 14.0 -> 13.6 us a tile, n 4 22.85 -> 22.60)
    # 2026-11-18 (B200, S90, inference): cuda:b200_sm100 46 = the D 128 n 2 inference pair forward (tr_fwd_sm100.cu) with 3 x stages and a 2-slot weight ring
    # (NXS_ / NW_: the tile period is the x stage lifetime / stages), 4.59 -> 3.91 us a tile at [1 048 576, 128], bit-identical
    # (S90 part 7, training, 53): the D 256 training forward on the inference X2 build (two xn buffers, residual re-read, y from registers; the kernel now takes x / y as
    # extra arguments, rstd / c1 stay the saved statistics): [98 304 / 147 456, 256] n 4 step 672 / 913 -> 647 / 883 us, n 2 [147 456] 565 -> 532; one tile a CTA keeps the old build.
    # 2026-11-18 (B200, S90 part 6, training): cuda:b200_sm100 53 (51 in the log: part 6) = the D 64 n 2 training forward with ROLES 1 (FWD_TRAIN_FLAGS): [589 824, 64] step 239.7 -> 227.3 us,
    # y bit-identical; (45: the D 64 pair backward of part 5).
    # 2026-11-18 (B200, S90 part 5, training): cuda:b200_sm100 45 = the D 64 two-role backward (rows 209 / 210 / 220): the DX input stage carries x (XSEP: no
    # x reload behind the producer's in-order issue), n 2 on one gate warpgroup + two LayerNorm-backward epilogues on alternate tiles, a double-buffered
    # d_xn accumulator and three tile buffers, D 64 n 2 weight-role replicas 34 from 129 tiles: [589 824, 64] n 2 step 291 -> 242 us, [147 456] 88.1 -> 79.8,
    # [65 536] 51.2 -> 49.2, n 4 [589 824] 387 -> 371; y / dx bit-identical, dW regrouped (GREEN), deterministic.
    # 2026-11-18 (B200, S90, training): cuda:b200_sm100 43 = fp32 training: D >= 256 keeps a, b from the TF32 forward (GATE_AB; D 256 leaves the fp16
    # forward) and D 128 / 512 take d_xn from cuBLAS (fused_generic_sm100a._AB_F32_MIN_D, _DXN_CUBLAS_F32): D 512 n 2 step -22 to -31 %, D 256
    # -4 to -7 %, D 128 -2 to -4 %.
    # bf16 two-role backward: tile-dependent weight-role replicas (D 128 n 2 15 / 16 above 128 / 2048 tiles, D 128 n 4 10 from 3200 tiles,
    # D 64 n 4 19): D 128 n 2 [262 144 .. 589 824] -5 to -9 %, n 4 [589 824] -3 %, D 64 n 4 -2 to -3 %; dx / y bit-identical, dW regrouped
    # (GREEN). Grads bit-identical to 42 except D 256 >= 65 536 rows (forward on TF32 instead of fp16); deterministic.
    # 2026-11-18 (B200, round 3 part 20, inference): cuda:b200_sm100 42 = the fp32 inference forward tg_fused_h16 (rows 27 / 29 float32):
    # SwiGLU sigmoid as 0.5 tanh.approx(z / 2) + 0.5 (one MUFU op a unit; HSIG 1, the bf16 kernels' form), and a CTA's last tile takes
    # its residual from a TMA copy into the idle weight ring and writes y by TMA stores (RXT; the same bits); a reconverging __syncwarp
    # before the final barrier. Random weights vs fp64: rms 3.652e-4 / 3.880e-4 at [16 384, 256] / [65 536, 128] n 2 (was 3.652e-4 /
    # 3.879e-4); 4-10 % faster at every row count. The training build (SAVE_BUILD) keeps its sigmoid and epilogue.
    # 2026-11-18 (B200, round 3 part 19, inference): cuda:b200_sm100 40 = bf16 inference at D 768 n 2 13-16 row tiles and D 384 n 4 13-31
    # tiles (rows 37 / 36 L 2048-3968, ragged rows too) on the small-M chain with the PERSISTENT pair expand (tsx_expand2p.cu: M = 256
    # pair products, resident weight panels, 72 pairs looping over 256-row blocks, double-buffered accumulators) and the cuBLAS squeeze
    # onto the x copy the LayerNorm launch writes (fused_small_sm100a.PEX), instead of the generic kernels
    # 2026-11-18 (B200, round 3 part 18, inference): cuda:b200_sm100 38 = bf16 inference at D 768 n 2 7-12 row tiles, D 384 n 2 16 tiles and
    # D 384 n 4 8-12 tiles (rows 37 / 35 / 36 L 1024-2048) on the small-M chain (fused_small_sm100a.INFER_CELLS: tsx_ln -> K-split
    # tsx_expand -> the squeeze on a CTA pair, tsx_squeeze2 bf16 NB 128 / 64) instead of the generic kernels with the cuBLAS squeeze
    # 2026-11-18 (B200, round 3 part 17, inference): cuda:b200_sm100 36 = the small-M chains' K-split partials exchanged chunk-major
    # (tsx_expand / tsx_squeeze -DCHM: one 512-B contiguous st.async run per warp instruction; same bits, training too) and the fp32
    # inference fp16 chain's squeeze on a CTA pair (tsx_squeeze2 -DH16, no partial exchange) at the H16_PAIR_SQ cells
    # 2026-11-18 (B200, round 3, training part 5): cuda:b200_sm100 33 = the two-role backwards' gates reordered (same bits): the D 128 / D 64
    # weight role computes h / dA / dB before waiting for the previous tile's dW products (only its stores wait), the D 128 input role
    # computes the a | b part of the gate (s, l, u) before waiting for dh_j
    # 2026-11-18 (B200, round 3 part 16, inference; merged into 34 with training 33): cuda:b200_sm100 32 = tsx_ln (the small-M / pair chains' LayerNorm, bf16 and the fp16
    # inference build) loads gamma / beta before griddepcontrol.wait instead of after the row statistics (one dependent DRAM round trip
    # off the chain's first launch; bitwise identical): fine device time of row 37 master [1, 256 / 384 / 768, 768] -0.9 / -1.2 / -1.1 us
    # 2026-11-18 (B200, round 3, training part 4): cuda:b200_sm100 31 = the training forward (D 128) and the two-role backwards (D 64, D 128)
    # launched as programmatic dependents (griddepcontrol.wait at the forward's entry and after the backward's prologue; same bits), the
    # D 128 n 2 reduce too; D 128 n 4 backward with 8 weight-role replicas up to 128 row tiles (dW partial sums regrouped)
    # 2026-11-18 (B200, round 3, training part 3): cuda:b200_sm100 29 = tg_bwd_small: barrier arrivals as gpu-scope release atomics, the
    # proxy fences only at the barrier whose data is TMA-loaded next, the gate elementwise with two vectors per load round (no re-reads of
    # unused slabs), the LayerNorm phase on several warps per row at D 768 / <= 2 rows per CTA (dx bits change: row sums in part order),
    # measured phase-2 / gate plans for bf16 [512-768, 768] (SB_PLAN_TUNED, SB_GATE_TUNED)
    # 2026-11-18 (B200, round 3 part 14, inference): cuda:b200_sm100 28 = bf16 D 768 n 2 inference at 1-6 row tiles (row 37 single stream) on the
    # CTA-pair chain tsx_ln -> tsx_expand2 -> tsx_squeeze2 (cta_group::2, M = 128 pair products, 3-D multi-K-block TMA boxes; fp32 sums over the
    # whole K instead of the K-split partials: output within one bf16 ulp of 26 / 27)
    # 2026-11-18 (B200, round 3, training): cuda:b200_sm100 27 = the small-M chain cells (fp32 D 384 / 768, bf16 D 768, n 2, 1-6 row tiles) train
    # with the one-launch backward tg_bwd_small (persistent grid, software grid barriers: split-K gate + elementwise gate, d_xn beside dW, the
    # LayerNorm backward, dgamma | dbeta), the weights read MN-major (no transposed copies) and their per-step forms made in the forward chain's
    # LayerNorm launch (no tg_prep); fp32 dy as it is (TF32-truncated by the MMAs, no split): YELLOW 1.37-1.60 x tol (was 1.00-1.08)
    # 2026-11-17 (B200, round 3 part 9, training): cuda:b200_sm100 25 = fp32 training weight gradients on cuBLAS over 8 / 16 row slabs (one
    # bmm each, the slabs and the dgamma | dbeta partials added in fixed order by one tg_wsum launch; tg_dw now only to 2^21 x elements);
    # the backward's partial reductions (transition_bwd_reduce, tbwd_d64 / d384 reduce, tg_wsum, tsx_dw,
    # transition_lnbwd_w_reduce) with their loads batched (same order, same bits)
    # 2026-11-18 (B200, round 3 part 11, inference): cuda:b200_sm100 26 = tg_fused_h16 D 256 inference: both 16-row groups' LayerNorm statistics
    # before the wait for the xn buffer, the first tile's first group normalized by the idle SwiGLU warps (LNA) from an x tile TMA-staged in
    # the second expand stage / squeeze ring (XS0); the same bits. Training build (SAVE_BUILD) unchanged.
    # 2026-11-17 (B200, round 3 part 10, inference): cuda:b200_sm100 24 = the fp32 inference forward tg_fused_h16 (D 64 / 128 / 256): gamma | beta
    # staged in shared memory by a bulk copy, the epilogue's residual read coalesced through the staging slab and y written by coalesced stores
    # (the same bits); D 256: the LayerNorm reads its rows in the tcgen05 16x256b fragment order (4x fewer L1 wavefronts; xn within the
    # LayerNorm's summation order), x tiles prefetched into L2. The training build (SAVE_BUILD) is unchanged
    # 2026-11-17 (B200, round 3 part 9, inference): cuda:b200_sm100 22 = inference cubins of the fused wide forwards (fused_wide_sm100a.FWD_INFER_FLAGS):
    # D 64 n 2 with the epilogue on its own warpgroup (same bits), D 128 n 2 with the LayerNorm and the epilogue split over two warpgroups (same
    # bits), D 256 with the LayerNorm and the epilogue on two warpgroups and the x / weight boxes issued from one lane each (xn within the LayerNorm's
    # summation order); triton:b200_sm100 22 = fp32 inference with K > 128: the LayerNorm stores xn on the TF32 grid (layernorm/triton TF32_OUT,
    # the same bits), no launcher rounding pass (one launch fewer)
    # 2026-11-17 (B200, round 3 part 8): cuda:b200_sm100 20 = bf16 D 384 inference from 8 row tiles (n 4) / 16 (n 2) on the generic kernels with
    # the cuBLAS squeeze (also wherever the RESID squeeze's tiles leave SMs idle); fp32 D 256 training d_xn on cuBLAS; wide-build training at
    # D >= 256 from fp32 masters: the bf16 forms + Ws^T / [Wa; Wb]^T in one tg_prep launch, the dgamma | dbeta sum inside tg_dw / tg_wsum
    # 2026-11-17 (B200, round 3 part 7): cuda:b200_sm100 7 = fp32 TRAINING forward on the fp16 tensor cores at D 64 / 128 / 256 from 65 536 rows
    # (tg_fused_h16 with saves of xn / mean / rstd; the step's fp16 forms + scales and the backward's TF32 copies in one tg_prep16 launch; the
    # H100's F32_TRAIN_SHAPES); fp32 inference: the fp16 chain from one row tile at D 384 and up to 12 tiles at D 768, the three-launch fp16
    # forward at D 768 above (YELLOW inside the engine standard)
    # 2026-11-17 (B200, round 3 part 5): cuda:b200_sm100 6 = the backward's weight gradients on one tg_dw launch (MN-major tcgen05 GEMM over the rows,
    # kind::tf32 with the 128-B-swizzle 32-B-atom layout for fp32, a deterministic row split whose slabs tg_wsum adds in order) instead of the two
    # cuBLAS GEMMs, in the generic and the wide (D 256 / 384 / 512) backward; the generic GEMM / row launches as programmatic dependent launches;
    # tg_colsum folded into tg_dw where the rows are not split; fp32 takes tg_dw only up to 2^22 x elements (cuBLAS above: measured faster)
    # 2026-11-17 (B200, round 3 part 4): cuda:b200_sm100 5 = fp32 inference on the fp16 tensor cores beyond D 256: the small-M chains at D 384 (2-16 row
    # tiles) / D 768 (up to 8) as -DH16 builds (fp16 xn / weights / h, fp32 x / out) and D 512 in three launches (tsx_ln -DH16 -> tg_gemm -DH16 SWIGLU ->
    # RESID), the H100 f32io / lnsg_f32 scheme; the fp16 h scale from a Cauchy-Schwarz bound (s_h = 1 for ordinary weights), which also lifts the D 256
    # 2^20-row cap of tg_fused_h16
    # 2026-11-17 (B200, round 3 part 2): cuda:b200_sm100 4 = the small-M chains (tsx_ln -> tsx_expand -> tsx_squeeze, programmatic dependent launches) beyond
    # D 384 bf16: fp32 on the TF32 tensor cores at D 384 / 768 n 2 and bf16 at D 768 n 2 (fused_small_sm100a; K-block rings, the expand's K split over a
    # cluster), in inference and as the forward of the generic training step (packed [a | b] saved for GATE_AB)
    # 2026-11-17 (B200, round 3): cuda:b200_sm100 3 = fp32 inference at D 64 / 128 / 256 (D 256 up to 2^20 rows) on the one-launch tg_fused_h16 (fp16
    # operands as the H100 f32io forward; the TF32 tg_fused_f32 removed), D 64 n 2 training back on the wide build (tbwd_d64 -DDXN32: d_xn fp32
    # through the LayerNorm backward, GREEN), inference weight forms read by a graph capture as the warm-up made them (capture_reads_eager)
    # 2026-11-17 (B200): cuda:b200_sm100 2 = D 64 n 2 training on the generic kernels (route_sm100a.GENERIC: the wide build was YELLOW there)
    # 2026-11-17 (B200): triton:b200_sm100 1 = fp32 with K > 128: x rounded to the TF32 grid in a launcher pass instead of inside the expand kernel
    # (kernels/transition/triton/main.ROUND_X_IN_KERNEL_SM100), the TF32 weight forms cached by address (kernels/_tf32.tf32_cached by_address)
    # 2026-11-16 (B200): cuda:b200_sm100 1 = routed on the B200 (kernels/transition/cuda/route_sm100a: fused_sm100a, fused_wide_sm100a, fused_generic_sm100a,
    # ported from the b200/ladder-final branch)
    # 2026-11-19 (RTX 5090): cuda:5090_sm120 7 = the xn rebuild of 6 only up to 147456 rows (L 384 pair; fused_wide_sm80.SM120_XN_RECOMPUTE_MAX_ROWS): above it the ln_apply pass cost 4-10 % of the D 64 / 128 training step at L 512 .. 768 (tile / routing items of this version: see the commit message)
    # 2026-11-14 (RTX 5090): cuda:5090_sm120 4 = calls past 64 x 65535 rows (L 2048 pair, D >= 256 eval) take 128-row tiles (they failed with a grid y
    # overflow: invalid configuration argument)
    # 2026-11-13: anthropic 3 = the pair transition folds the residual in the kernel (the kits' block statement); the esm_t16 cell served wherever the driver loads its CUDA-13 cubin
    # 2026-11-13 (RTX 5090): cuda:5090_sm120 2 = the one-kernel D 64 / 128 backward on sm_120a (32-row PW stages), half tiles of the fused forward at few rows;
    # 3 = the main merge (the A100 changes 5 / 6 run on the sm_120a build too)
    # 2026-11-13 (H100): cuda 5 = the inference forward casts fp32-master weights once per parameter version (param_forms), not per call
    # 2026-11-18 (H100): cuda 7 = TRAINING runs the generic kernels at D 768 (n 2) above 768 rows, D 384 above 16384, D 512 above 12288 rows (they save xn / a / b
    # for the backward): D 768 step 1.2 - 1.85x, D 512 8 - 17 %, D 384 up to 12 % faster; the D 384 / 512 chunked inference writes in place (same bits)
    # (the training arithmetic differs from the wide build; judge standard: engine rms vs float64 3.6 - 4.9e-3, PyTorch 5.0 - 6.9e-3)
    # 2026-11-17 (H100): cuda 6 = D 768 (n 2) inference above 768 rows runs the generic kernels, not the wide D768 training structure (1.4 - 1.9x
    # faster at 1024 - 8192 rows; ~2.3 % of the outputs differ, the error against float64 is the same: judge max 3.50e-3, rms 4.37e-3)
    # 2026-11-10: kits; H100 d384 cuBLAS; fp32 eval f32io; fp32 fused fwd d256 / d512 + train; A100 capture forms + fp32 (TF32) build.
    # 2026-11-11 (A100): cuda 3 = the fused D 64 / 128 builds again + the fused fp32 eval forward; triton 2 = TF32-tuned fp32 d768 keys
    # 2026-11-12 (A100): cuda 5 = a call split past one launch's rows (OpenDDE refiner pair rows) runs balanced blocks written into one output (no join copy); the same bits
    # 2026-11-12 (A100): cuda 6 = the saved-activation training backward runs its weight-gradient GEMMs on a high-priority side stream and ln_bwd keeps two rows in flight (D 256-512); the same bits

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader): inference returns the residual input.
    dead: bool = False

    def __init__(
        self,
        d_hidden: int = 128,
        n: int = 4,
        master_weights: bool = False,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.n = n
        self.ln_in = LayerNorm(d_hidden, dtype=torch.bfloat16)
        self.expand_a = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=torch.bfloat16)
        self.expand_b = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=torch.bfloat16)
        self.squeeze = Linear(d_hidden * n, d_hidden, bias=False, init="zero", dtype=torch.bfloat16)
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """ALWAYS returns the residual output ``y = x + transition(x)`` (the residual is this module's own input ``x``)."""
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return x
        # The norm affine is fp32-pinned and the projections may be fp32 masters: every parameter is cast to the activation dtype.
        h = F.layer_norm(
            x,
            (self.d_hidden,),
            self.ln_in.weight.to(x.dtype),
            self.ln_in.bias.to(x.dtype),
            self.ln_in.eps,
        )
        a = F.linear(h, self.expand_a.weight.to(x.dtype))
        b = F.linear(h, self.expand_b.weight.to(x.dtype))
        h = swish_gate(a, b)
        return F.linear(h, self.squeeze.weight.to(x.dtype)) + x

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        shape = stream_shape(stream, samples=samples, length=length, depth=depth, width=self.d_hidden)
        return {"x": random_input(shape, dtype=dtype, device=device, generator=generator)}

    # ------------------------------------------------------------------------------------------------------------------ anthropic
    #: The activation dtypes the provider has transition cells for (its cell table: bf16 | fp32).
    _ANTHROPIC_DTYPES = (torch.bfloat16, torch.float32)
    #: Provider cells of capability 9.0 that dump core below a CUDA-13 driver (H100: driver 575 / CUDA 12.9), refused by name: pair_c256_n4 (its
    #: serving call engages the cell's esm_t16 cubin, an sm_90a binary built with nvrtc 13.0, at every size measured: L 128 and 384). The same
    #: cell name on capability 8.0 (A100) is a portable Triton row (pf / v1): never excluded there.
    _ANTHROPIC_EXCLUDED_CELLS: tuple[str, ...] = ("|pair_c256_n4|",)

    @staticmethod
    def _anthropic_family(x: torch.Tensor) -> str:
        """The provider's cell family of the call: ``pair`` ([B, L, L, c]: L^2 rows), ``single`` ([B, L, c]), ``rows`` (an MSA [B, S, L, c])."""
        if x.dim() == 4 and x.shape[1] == x.shape[2]:
            return "pair"
        return "single" if x.dim() == 3 else "rows"

    def _anthropic_select(self, x: torch.Tensor) -> tuple[Any, tuple[Any, str | None]]:
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.transition")
        return prov, _anthropic.kernel_selection(
            prov,
            "fast",
            c=self.d_hidden,
            hidden=self.d_hidden * self.n,
            n_tokens=int(x.shape[-2]),
            dtype={torch.bfloat16: "bf16", torch.float32: "fp32"}[x.dtype],
            family=self._anthropic_family(x),
            residual=self._anthropic_family(x) == "pair",
            cc="{}.{}".format(*torch.cuda.get_device_capability(x.device)),
            rows_count=x.numel() // self.d_hidden,
        )

    def _anthropic_core_refusal(self, x: torch.Tensor) -> str | None:
        sel, why = self._anthropic_select(x)[1]
        from miniworld_engine.modules import _anthropic

        if (
            sel is not None
            and str(sel.cell_key).startswith("9.")
            and not _anthropic.cuda13_cubins_load()
            and any(c in str(sel.cell_key) for c in self._ANTHROPIC_EXCLUDED_CELLS)
        ):
            return f"the provider cell {sel.cell_key} dumps core below a CUDA-13 driver (its sm_90a cubin): refused by name"
        return why

    #: Where the shared core's provider has no cell for the call, the kits' own transition kernels of that form (the module is model-agnostic:
    #: the kit of the models the row serves):
    #:   ``"ef2"``  an MSA stream of width 128, bf16 (ESMFold2 / OpenDDE / Protenix-v2 rows): the esmfold2 kit's ``ef2_msa_v2.mtr_fused``
    #:              (LN + W12 + silu(x1) * x2 + W3 + the residual in one Triton kernel per row block; hidden a multiple of 64)
    #:   ``"pf"``   otherwise: ``opt_core.attn.pair_fused.transition`` (impl fpf, LN in the kernel; the pair transition the Boltz-2 pairfuse and
    #:              OpenFold3 kits install), served where its cell table / safe settings take (C, n*C)
    def _anthropic_route(self, x: torch.Tensor) -> tuple[str | None, str | None]:
        from miniworld_engine.modules import _anthropic

        if (
            self._anthropic_family(x) == "pair"
            and (self.d_hidden, self.d_hidden * self.n) == (384, 1536)
            and x.dtype == torch.bfloat16
            and not self.training
        ):
            # the pair transition of OpenDDE (c_z 384 x 4; the row is OpenDDE's alone): its kit's own composition (fpf_transition_odde sep16:
            # the module's LN -> bf16, two cuBLAS GEMMs, the Triton silu(a) * b kernel, the out GEMM)
            why = _anthropic.missing("fpf_transition_odde", path=_ODDE_ARMT)
            if why is None:
                return "odde", None
        why_core = self._anthropic_core_refusal(x)
        if why_core is None:
            return "core", None
        if (
            x.dtype == torch.bfloat16
            and self._anthropic_family(x) == "rows"
            and self.d_hidden == 128
            and (self.d_hidden * self.n) % 64 == 0
        ):
            why = _anthropic.missing("ef2_msa_v2", path="esmfold2/opt/forward/fast_inference/driver")
            if why is None:
                return "ef2", None
            why_core += f"; esmfold2 mtr_fused: {why}"
        why = _anthropic.missing("opt_core.attn.pair_fused")
        if why is None:
            ok, why = _anthropic.load("opt_core.attn.pair_fused").supported_transition(
                x, self._anthropic_pf_weights(x), impl="fpf", ln="fused", residual=False
            )
            if ok:
                return "pf", None
        return None, f"{why_core}; pair_fused.transition: {why}"

    def _anthropic_pf_weights(self, x: torch.Tensor):
        from miniworld_engine.modules import _anthropic

        params = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.ln_in.weight, self.ln_in.bias)
        key = (x.device, *((p.data_ptr(), p._version, p.dtype) for p in params))
        if getattr(self, "_anthropic_pf_key", None) != key:
            self._anthropic_pf = _anthropic.load("opt_core.attn.pair_fused").pack_transition_weights(
                w_out=params[2], ln_w=params[3], ln_b=params[4], w_a=params[0], w_b=params[1], eps=self.ln_in.eps, device=x.device
            )
            self._anthropic_pf_key = key
        return self._anthropic_pf

    def _supports_anthropic(self, x: torch.Tensor) -> str | None:
        from miniworld_engine.modules import _anthropic

        why = _anthropic.unavailable("opt_core.kernels.transition", x.device)
        if why is not None:
            return why
        if x.dtype not in self._ANTHROPIC_DTYPES:
            return f"dtype {x.dtype}: the provider has bfloat16 / float32 cells only"
        if x.dim() < 2:
            return "x must be [..., L, d_hidden]"
        if torch.is_grad_enabled():
            if any(p.requires_grad for p in self.parameters()):
                return (
                    "training with parameter gradients: the kits' differentiable transitions (opt_core transition_autograd rows esm_kd3*, "
                    "ef2inv LNLinearSwiGLUFrozen) are dX-only with frozen weights; freeze the parameters or call under torch.no_grad()"
                )
            return self._anthropic_autograd_select(x)[1][1]
        return self._anthropic_route(x)[1]

    def _anthropic_autograd_select(self, x: torch.Tensor):
        """The provider's differentiable row (``transition_autograd``: esm_kd3 / esm_t15_kd3 / esm_t16_kd3, forward + dX backward, frozen
        weights) for a frozen module whose input takes a gradient."""
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.transition")
        return prov, _anthropic.kernel_selection(
            prov,
            "fast",
            c=self.d_hidden,
            hidden=self.d_hidden * self.n,
            n_tokens=int(x.shape[-2]),
            dtype={torch.bfloat16: "bf16", torch.float32: "fp32"}[x.dtype],
            direction="fwdbwd",
            family="esmpair",
            residual=True,
            cc="{}.{}".format(*torch.cuda.get_device_capability(x.device)),
            rows_count=x.numel() // self.d_hidden,
        )

    def _forward_anthropic(self, x: torch.Tensor) -> torch.Tensor:
        if self.dead:  # inference only: a dead instance returns the residual input (as the reference does)
            return x
        if torch.is_grad_enabled():  # a frozen module, the input takes a gradient: the provider's differentiable row (dX backward)
            prov, (sel, _why) = self._anthropic_autograd_select(x)
            with torch.no_grad():
                pack = prov.pack(
                    w_a=self.expand_a.weight.to(x.dtype),
                    w_b=self.expand_b.weight.to(x.dtype),
                    w_o=self.squeeze.weight.to(x.dtype),
                    ln_w=self.ln_in.weight.to(x.dtype),
                    ln_b=self.ln_in.bias.to(x.dtype),
                    eps=self.ln_in.eps,
                )
            return prov.transition_autograd(x.contiguous(), pack, word="fast", n_tokens=int(x.shape[-2]), _sel=sel)[0]
        route, _ = self._anthropic_route(x)
        if route == "odde":
            from types import SimpleNamespace

            from miniworld_engine.modules import _anthropic

            kit = _anthropic.load("fpf_transition_odde", path=_ODDE_ARMT)
            if getattr(self, "_anthropic_odde", None) is None:  # OpenDDE's attribute schema; the kit caches its bf16 weights by version
                self._anthropic_odde = SimpleNamespace(
                    c_in=self.d_hidden,
                    linear_no_bias_a=self.expand_a,
                    linear_no_bias_b=self.expand_b,
                    linear_no_bias=self.squeeze,
                    layernorm1=self.ln_in,
                    training=False,
                )
            return kit.forward_sep16(self._anthropic_odde, x.contiguous()).to(x.dtype) + x
        if route == "ef2":
            from types import SimpleNamespace

            from miniworld_engine.modules import _anthropic

            kit = _anthropic.load("ef2_msa_v2", path="esmfold2/opt/forward/fast_inference/driver")
            key = tuple((p.data_ptr(), p._version) for p in self.parameters())
            if getattr(self, "_anthropic_ef2_key", None) != key:  # the kit caches W12^T / W3^T on this object by the weights' identity
                with torch.no_grad():
                    w12 = torch.cat([self.expand_a.weight, self.expand_b.weight], 0)  # [x1 | x2] rows: x1 the silu branch
                self._anthropic_ef2 = SimpleNamespace(
                    norm=self.ln_in, ffn=SimpleNamespace(w12=SimpleNamespace(weight=w12), w3=self.squeeze)
                )
                self._anthropic_ef2_key = key
            return kit.mtr_fused(self._anthropic_ef2, x.contiguous())  # x + transition(x), bf16
        if route == "pf":
            from miniworld_engine.modules import _anthropic

            PF = _anthropic.load("opt_core.attn.pair_fused")
            x = x.contiguous()
            return PF.transition(x, self._anthropic_pf_weights(x), residual=False, ln="fused", impl="fpf").to(x.dtype) + x
        prov, (sel, _why) = self._anthropic_select(x)
        if sel.row in (
            "engine_module",
            "compile",
        ):  # the provider measured the caller's own module fastest for this cell: the release runs it
            return self._forward_pytorch(x)
        params = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.ln_in.weight, self.ln_in.bias)
        key = (x.dtype, *((p.data_ptr(), p._version, p.dtype) for p in params))
        if getattr(self, "_anthropic_pack_key", None) != key:
            # the reference casts every parameter to the activation dtype (fp32 masters, the fp32-pinned LayerNorm affine): the same values here
            self._anthropic_pack = prov.pack(
                w_a=params[0].to(x.dtype),
                w_b=params[1].to(x.dtype),
                w_o=params[2].to(x.dtype),
                ln_w=params[3].to(x.dtype),
                ln_b=params[4].to(x.dtype),
                eps=self.ln_in.eps,
            )
            self._anthropic_pack_key = key
        x = x.contiguous()
        folded = self._anthropic_family(x) == "pair"  # the kits fold the pair block's residual into the kernel (z += T(z))
        y, served = prov.transition(
            x, self._anthropic_pack, word="fast", residual=folded, n_tokens=int(x.shape[-2]), family=self._anthropic_family(x)
        )
        if served.row != sel.row:
            from miniworld_engine.modules.primitives import BackendUnsupported

            raise BackendUnsupported(
                f"Transition backend 'anthropic': the provider served row {served.row!r}, its selection was {sel.row!r}"
            )
        # pair: x + update folded in the kernel (the kits' block statement); otherwise the update, then the residual (two roundings)
        return y if folded else y + x

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_engine(self, x: torch.Tensor) -> str | None:
        if x.dim() < 2 or x.shape[-1] != self.d_hidden:
            return f"x must be [..., {self.d_hidden}]"
        if x.numel() == 0:
            return "empty input"
        for p in (self.ln_in.weight, self.ln_in.bias):
            if p is None:
                return "the kernels take the affine LayerNorm (weight and bias)"
        return None

    def _supports_cuda(self, x: torch.Tensor) -> str | None:
        if a100(x) or sm120(x):  # A100 / RTX 5090: the sm_80 sources (built for sm_120a on the 5090)
            return self._supports_cuda_sm80(x)
        if b200(x):
            return self._supports_cuda_sm100(x)
        from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a, fused_sm90a, fused_wide_sm90a

        why = h100_unsupported(x) or self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {x.dtype}: the sm_90a kernels take bfloat16 (float32 on TF32) activations"
        matrices = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight)
        if any(p.dtype not in (torch.bfloat16, torch.float32) for p in matrices) or len({p.dtype for p in matrices}) != 1:
            return "the projections must share one dtype, bfloat16 or float32"
        wa, ws = self.expand_a.weight, self.squeeze.weight
        if x.dtype is torch.float32 and not fused_generic_sm90a._tf32_allowed():  # TF32 off: the IEEE fp32 chain (kernels/ieee_f32)
            from miniworld_engine.kernels import ieee_f32

            if ieee_f32.supported_transition(x, wa, self.squeeze.weight, hopper=True):
                return None
            return f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no IEEE fp32 build takes it (D a multiple of 4 up to {ieee_f32.MAX_D}, hidden a multiple of 4)"
        if any(k.supported(x, wa, ws) for k in (fused_sm90a, fused_wide_sm90a, fused_generic_sm90a)):
            return None
        return (
            f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}, {x.numel() // self.d_hidden} rows: no sm_90a build takes it (generic: d a "
            "multiple of 64 up to 1024, hidden a multiple of 128)"
        )

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return x
        if a100(x) or sm120(x):
            from miniworld_engine.kernels.transition.cuda import transition_sm80

            out = transition_sm80(
                x, self.ln_in.weight, self.ln_in.bias, self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.ln_in.eps
            )
            if out is None:
                from miniworld_engine.modules.primitives import BackendUnsupported

                raise BackendUnsupported("Transition backend 'cuda': no sm_80 Transition build takes this call")
            return out
        if b200(x):
            from miniworld_engine.kernels.transition.cuda.route_sm100a import transition_sm100a

            out = transition_sm100a(
                x, self.ln_in.weight, self.ln_in.bias, self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.ln_in.eps
            )
            if out is None:
                from miniworld_engine.modules.primitives import BackendUnsupported

                raise BackendUnsupported("Transition backend 'cuda': no sm_100a Transition build takes this call")
            return out
        from miniworld_engine.kernels.transition.cuda import transition_sm90a

        out = transition_sm90a(
            x, self.ln_in.weight, self.ln_in.bias, self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.ln_in.eps, self.n
        )
        if out is None:  # no build takes the call (a build failure raises _nvcc.KernelBuildError before this); never another backend
            from miniworld_engine.modules.primitives import BackendUnsupported

            raise BackendUnsupported("Transition backend 'cuda': no sm_90a Transition build takes this call")
        return out

    def _supports_cuda_sm80(self, x: torch.Tensor) -> str | None:
        """A100 and GeForce RTX 5090 (the same sources built for sm_120a; there only the wide chain: the one-kernel builds need more than its
        99 KB of shared memory a block): ``kernels/transition/cuda.transition_sm80`` (``fused_sm80`` at D 128 / n 4 with whole 256-row tiles and enough rows,
        ``fused_wide_sm80`` at D 64 / 128 / 256 / 384 / 512 / 768 with a hidden width a multiple of 64, any row count): bf16 activations,
        bf16 or fp32-master parameters, forward and backward; fp32 activations where torch allows TF32 (``fused_wide_sm80.transition_f32_sm80``:
        fp16 products at TF32's rounding, a TF32 backward) and, with TF32 off, IEEE fp32 (``kernels/ieee_f32``: the engine's own FFMA GEMMs, forward and backward)."""
        from miniworld_engine.kernels.transition.cuda import fused_fwd_sm80, fused_wide_sm80

        why = self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {x.dtype}: the sm_80 Transition kernels take bfloat16 (float32 on TF32) activations"
        matrices = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight)
        if any(p.dtype not in (torch.bfloat16, torch.float32) for p in matrices) or len({p.dtype for p in matrices}) != 1:
            return "the projections must share one dtype, bfloat16 or float32"
        wa, ws = self.expand_a.weight, self.squeeze.weight
        if x.dtype is torch.float32 and not fused_wide_sm80.tf32_allowed():
            # fp32 with TF32 off = IEEE products (the 2026-11-06 rule): the engine's own IEEE fp32 CUDA chain (kernels/ieee_f32: FFMA GEMMs with the SwiGLU /
            # residual / SwiGLU-backward epilogues, LayerNorm row kernels), any row count, forward and backward
            from miniworld_engine.kernels import ieee_f32

            if not ieee_f32.supported_transition(x, wa, ws):
                return (
                    f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no IEEE fp32 build takes it (D a multiple of 4 up to {ieee_f32.MAX_D}, "
                    "hidden a multiple of 4)"
                )
            return None
        rows, limit = x.numel() // x.shape[-1], fused_fwd_sm80.max_rows(x.device, x.dtype)
        if rows > limit and torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in self.parameters())):
            return (
                f"{rows} rows with a gradient: the sm_80 Transition kernels take at most {limit} rows per call (one launch's work lists / row "
                "grid); inference splits the rows, training does not"
            )
        if x.dtype is torch.float32:
            if not fused_wide_sm80.f32_supported(x, wa, ws):
                return (
                    f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no sm_80 fp32 build takes it (D in {fused_wide_sm80.F32_WIDTHS}, hidden a "
                    "multiple of 64)"
                )
            if not fused_wide_sm80.f32_available(x, wa, ws):
                return "the sm_80 Transition extension failed to build"
            return None
        if not fused_wide_sm80.supported(x, wa, ws):
            return (
                f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no sm_80 build takes it (D in {fused_wide_sm80.WIDTHS}, hidden a "
                "multiple of 64)"
            )
        if fused_wide_sm80.route(x, wa, ws) is None:
            return "the sm_80 Transition extension failed to build"
        return None

    def _supports_cuda_sm100(self, x: torch.Tensor) -> str | None:
        """B200 (sm_100a): ``kernels/transition/cuda/route_sm100a.transition_sm100a`` -- ``fused_sm100a`` (D 128, n 4) and
        ``fused_wide_sm100a`` (D 64-512 with n 2 / 4, D 768 with n 2) on whole 128-row tiles (a ragged bf16 call zero-padded to them), else
        ``fused_generic_sm100a`` (D a multiple of 64 up to 1024, hidden a multiple of 128, any row count; fp32 activations on TF32
        tensor cores, so only where torch allows TF32). bf16 / fp32 parameters (fp32 masters cast inside, their gradients fp32)."""
        from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a, route_sm100a

        why = self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {x.dtype}: the sm_100a kernels take bfloat16 (float32 on TF32) activations"
        matrices = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight)
        if any(p.dtype not in (torch.bfloat16, torch.float32) for p in matrices) or len({p.dtype for p in matrices}) != 1:
            return "the projections must share one dtype, bfloat16 or float32"
        wa, ws = self.expand_a.weight, self.squeeze.weight
        if x.dtype is torch.float32 and not fused_generic_sm100a._tf32_allowed():  # TF32 off: the IEEE fp32 chain (kernels/ieee_f32 built for sm_100a)
            from miniworld_engine.kernels import ieee_f32

            if ieee_f32.supported_transition(x, wa, ws, hopper=True):
                return None
            return f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no IEEE fp32 build takes it (D a multiple of 4 up to {ieee_f32.MAX_D}, hidden a multiple of 4)"
        if not route_sm100a.supported(x, wa, ws):
            return (
                f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no sm_100a build takes it (generic: d a multiple of 64 up to 1024, hidden a "
                "multiple of 128)"
            )
        return None

    def _supports_triton(self, x: torch.Tensor) -> str | None:
        why = triton_unsupported(x) or self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f"dtype {x.dtype}: the Triton kernels take bfloat16 / float16 / float32 activations"
        return None

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return x
        from miniworld_engine.kernels.transition.triton.residual import transition_residual

        rows = _as_rows3(x)
        # the reference casts every projection to the activation dtype (fp32 masters, bf16 weights under fp32 activations): the casts stay
        # in autograd, so a master's gradient lands on it in fp32
        y = transition_residual(
            rows,
            self.ln_in.weight,
            self.ln_in.bias,
            self.expand_a.weight.to(x.dtype),
            self.expand_b.weight.to(x.dtype),
            self.squeeze.weight.to(x.dtype),
            self.ln_in.eps,
        )
        return y.reshape(x.shape)


class ConditionedTransition(BackendModule):
    """AdaLN + SwiGLU transition + sigmoid conditioning gate. Given the RAW activation ``x`` and conditioning ``cond``::

        h = ada_ln_in(x, cond)                 # AdaLN
        a = h @ Wa^T ; b = h @ Wb^T            # expand_a, expand_b (no bias)
        out = (silu(a) * b) @ Ws^T             # SwiGLU, squeeze (no bias)
        scale = cond @ Wsc^T + b_sc            # to_scale (with bias)
        return x + sigmoid(scale) * out        # output gate, residual

    Parameters
    ----------
    d_hidden : int
        Feature dimension of ``x`` (and of the output).
    d_cond : int
        Feature dimension of the conditioning input ``cond``.
    n : int
        SwiGLU expansion factor (hidden width ``n * d_hidden``).
    dtype : torch.dtype
        Dtype of the projections and of the AdaLN (fp32 by default).
    backend : str
        See :class:`BackendModule`.

    ``backend="triton"`` (engine 2.4's portable path): the AdaLN on its own ``"triton"`` backend, then the tail on
    ``kernels/conditioned_transition/triton`` -- a gradient: ``training.cond_transition_train`` (cuBLAS GEMMs around the Triton SwiGLU /
    gate passes, the residual add in the gate pass; the backward's gate-grad and SwiGLU-grad in Triton); inference:
    ``inference.cond_transition_inference`` (the fully fused b2b kernel, d_hidden <= 128) or ``composed.cond_transition_inference_composed``
    (expand + SwiGLU, then squeeze + gate + residual; wider d). bf16 or fp32 activations (fp32 master weights under bf16 autocast cast
    per call), CUDA. A part of the bias-only token DiT's Triton path, not a requirements module of its own.
    """

    backends = ("pytorch", "triton")

    def __init__(
        self,
        d_hidden: int = 128,
        d_cond: int = 128,
        n: int = 2,
        dtype: torch.dtype = torch.float32,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_cond = d_cond
        self.n = n
        # Checkpoint keys: ada_ln_in.* + the flat tail.
        self.ada_ln_in = AdaptiveLayerNorm(d_hidden=d_hidden, d_cond=d_cond, dtype=dtype)
        self.dtype = dtype
        self.expand_a = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=dtype)
        self.expand_b = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=dtype)
        self.squeeze = Linear(d_hidden * n, d_hidden, bias=False, init="zero", dtype=dtype)
        self.to_scale = Linear(d_cond, d_hidden, bias=True, init="default", dtype=dtype)
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ALWAYS returns the residual output ``x + delta(x, cond)`` (the residual is this module's own input). A caller that composes the
        update some other way (a magnitude-preserving sum, a released checkpoint's own residual) takes :meth:`delta`."""
        return x + self.delta(x, cond)

    def delta(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """The update alone (no residual). ``x`` is the RAW residual-stream activation, ``cond`` the conditioning signal."""
        h = self.ada_ln_in(x, cond)
        out = master_linear(self.squeeze, self._hidden(h))
        scale = master_linear(self.to_scale, cond)
        return torch.sigmoid(scale) * out

    def _hidden(self, h: torch.Tensor) -> torch.Tensor:
        """The SwiGLU hidden ``silu(W_a h) * (W_b h)`` of the normalised ``h`` (``[..., n * d_hidden]``)."""
        a = master_linear(self.expand_a, h)
        b = master_linear(self.expand_b, h)
        return F.silu(a) * b

    # ------------------------------------------------------------------------------------------------------------ "triton"
    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        ada = getattr(self, "ada_ln_in", None)  # absent while nn.Module.__init__ runs the first set_backend
        if isinstance(ada, BackendModule):
            ada.set_backend(backend if ada.has_backend(backend) else "pytorch")
        return self

    def _supports_triton(self, x: torch.Tensor, cond: torch.Tensor) -> str | None:
        why = triton_unsupported(x) or _activation_dtype_reason(x, cond)
        if why is not None:
            return why
        if type(self) is not ConditionedTransition:
            return f"{type(self).__name__}: the Triton tail kernels compute the plain SwiGLU (no up-gate projection)"
        if not isinstance(self.ada_ln_in, AdaptiveLayerNorm):
            return f"the AdaLN is a {type(self.ada_ln_in).__name__} (the Triton path runs AdaptiveLayerNorm)"
        if any(m.bias is not None for m in (self.expand_a, self.expand_b, self.squeeze)) or self.to_scale.bias is None:
            return "bias-free expand / squeeze and a to_scale with a bias (the tail kernels' operands)"
        if tuple(x.shape[:-1]) != tuple(cond.shape[:-1]) or x.dim() < 2:
            return f"x [..., d_hidden] and cond [..., d_cond] with the same leading shape, got {tuple(x.shape)} / {tuple(cond.shape)}"
        return self.ada_ln_in._supports_triton(x, cond)

    def _forward_triton(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.conditioned_transition.triton import (
            cond_transition_inference,
            cond_transition_inference_composed,
            cond_transition_train,
        )

        h = self.ada_ln_in(x, cond)  # the AdaLN's "triton" backend (set_backend)
        dt, d = x.dtype, x.shape[-1]
        length = x.shape[-2]  # the kernels' shape key: the token / atom count of the un-flattened activation
        wa, wb, ws, wsc, bsc = (
            t.to(dt) for t in (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.to_scale.weight, self.to_scale.bias)
        )
        h2, c2, res = h.reshape(-1, d), cond.reshape(-1, cond.shape[-1]), x.reshape(-1, d)
        with torch.autocast("cuda", enabled=False):
            if _wants_grad(self, x, cond):
                y = cond_transition_train(h2, c2, wa, wb, ws, wsc, bsc, length, res)
            elif d <= 128:
                y = res + cond_transition_inference(h2.contiguous(), c2.contiguous(), wa, wb, ws, wsc, bsc, length)
            else:
                y = cond_transition_inference_composed(h2.contiguous(), c2.contiguous(), wa, wb, ws, wsc, bsc, length, res.contiguous())
        return y.reshape(x.shape)


class UpGatedConditionedTransition(ConditionedTransition):
    """Boltz-2's conditioned transition: :class:`ConditionedTransition` whose SwiGLU hidden is gated once more by a third projection of the
    normalised input (``up``, no bias)::

        hidden = silu(W_a h) * (W_b h) * (W_u h)
    """

    def __init__(
        self,
        d_hidden: int = 128,
        d_cond: int = 128,
        n: int = 2,
        dtype: torch.dtype = torch.float32,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(d_hidden, d_cond, n, dtype, backend=backend)
        self.up = Linear(d_hidden, d_hidden * n, bias=False, init="default", dtype=dtype)

    def _hidden(self, h: torch.Tensor) -> torch.Tensor:
        return super()._hidden(h) * master_linear(self.up, h)
