"""B200 (sm_100a) build and launch of the backbone pair-embedding kernels (``pair_embed_geom_sm100.cu``, 2026-11-16): the A100 source
``sm80/pair_embed_geom_sm80.cu`` (``sm90/pair_embed_geom.cu`` compiled unchanged with the mma.sync forms of its wgmma helpers: legal on
sm_100a, which has no wgmma) built for sm_100a, plus the table gradient on the B200 tensor cores (``pe_tbl_sm100.cuh``: tcgen05 / TMEM /
TMA, the kernel of the B200 branch). The launchers are the H100 ones (``sm90/__init__.py``) bound to this build, as the A100's are. Built
on first use, never at import."""

from __future__ import annotations

import functools
import types
from pathlib import Path

import torch

from miniworld_engine.kernels.pair_embed_geom.cuda import sm80 as _a100
from miniworld_engine.kernels.pair_embed_geom.cuda import sm90 as _h100

_dir = Path(__file__).parent
NBLK_TC = 74  # pair slices of the tcgen05 table gradient (x 2 channel halves = 148 CTAs, one per SM): pe100::NBLK


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="pair_embed_geom_sm100",
        sources=[str(_dir / "pair_embed_geom_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("100a"), f"-I{_a100._dir}"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


#: the launchers' module globals: the H100 launcher module's, with ``ext`` this build
_G = {**vars(_h100), "ext": ext}


def _on_this_build(fn):
    return types.FunctionType(fn.__code__, _G, fn.__name__, fn.__defaults__, fn.__closure__)


_forward_h100 = _on_this_build(_h100.forward)


def forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=False, ln_eps=None, jp=0, gt=None):
    """The encoder pair [B, n, n, 256] bf16 (and its codes): the H100 launch for int32 ``idx`` / ``chains`` (gt 1), the one-launch kernels of
    this build (gt 2: they read int32 or int64 indices and write the same codes) for the int64 indices of the module's lean preparation
    (2026-11-17: two cast kernels fewer per training step)."""
    if gt is None:
        gt = 2 if idx.dtype == torch.int64 else _h100.GT
    return _forward_h100(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes, ln_eps, jp, gt)


_forward_ca_h100 = _on_this_build(_h100.forward_ca)
#: the decoder pair's H100 launch on this card: pairs per item (jp; 0: the occupancy rule) and the table through the L1 (gt 1) or shared
#: memory (0). Measured jp 2-32 x gt 0 / 1: jp 16 through the L1 the best or within 3 % from L384 x S1 / L256 x S5 on
CA_JP, CA_GT = 16, 1


def forward_ca(tbl, ca, mask, idx, dlim, with_codes=False, ln_eps=None):
    """Decoder pair [B, n, n, 256] bf16 (the H100 launch with this card's :data:`CA_JP` / :data:`CA_GT`)."""
    return _forward_ca_h100(tbl, ca, mask, idx, dlim, with_codes, ln_eps, CA_JP, CA_GT)


def _eval_table(weight: torch.Tensor) -> torch.Tensor:
    """The bf16 table ``W^T`` [rows, 256] of an inference launch (the values of ``pg120_table``: ``tbl.copy_(weight.t())``), made once per weight
    version (``_capture.param_forms``; a capture reads the entry its warm-up made, so a replay runs the forward kernel alone: 2026-11-17, a
    CUDA-graph node costs ~2 us on this card and the table kernel was 2 of the 12-24 us of the L128 cases)."""
    from miniworld_engine.kernels import _capture

    def build():
        tbl = torch.empty((weight.shape[1], weight.shape[0]), dtype=torch.bfloat16, device=weight.device)
        tbl.copy_(weight.detach().t())
        return tbl

    return _capture.param_forms("pair_embed_geom_eval_tbl", (weight,), build, capture_reads_eager=True)


def forward_eval(weight, bb, ca_mask, has_cb, mask, residue_idx, chains, ln_eps):
    """B200 inference: the encoder pair [B, n, n, 256] bf16 (its LayerNorm when ``ln_eps`` >= 0) from the module's own inputs: the RTX 5090's
    one-launch forward (``sm80.forward_eval``, gt 2) on the cached table (:func:`_eval_table`)."""
    B, n = mask.shape
    dev = bb.device
    idx, ch = _a100._indices(residue_idx, chains)
    out = torch.empty(B, n, n, 256, device=dev, dtype=torch.bfloat16)
    none = torch.empty(0, device=dev, dtype=torch.int32)
    ext().forward(
        _eval_table(weight), bb.float().contiguous(), _a100._bytes(ca_mask), _a100._bytes(has_cb), _a100._bytes(mask), idx, ch,
        _a100._limits(dev, "dist"), _a100._limits(dev, "angle"), out, none, int(ln_eps >= 0), float(max(ln_eps, 0.0)), 0, 2,
    )
    return out


#: the decoder pair's inference launch (``pe_ca_sm100.cuh``): at most CA_CPS CTAs of 128 threads per SM (persistent warps). Measured
#: 2026-11-16 (measurer timing, CUDA graph + L2 flush), S1 / S5 x L128 / 256 / 384: 1.24-1.50x the Triton kernel without the LayerNorm,
#: 1.45-1.68x with it (S5 L384 63.4 / 65.5 us vs 79.7 / 100.2); the RTX 5090 one-launch kernel it replaces on this card: 12.2-168 us
CA_CPS = 32


def forward_ca_eval(weight, ca, mask, residue_idx, ln_eps, cps=None):
    """B200 inference: the decoder pair [B, n, n, 256] bf16 (its LayerNorm when ``ln_eps`` >= 0) from the module's inputs and its weight [256, 157]
    (``forward_ca_b200``: persistent warps, 8 lanes a pair, the inputs of the next 32 pairs in flight while the current ones are written)."""
    B, n = mask.shape
    dev = ca.device
    if residue_idx is None:
        residue_idx = torch.arange(1, n + 1, device=dev, dtype=torch.int32).expand(B, n).contiguous()
    (idx,) = _a100._indices(residue_idx)
    out = torch.empty(B, n, n, 256, device=dev, dtype=torch.bfloat16)
    eps = float(max(ln_eps, 0.0))
    ext().forward_ca_b200(
        _eval_table(weight), ca.float().contiguous(), _a100._bytes(mask), idx, _a100._limits(dev, "ca"), out, _none(dev), int(ln_eps >= 0), eps,
        int(cps or CA_CPS),
    )
    return out


LN_BWD_L1 = False  # the LayerNorm backward: False the H100 kernel (the table in shared memory), True the RTX 5090 one (the table through the L1)
#: inference from the module's own inputs per module (``modules/pair_embed.py`` _eval_sm120): the encoder pair by the RTX 5090's one-launch
#: forward (gt 2: S1 L128 16.6 vs 27.2 us the H100 launch, S5 L384 189.5 vs 226.0), the decoder pair by :func:`forward_ca_eval`
EVAL_ONE_LAUNCH = {"geom": True, "ca": True}


def one_launch(kind: str, x: torch.Tensor) -> bool:
    """Whether B200 inference of ``kind`` (geom / ca) on input ``x`` [B, n, ...] takes the one-launch forward."""
    return EVAL_ONE_LAUNCH[kind]


def ln_backward(tbl, codes, dh, mode, eps):
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) = ``dh`` (the pair recomputed from the codes and the table; zeros for padding)."""
    dx = torch.empty_like(dh)
    (ext().ln_backward if LN_BWD_L1 else ext().ln_backward_h100)(tbl, codes, dh, dx, int(mode), float(eps))
    return dx


#: the table gradient per mode (encoder 0 / decoder 1): "tc" the tcgen05 kernel; 1 the warpgroup kernel as mma.sync, 2 the mma.sync
#: kernel, 3 the A100's mma.sync kernel over KS concurrent pair chunks, 0 the shared-memory kernel (A/B)
BWD_IMPL: dict = {0: "tc", 1: "tc"}
#: at most this many pairs (B n n) the warpgroup kernel as mma.sync (impl 1) instead: measured L128 x S1 80.7 vs 83.9 us (encoder), 57.8 vs
#: 62.2 (decoder); from L256 / S5 on the tcgen05 kernel is 1.15-1.35x faster
SMALL_PAIRS = 128 * 128


def backward(g, codes, mode, impl=None):
    """Table gradient [N_ROWS[mode], 256] bf16 from d pair = g [B, n, n, 256] bf16 (deterministic, fixed-order partial sums)."""
    if impl is None:
        impl = 1 if g.shape[0] * g.shape[1] * g.shape[2] <= SMALL_PAIRS else BWD_IMPL[mode]
    nt = _h100.N_ROWS[mode]
    parts = NBLK_TC if impl == "tc" else _a100.parts_of(impl, mode)
    tpart = torch.empty(parts * nt * 256, device=g.device, dtype=torch.float32)
    dtbl = torch.empty(nt, 256, device=g.device, dtype=torch.bfloat16)
    if impl == "tc":
        ext().backward_tc(g, codes, tpart, dtbl, int(mode))
    else:
        ext().backward(g, codes, tpart, dtbl, int(mode), int(impl))
    return dtbl



# ---- round 3 (2026-11-17): the B200 kernels of both builders from the module's own inputs (pe_geom_sm100.cuh, pe_ln_sm100.cuh) -------------
def _none(dev) -> torch.Tensor:
    return torch.empty(0, device=dev, dtype=torch.int32)


#: encoder forward / LayerNorm backward: lanes a pair (8 / 16 / 32) and CTAs of 128 threads per SM at most (persistent warps); the forward
#: without the LayerNorm above GEOM_BIG pairs: GEOM_G_BIG lanes. Measured 2026-11-17 (measurer timing, S1 L128 / 256 / 384, us): forward G 16 x
#: CPS 4 eval 20.5 / 40.9 / 63.5 (G 8 x CPS 8: 24.4 / 51.0 / 81.6; G 32 x CPS 4: 20.5 / 38.9 / 61.4, with the LayerNorm 22.5 / 47.0 / 77.9 vs
#: G 16 22.5 / 45.0 / 73.8); LayerNorm backward CPS 8 vs 4: training L256 / L384 104.5 / 190.3 vs 108.5 / 192.5 (encoder), 51.2 / 84.0 vs
#: 57.3 / 98.2 (decoder)
GEOM_G, GEOM_CPS = 16, 4
GEOM_G_BIG, GEOM_BIG = 32, 128 * 128
#: above GEOM_BIG pairs GEOM_CPS_BIG CTAs per SM, on the forward build with its registers capped for 6 CTAs (pe_geom_sm100.cuh MB; S1 L256 /
#: L384 eval 36.7 / 59.3 us vs 38.8 / 63.4 at CPS 4 uncapped, with the LayerNorm 44.9 / 71.6 vs 44.9 / 77.7; at L128 the cap costs 2 us)
GEOM_CPS_BIG = 6
#: the forward with the table in shared memory (fwd_smt, one CTA of SMT_NW warps per SM; 0: off) above SMT_MIN pairs; lanes a pair
SMT_NW, SMT_MIN, SMT_G, SMT_G_LN, SMT_TRAIN, SMT_CH = 16, 128 * 128, 16, 16, True, 32
LN_G, LN_CPS = 16, 8
#: the LayerNorm backward at most GEOM_BIG pairs: LN_G_SMALL lanes a pair (2026-11-17, round 3 restart, measurer timing, S1 L128 training
#: with the LayerNorm: decoder 32.6 -> 30.6 us, encoder 51.1 either; at L256 / L384 8 lanes cost the decoder 2 - 6 us)
LN_G_SMALL = 8
#: pair slices (= fp32 partials, x 2 channel halves = CTAs) of the tcgen05 table gradient by pair count: (most pairs, slices), else NBLK_TC
TC_NBLK: tuple = ()
#: the reduce of the partials: 0 four columns a thread, 16 partials in flight; 1 two columns a thread, all partials in flight; 2 inside the
#: table-gradient launch after a grid barrier (cooperative, pe_tbl_sm100.cuh FR)
RED_V = 2
_BAR: dict = {}


def _bar(dev) -> torch.Tensor:
    """The fused reduce's grid barrier words (arrivals, generation) of a device: zeroed once, left at zero arrivals by every launch."""
    key = torch.device(dev).index
    if key not in _BAR:
        _BAR[key] = torch.zeros(2, device=dev, dtype=torch.int32)
    return _BAR[key]


def _chunk(pairs: int, cps: int, device) -> int:
    """Pairs a warp chunk: the largest of 32 / 16 / 8 that still gives every warp slot of the launch (cps CTAs of 4 warps per SM) a chunk."""
    slots = torch.cuda.get_device_properties(device).multi_processor_count * cps * 4
    for c in (32, 16):
        if pairs >= slots * c:
            return c
    return 8


def table(weight: torch.Tensor, train: bool) -> torch.Tensor:
    """The bf16 table ``W^T`` [rows, 256] the B200 kernels read: one kernel in a training step; inference: made once per weight version (a
    capture reads the entry its warm-up made)."""
    if not train:
        return _eval_table(weight)
    tbl = torch.empty((weight.shape[1], weight.shape[0]), dtype=torch.bfloat16, device=weight.device)
    tbl.copy_(weight.detach().t())
    return tbl


def forward_b200(tw, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps, train, g=None, cps=None, ch=None):
    """(encoder pair [B, n, n, 256] bf16 -- its LayerNorm when ``ln_eps`` >= 0 --, codes [B, n, n, 4] int32, empty unless ``train``, the bf16
    table) from the bf16 table (:func:`table`) or the module's weight [256, 317] (a training step: the launch makes the table, pg120_table,
    and the forward follows it by PDL), fp32 ``bb``, byte masks and int32 / int64 ``idx`` / ``chains``."""
    B, n = idx.shape
    dev = bb.device
    out = torch.empty(B, n, n, 256, device=dev, dtype=torch.bfloat16)
    codes = torch.empty(B, n, n, 4, device=dev, dtype=torch.int32) if train else _none(dev)
    cps = int(cps or (GEOM_CPS_BIG if B * n * n > GEOM_BIG else GEOM_CPS))
    g = g or (GEOM_G_BIG if ln_eps < 0 and B * n * n > GEOM_BIG else GEOM_G)
    if SMT_NW and B * n * n > SMT_MIN and (not train or SMT_TRAIN):
        g = SMT_G_LN if ln_eps >= 0 else SMT_G
        made = ext().forward_b200s(
            tw, bb, ca, cb, mask, idx, chains, dlim, alim, out, codes, int(ln_eps >= 0), float(max(ln_eps, 0.0)), int(g), int(SMT_NW),
            int(ch or SMT_CH or _chunk(B * n * n, SMT_NW / 4, dev)),
        )  # fmt: skip
        return out, codes, made
    made = ext().forward_b200(
        tw, bb, ca, cb, mask, idx, chains, dlim, alim, out, codes, int(ln_eps >= 0), float(max(ln_eps, 0.0)), int(g), cps,
        int(ch or _chunk(B * n * n, cps, dev)),
    )  # fmt: skip
    return out, codes, made


#: a training step passes the module's weight to the forward launch, which makes the table (pg120_table, PDL behind it) instead of
#: ``tbl.copy_(weight.t())`` (a non-PDL elementwise kernel, 2.8 - 3.8 us)
TBL_IN_LAUNCH = True
#: (rejected 2026-11-17: the decoder forward making the table in every CTA's shared memory from the fp32 weight -- one launch less -- ran
#: 25.5 us at S1 L128 against 4.6 us + 2.3 us for pg120_table and this forward: 46.9 vs 30.5 us a training step)


def weight_launch(weight: torch.Tensor) -> bool:
    """Whether a training step passes the weight [256, rows] to the forward launch (:data:`TBL_IN_LAUNCH`) instead of a table."""
    return TBL_IN_LAUNCH and weight.dtype in (torch.float32, torch.bfloat16) and weight.is_contiguous()


def forward_ca_b200(tbl, ca, mask, idx, dlim, ln_eps, train, cps=None):
    """(decoder pair [B, n, n, 256] bf16 -- its LayerNorm when ``ln_eps`` >= 0 --, codes, empty unless ``train``, the bf16 table the launch made,
    else empty) from the bf16 table [157, 256], or from the weight [256, 157] (the launch makes the table: :func:`weight_launch`)."""
    B, n = idx.shape
    dev = ca.device
    out = torch.empty(B, n, n, 256, device=dev, dtype=torch.bfloat16)
    codes = torch.empty(B, n, n, 4, device=dev, dtype=torch.int32) if train else _none(dev)
    made = ext().forward_ca_b200(
        tbl, ca, mask, idx, dlim, out, codes, int(ln_eps >= 0), float(max(ln_eps, 0.0)), int(cps or CA_CPS)
    )
    return out, codes, (made if tbl.shape[0] == 256 else torch.empty(0, device=dev, dtype=torch.bfloat16))


def ln_backward_b200(tbl, codes, dh, mode, eps, g=None, cps=None, ch=None):
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) = ``dh`` (``tbl``: :func:`table`)."""
    dx = torch.empty_like(dh)
    cps = int(cps or LN_CPS)
    pairs = dh.shape[0] * dh.shape[1] * dh.shape[2]
    g = g or (LN_G_SMALL if pairs <= GEOM_BIG else LN_G)
    ext().ln_backward_b200(tbl, codes, dh, dx, int(mode), float(eps), int(g), cps, int(ch or _chunk(pairs, cps, dh.device)))
    return dx


def backward_w(g, codes, mode, nblk=None):
    """The weight gradient [256, N_ROWS[mode]] fp32 (the weight's layout; entries rounded to bf16 as the bf16 table gradient) from d pair = g
    [B, n, n, 256] bf16: the tcgen05 table gradient + its reduce (deterministic, fixed-order partial sums)."""
    nt = _h100.N_ROWS[mode]
    if nblk is None:
        pairs = g.shape[0] * g.shape[1] * g.shape[2]
        nblk = next((nb for most, nb in TC_NBLK if pairs <= most), NBLK_TC)
    tpart = torch.empty(nblk * nt * 256, device=g.device, dtype=torch.float32)
    dw = torch.empty(256, nt, device=g.device, dtype=torch.float32)
    ext().backward_w_b200(g, codes, tpart, dw, int(mode), int(nblk), int(RED_V), _bar(g.device))
    return dw
