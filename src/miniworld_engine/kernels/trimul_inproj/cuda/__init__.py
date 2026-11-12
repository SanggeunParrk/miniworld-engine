"""Packaged H100 TriMul runtime and selected CUDA sources."""

import functools
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

#: The module classes whose backends launch the family's kernels (requirements_kernel.csv ``modules``); a Transition row of the
#: family (``trimul_input_ln_residual_bwd`` serves Transition too) names no TriMul build.
_TRIMUL_CLASSES = ("TriangleMultiplication", "BidirectionalTriangleMultiplication", "PairformerBlock")

#: ``(width, contraction channels)`` the row-padded K1 / K3 (any L, inference) is built for: ``rowpad_serves`` -- D64 / D128 with the
#: channels of one direction or of both, D256 / D384 one direction.
_ROWPAD = {64: (64, 128), 128: (128, 256), 256: (256,), 384: (384,)}


def _call(fn, *args, **kwargs):
    """A planned call of ``fn``; device-cached loaders (``_h100_runtime.device_cache``) are called undecorated, since the
    cache key asks torch for the current CUDA device."""
    return functools.partial(getattr(fn, "__wrapped__", fn), *args, **kwargs)


def _bucket(n):
    """The 384 / 768 length bucket the selection tables key (``str(n) if str(n) in table else "384" if n <= 384 else "768"``)."""
    return 384 if n <= 384 else 768


@dataclass
class _Mode:
    """What a cell must serve in one mode: the lengths and the master-weight switches (``dtype == master``: MASTER_FP32=1; a
    ``bfloat16`` row: 0; ``float32`` rows build nothing -- no TriMul build exists for them)."""

    lengths: set = field(default_factory=set)
    masters: set = field(default_factory=set)


@dataclass
class _Cell:
    """One TriMul shape: pair width ``d`` and contraction channels ``ch`` of one direction (``bidir``: both directions, ``2 x
    d_hidden``; single direction: ``d_hidden``; d_hidden defaults to d_pair), with what the requirements ask of it per mode."""

    bidir: bool
    d: int
    ch: int
    modes: dict = field(default_factory=dict)
    kernels: set = field(default_factory=set)


def _cells(rows):
    """The (width, channels) cells of the TriMul classes in ``rows`` (``_prebuild.family_rows``), with per-mode lengths / master
    switches and the names of the family's kernels that need them."""
    from miniworld_engine.kernels._prebuild import _kv

    cells = {}
    for r in rows:
        if r["module"] not in _TRIMUL_CLASSES:
            continue
        dims = _kv(r["dims"])
        d = int(dims["d_pair"])
        bidir = r["module"] == "BidirectionalTriangleMultiplication"
        hidden = int(dims.get("d_hidden_tri_multi") or dims.get("d_hidden") or d)
        cell = cells.setdefault((bidir, d, 2 * hidden if bidir else hidden), _Cell(bidir, d, 2 * hidden if bidir else hidden))
        cell.kernels |= r["kernels"]
        if r["dtype"] == "float32":
            continue
        mode = cell.modes.setdefault(r["mode"], _Mode())
        mode.lengths |= set(r["lengths"])
        mode.masters.add(r["dtype"] == "master")
    return [c for c in cells.values() if c.modes]


def prebuild(target, rows):
    """Every native build of the TriMul family on ``target`` (kernels/_prebuild.py): calls of the family's own loaders over the
    variants its launchers can ask for. From ``rows`` (the requirements): the (width, channels) cells and whether each is served
    by one direction or both, its modes (eval: the inference builds; train: the training builds), its lengths (the
    length-specialized builds) and its parameter dtype (``master`` / ``bfloat16``: the ``MASTER_FP32`` switch). Family knowledge:
    the selection / config JSONs under ``h100_sources/``, the K1 / K3 tile table, the hard-wired per-width tiles, and the
    build-time switches no requirement names (token vs pair mask, weight layout, batched samples, preserved-native partial sums)."""
    cells = _cells(rows)
    if not cells:
        return
    if target.arch == "sm100":
        from . import b200_infer

        yield b200_infer._ext
        return
    if target.arch == "sm120":
        yield from _sm120(cells)
        return
    if target.arch == "sm80":
        yield from _sm80(cells)
        return
    if target.arch != "sm90":
        return
    yield from _h100(cells)


def _h100(cells):
    from . import (
        _h100_infer_kernel as K,
        _h100_pack as P,
        _h100_runtime as T,
        h100_b1,
        h100_b7,
        h100_b200style as BS,
        h100_b200style_aux,
        h100_b200style_d384_tmasave,
        h100_b200style_direct,
        h100_b200style_reorder,
        h100_b200style_scalar_tmasave,
        h100_b200style_ss,
        h100_b200style_ss_regtune,
        h100_b200style_ss_saved,
        h100_b200style_static,
        h100_b200style_tmasave,
        h100_d64_b200gate,
        h100_inference,
        h100_native,
        h100_output,
        h100_single_b7,
        h100_single_output,
        h100_uni_d64_b200gate,
        h100_uni_wide_inference as UWI,
        h100_uni_wide_training as UWT,
        h100_wide_inference as WI,
        h100_wide_training as WT,
    )

    def served(mode, bidir=None, d=None):
        """``(cell, its _Mode)`` of the cells that serve ``mode`` (of the given direction count / width)."""
        return [(c, c.modes[mode]) for c in cells if mode in c.modes and bidir in (None, c.bidir) and d in (None, c.d)]

    infer, train = served("eval"), served("train")

    # glue shared by every path
    yield _call(P._owned_blas_ext)
    yield _call(P._aux_unit)
    yield _call(h100_b200style_aux._unit)

    # packaged K1 / K3 units: one per (c_z, channels) of a cell the tile table has (the training fronts of D64 hidden 128 / D128
    # bidirectional also use them), + the D384 native-residual build of the inference
    pairs = {(c.d, c.ch) for c in cells}
    for cz, ch in sorted(pairs):
        if ("sm_90a", cz, ch, "b") in K.TILE_TABLE:
            yield _call(h100_inference._Kernels.unit, SimpleNamespace(_units={}, native_residual=False), cz, ch)
    if any(c.d == 384 and c.ch == 384 for c, _ in infer):
        yield _call(h100_inference._Kernels.unit, SimpleNamespace(_units={}, native_residual=True), 384, 384)

    # D128 bidirectional training (h100_training): output K3 with / without saves, B1 per length, B7 per length and mode
    for _, m in served("train", bidir=True, d=128):
        yield _call(h100_output.kernel, "k3", 3, 2, False, -1)
        yield _call(h100_output.kernel, "k3", 0, 0, False, -1)
        b1 = T.read_config("b1/configs.json")
        for n in sorted(m.lengths):
            cfg = b1[str(n) if str(n) in b1 else "384" if n <= 384 else "768"]
            for master in sorted(m.masters):
                for batched in (False, True):
                    defines = dict(cfg.get("defines") or {}, MASTER_FP32=int(master), **({"B1_BATCHED": 1} if batched else {}))
                    yield _call(h100_b1.build, cfg["count"], n, cfg["part"], tuple(sorted(defines.items())))
            for mode in range(512):  # Plan's mode 52 with the token-mask (128), row-major (64), master (256) bits
                if mode & ~(128 | 64 | 256) == 52 and bool(mode & 256) in m.masters:
                    yield _call(h100_b7.kernel_plan, n, 10, mode)

    # D128 single direction training (h100_single): K1 / K3 / B1 / B7 of each selected length bucket
    for _, m in served("train", bidir=False, d=128):
        sel = json.loads((T.SOURCES / "single_output/selection.json").read_text())
        for n in sorted(m.lengths):
            cfg = sel[str(n) if str(n) in sel else str(_bucket(n))]
            yield _call(h100_native.kernel, "k1", 128, tuple(cfg["k1"]), True, True, 128)
            yield _call(h100_single_output.k3_build, tuple(cfg["k3"]))
            for master in sorted(m.masters):
                yield functools.partial(h100_single_b7.Plan, {"master_weights": master}, None, None, None, None, None, **cfg["b7"])
        for master in sorted(m.masters):
            yield _call(h100_single_output.build, False, master)

    # D64 training: d_hidden 64 bidirectional (channels 128), single direction d_hidden 64 (channels 64), and single direction d_hidden
    # 128 (Protenix template module, channels 128: ``_hidden_twice``, served by the bidirectional D64 kernels)
    for c, m in train:
        if c.d != 64 or c.ch not in (64, 128):
            continue
        bidirectional_kernels = c.bidir or c.ch == 128
        mod = h100_d64_b200gate if bidirectional_kernels else h100_uni_d64_b200gate
        yield _call(mod._k3, mod.K3_TILE)
        for master in sorted(m.masters):
            for gatepipe in (False, True):  # gatepipe = L >= 256 (bidirectional); not (outgoing and L == 256) (single direction)
                if bidirectional_kernels:
                    yield _call(h100_d64_b200gate._bwd_wg, False, master, gatepipe)
                else:
                    yield _call(h100_uni_d64_b200gate._bwd_wg, master, False, gatepipe)

    # B200-style port (D256 / D384 / D512, one direction or both): the forward front serves both modes, the saving front / backward
    # the training, the inference outputs / the saving outputs follow their modes
    for c in cells:
        d, h = c.d, c.ch
        if d not in BS.WIDTHS:
            continue
        ev, tr = "eval" in c.modes, "train" in c.modes
        if d in (256, 384):
            yield _call(BS._front_kernel, d, h, False, BS._config(d))
            if tr:
                yield _call(BS._front_kernel, d, h, True, BS._config(d, True))
            yield _call(h100_b200style_direct._kernel, d, h, BS._config(d))
        else:
            if ev:
                yield _call(h100_b200style_ss_regtune._kernel, d, h, False, h100_b200style_ss_regtune._config(d, False))
            if tr:
                for backward in (False, True):
                    yield _call(h100_b200style_ss_saved._kernel, d, h, backward, h100_b200style_ss_saved._config(d, backward))
                yield _call(h100_b200style_ss._kernel, d, h, True, h100_b200style_ss._config(d, True))
        cfg = BS._output_config(d)
        if ev:
            yield _call(h100_b200style_static.output_kernel, d, h, cfg)
            yield _call(BS._output_kernel_standard, d, h, cfg)
            if d == 512:
                yield _call(h100_b200style_reorder.output_kernel, d, h, cfg)
        if tr:
            save = {512: h100_b200style_tmasave, 384: h100_b200style_d384_tmasave, 256: h100_b200style_scalar_tmasave}[d]
            yield _call(save.output_kernel, d, h, cfg)

    # row-padded K1 / K3 (inference, any L): per length of the cell
    for c, m in infer:
        if c.ch in _ROWPAD.get(c.d, ()):
            for n in sorted(m.lengths):
                yield _call(h100_b200style_direct._kernel, c.d, c.ch, BS._config(c.d, n=n))
            yield _call(h100_b200style_static.output_kernel, c.d, c.ch, BS._output_config(c.d), True, BS.K3_XBUF.get(c.d, 1))

    # wide inference: bidirectional (wide_forward/selection.json, with / without a mask) and single direction D512
    wide = json.loads((T.SOURCES / "wide_forward/selection.json").read_text())
    for key, s in wide.items():
        D, bucket = map(int, key.split("-"))
        if any(c.bidir and c.d == D and any(_bucket(n) == bucket for n in m.lengths) for c, m in infer):
            gfold = s.get("gfold", False)
            for has_mask in (True, False):
                yield _call(WI._compile, "k1", D, tuple(s["k1"]), s["normalize"], gfold and s["normalize"], has_mask)
            yield _call(WI._compile, "k3", D, (*s["k3s"], gfold))
    uni = json.loads((T.SOURCES / "uni_wide/selection.json").read_text())
    for key, s in uni.items():
        D, bucket = map(int, key.split("-"))
        if any(not c.bidir and c.d == D and any(_bucket(n) == bucket for n in m.lengths) for c, m in infer):
            yield _call(UWI._compile, "k1", D, D, tuple(s["k1"]), s["normalize"])
            yield _call(UWI._compile, "k3", D, D, tuple(s["k3"]))

    # wide training: bidirectional (h100_wide_training) and single direction / native buckets (h100_uni_wide_training)
    units = []
    for c, m in train:
        for n in sorted(m.lengths):
            if c.bidir and WT.supports(c.d, n):
                units += WT.prebuild_units(c.d, n, sorted(m.masters))
            units += UWT.prebuild_units(c.d, n, c.bidir)
    for source, include, defines, pool in dict.fromkeys(units):
        yield _call(WT._unit, source, include, defines, pool)


def _sm80(cells):
    """The A100 builds (``kernels/integrations/trimul_sm80``; bf16 activations only, ``_cells`` already drops float32 rows): the fused D128 set
    (``sm80``: D128 with 128 hidden channels per direction), the wide set (``sm80_wide``: every other width, and D512 inference) and the D512
    training rows (``sm80_wide512``)."""
    from . import sm80, sm80_wide, sm80_wide512

    calls = {}
    for c in cells:
        if c.d == 128 and c.ch == (256 if c.bidir else 128):
            calls[sm80._ext] = None
            continue
        calls[sm80_wide._ext] = None
        if c.d == 512 and "train" in c.modes:
            calls[sm80_wide512._ext] = None
    yield from calls


def _sm120(cells):
    """The RTX 5090 builds: the sm_80 wide sources built for sm_120a (``sm80_wide``: every width, D128 included -- the fused D128 set is
    A100-only, its tiles need 164 KB of shared memory -- and ``sm80_wide512`` for the D512 training rows)."""
    from . import contract_sm80, sm80_wide, sm80_wide512

    calls = {sm80_wide._ext120: None, contract_sm80._ext120: None}  # + the own plane GEMM (the contraction, forward and backward)
    for c in cells:
        if c.d == 512 and "train" in c.modes:
            calls[sm80_wide512._ext120] = None
    yield from calls
