"""CUDA implementations for Transition forward kernels."""

import torch


def transition_sm90a(x, ln_weight, ln_bias, wa, wb, ws, eps, n):
    """The sm_90a (H100) Transition dispatch that ``modules.Transition`` and ``ops.transition`` share. Returns ``x + transition(x)``
    from the first hand-CUDA build that takes the call, or ``None`` when none does (the caller, Transition's ``cuda`` backend, then raises
    ``BackendUnsupported``: there is no fallback).

    Order: the generic kernels where they are the faster pick even though a tuned build would take the call
    (``fused_generic_sm90a.prefers_generic``), then ``fused_sm90a`` (d=128), ``fused_wide_sm90a`` (d=64/256/384/512/768) and
    ``fused_generic_sm90a`` (any other width, ragged rows, fp32 on TF32). Each ``available`` is the kernel's own requirement plus a
    successful build. Only the ``cuda`` backend calls this; the ``triton`` backend runs the Triton kernels."""
    from . import fused_generic_sm90a, fused_sm90a, fused_wide_sm90a

    training = torch.is_grad_enabled() and any(t.requires_grad for t in (x, ln_weight, wa, wb, ws))
    if fused_wide_sm90a.f32_available(
        x, wa, ws, training
    ):  # fp32: the f32io family (fused forwards, fp16 MMA at TF32 rounding; TF32 backward)
        return fused_wide_sm90a.transition_f32_sm90a(x, ln_weight, ln_bias, wa, wb, ws, eps)
    if fused_generic_sm90a.prefers_generic(x, training, n) and fused_generic_sm90a.available(x, wa, ws):
        return fused_generic_sm90a.transition_generic_sm90a(x, ln_weight, ln_bias, wa, wb, ws, eps)
    for kern, entry in (
        (fused_sm90a, fused_sm90a.transition_fused_sm90a),
        (fused_wide_sm90a, fused_wide_sm90a.transition_wide_sm90a),
        (fused_generic_sm90a, fused_generic_sm90a.transition_generic_sm90a),
    ):
        if kern.available(x, wa, ws):
            return entry(x, ln_weight, ln_bias, wa, wb, ws, eps)
    return None


def transition_sm80(x, ln_weight, ln_bias, wa, wb, ws, eps):
    """The sm_80 (A100) Transition dispatch of ``modules.Transition``'s ``cuda`` backend: ``x + transition(x)`` from ``fused_sm80`` (D = 128 /
    hidden 512, whole 256-row tiles, at least ``fused_wide_sm80.FUSED_MIN_ROWS`` rows: one forward kernel, a two-kernel backward) or else
    ``fused_wide_sm80`` (D in ``fused_wide_sm80.WIDTHS``, hidden a multiple of 64, any row count), or ``None`` when neither takes the call
    (the caller refuses: there is no fallback; a build failure raises ``_nvcc.KernelBuildError``). bf16 activations; the parameters
    bf16 or fp32 masters (cast inside, outside autograd, so a master's gradient is fp32). fp32 activations (TF32 allowed): ``fused_wide_sm80.transition_f32_sm80`` (D in ``F32_WIDTHS``, hidden a multiple of 64)."""
    from . import fused_fwd_sm80, fused_sm80, fused_wide_sm80

    rows = x.numel() // x.shape[-1] if x.dim() else 0
    limit = fused_fwd_sm80.max_rows(x.device, x.dtype)
    grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, ln_weight, ln_bias, wa, wb, ws))
    if rows > limit and not grad:
        # inference past one call's rows (the kernels' work lists / row grid): one call per contiguous row block (rows are independent), joined.
        # Training past it is refused by modules.Transition (the backward would need the same split of its weight-gradient sums).
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        parts = [transition_sm80(flat[r : r + limit], ln_weight, ln_bias, wa, wb, ws, eps) for r in range(0, rows, limit)]
        if any(p is None for p in parts):
            return None
        return torch.cat(parts).reshape(x.shape)
    if x.dtype is torch.float32:  # fp32 (TF32 allowed): the wide fp32 path (fp16 products at TF32's rounding, a TF32 backward)
        return (
            fused_wide_sm80.transition_f32_sm80(x, ln_weight, ln_bias, wa, wb, ws, eps)
            if fused_wide_sm80.f32_available(x, wa, ws)
            else None
        )
    route = fused_wide_sm80.route(x, wa, ws)
    if route == "fused":
        return fused_sm80.transition_fused_sm80(x, ln_weight, ln_bias, wa, wb, ws, eps)
    if route == "wide":
        return fused_wide_sm80.transition_wide_sm80(x, ln_weight, ln_bias, wa, wb, ws, eps)
    return None


def _table_configs(gpu, op, keep):
    """Every config the launch-config table of ``gpu`` holds for ``op`` (all ranks: ``tune`` times them), filtered by
    ``keep(dims, kwargs)`` -- what ``_launch_config.choose`` may return besides a kernel's default."""
    from miniworld_engine.kernels._launch_config import _parse, table

    out = []
    for key, cfgs in table("transition", gpu).get(op, {}).items():
        _, dims, _ = _parse(key)
        for c in cfgs:
            if keep(dict(dims), c["kwargs"]) and c["kwargs"] not in out:
                out.append(dict(c["kwargs"]))
    return out


def _shapes(rows):
    """``{(d, n): {(dtype, mode): row counts}}`` of the requirements rows: the channel width ``d`` and hidden multiple ``n`` of the
    Transition each row's module builds (``Transition``: d_hidden / n; ``PairformerBlock``: its pair transition, d_pair, n = 4;
    the token DiT classes: d_single / n), and the row counts (M = L * L on the pair stream, L otherwise) it runs at."""
    from miniworld_engine.kernels._prebuild import _kv

    out = {}
    for r in rows:
        dims = _kv(r["dims"])
        d = dims.get("d_hidden") or dims.get("d_single") or dims.get("d_pair")
        if d is None:  # a row of a module that builds no hand-CUDA Transition (atom blocks: d_atom)
            continue
        d = int(d)
        n = int(dims.get("n", 4))
        counts = out.setdefault((d, n), {}).setdefault((r["dtype"], r["mode"]), set())
        counts.update(L * L if r["stream"] == "token_pair" else L for L in r["lengths"])
    return out


def prebuild(target, rows):
    """Every build of the Transition family on ``target`` for the requirements ``rows`` (kernels/_prebuild.py). From the rows: the
    channel widths and hidden multiples (d, n) each build is specialized for; which dtypes reach it (the fused / wide builds take
    bf16 activations, i.e. the ``bfloat16`` and ``master`` rows; the generic build also takes the ``float32`` rows); eval -> the
    inference forward (no saved tensors), train -> the saving forward, the backward and, where a width keeps a / b, the saved-a/b
    build at the row counts the rows run. Family knowledge: the launch-config table entries for ``target.gpu`` and each kernel's
    default (and the first of its grid that a row count off the default's tile leaves), ``target.sms``, the fixed schedules."""
    import functools as ft

    shapes = _shapes(rows)
    half = {k: {m for (dt, m) in v if dt in ("bfloat16", "master")} for k, v in shapes.items()}  # modes served on bf16 activations
    if target.arch == "sm90":
        from . import fused_generic_sm90a, fused_sm90a, fused_wide_sm90a

        fused = {n: half[(128, n)] for n in fused_sm90a.NS if (128, n) in half}
        wide = {(d, n): half[(d, n)] for d, ns in fused_wide_sm90a.NS.items() for n in ns if (d, n) in half}
        if any(fused.values()) or any(wide.values()):
            yield fused_sm90a._native_ext
        # D = 128 fused: (ctas, dw_repl, save, n) of every schedule choose() can return; eval runs the non-saving forward,
        # train the saving forward and the backward
        builds = set()
        for n, modes in fused.items():
            for mode in modes:
                for backward in (False, True) if mode == "train" else (False,):
                    grid = fused_sm90a.transition_candidates(target.sms, backward=backward, n=n)
                    op = "transition_bwd_residual_sm90_cuda" if backward else "transition_fwd_residual_sm90_cuda"
                    configs = [
                        grid[0],
                        *_table_configs(
                            target.gpu,
                            op,
                            lambda dims, c, n=n, g=grid: dims.get("SMS") == target.sms and dims.get("ND") == 128 * n and c in g,
                        ),
                    ]
                    for c in configs:
                        builds.add((c["ctas"], c["dw_repl"], mode == "train", n))
                builds.add((target.sms, fused_sm90a.DW_REPL, mode == "train", n))  # _ext_for: the fixed DW_REPL schedule
        for args in sorted(builds):
            yield ft.partial(fused_sm90a._build_ext, *args)
        # the wide widths: every build _ext_for picks (the row count only selects a saved-a/b D512 cell)
        exts = {}
        for (d, n), modes in wide.items():
            for mode in modes:
                counts = set().union(*(c for (dt, m), c in shapes[(d, n)].items() if m == mode and dt != "float32"))
                for save_ab in (False, True) if mode == "train" else (False,):
                    for m in sorted(counts):
                        if save_ab and not fused_wide_sm90a.save_ab_enabled(d, m):
                            continue  # the saving build is requested only where save_ab_enabled (the training forward / backward)
                        args, kwargs = fused_wide_sm90a._ext_args(d, m, save_ab, n, target.sms)
                        exts[(args, tuple(sorted(kwargs.items())))] = (args, kwargs)
        for args, kwargs in exts.values():
            yield ft.partial(fused_wide_sm90a._ext, *args, **kwargs)

        # the generic build: float32 (TF32) rows, shapes the fused / wide builds do not take, and the row counts
        # ``prefers_generic`` hands it
        def generic(d, n, dtype, mode, counts):
            served = (d == 128 and n in fused_sm90a.NS) or n in fused_wide_sm90a.NS.get(d, ())
            small = fused_generic_sm90a._PREFER_GENERIC_ROWS.get(d, (0, 0))[mode == "train"]
            return dtype == "float32" or not served or any(m <= small for m in counts)

        if any(generic(d, n, dt, m, c) for (d, n), v in shapes.items() for (dt, m), c in v.items()):
            yield fused_generic_sm90a._ext

        # the fp32 forwards (one build per (D, n): every variant, the inference and the saving forward, the weight forms): the float32 eval
        # rows at F32_SHAPES, the float32 train rows at F32_TRAIN_SHAPES (their backward is the generic build's, planned above)
        def f32(d, n, v):
            return (("float32", "eval") in v and n in fused_wide_sm90a.F32_SHAPES.get(d, ())) or (
                ("float32", "train") in v and n in fused_wide_sm90a.F32_TRAIN_SHAPES.get(d, ())
            )

        for d, n in sorted(k for k, v in shapes.items() if f32(*k, v)):
            yield ft.partial(fused_wide_sm90a._f32_ext, d, n)
    elif target.arch == "sm100":
        from . import fused_sm100a, fused_wide_sm100a

        if any(half.values()):
            yield fused_sm100a._cubins
            yield fused_sm100a._ext
        for d, n in fused_wide_sm100a.SHAPES:
            if half.get((d, n)):
                yield ft.partial(fused_sm100a.build_cubins, f"d{d}n{n}", fused_wide_sm100a._specs(d, n))
    elif target.arch == "sm80":
        # A100 (transition_sm80): fused_sm80 at D 128 / n 4 with bf16 weights, the wide extension at every other width (and as fused_sm80's small-row path), its
        # one-kernel forward / backward builds (fused_fwd_sm80 / fused_bwd_sm80: (D, H) in their SHAPES) and the fp32-master cast; bf16 rows, and the float32 rows on the wide extension
        from . import fused_bwd_sm80, fused_fwd_sm80, fused_sm80, fused_wide_sm80, parameter_cast_sm80

        served = {k: v for k, v in half.items() if v and k[0] in fused_wide_sm80.WIDTHS}
        # the float32 rows (TF32): the wide extension's fp32 bindings (transition_f32_sm80) at D in F32_WIDTHS
        f32 = any(dt == "float32" for (d, _n), v in shapes.items() if d in fused_wide_sm80.F32_WIDTHS for (dt, _m) in v)
        if any(dt == "bfloat16" for (dt, _m) in shapes.get((128, 4), {})):  # fused_sm80 takes bf16 weights only (no master row reaches it)
            yield fused_sm80._ext
        if served or f32:
            yield fused_wide_sm80._ext_sm80
            yield parameter_cast_sm80._ext_sm80
        if any((d, d * n) in fused_fwd_sm80.SHAPES for d, n in served):
            yield fused_fwd_sm80._ext_sm80
        if any((d, d * n) in fused_fwd_sm80.SHAPES and "train" in modes for (d, n), modes in served.items()):
            yield fused_bwd_sm80._ext_sm80
    elif target.arch == "sm120":
        # GeForce RTX 5090 (transition_sm80 on sm_120a): the wide extension and the fp32-master cast built for sm_120a; fused_sm80 needs more shared
        # memory than its 99 KB a block and is not served there (kernels/_sm120_builds.py); the one-kernel forward (fused_fwd_sm80) and backward
        # (fused_bwd_sm80) are, with their sm_120 tiles (a shallower weight ring; 32-row PW stages)
        from . import fused_bwd_sm80, fused_fwd_sm80, fused_wide_sm80, parameter_cast_sm80

        served = {k: v for k, v in half.items() if v and k[0] in fused_wide_sm80.WIDTHS}
        f32 = any(dt == "float32" for (d, _n), v in shapes.items() if d in fused_wide_sm80.F32_WIDTHS for (dt, _m) in v)
        if served or f32:
            yield fused_wide_sm80._ext_sm120
            yield parameter_cast_sm80._ext_sm120
        if any((d, d * n) in fused_fwd_sm80.SHAPES for d, n in served) or any((d, d * n) in fused_fwd_sm80.F32_SHAPES for (d, n), v in shapes.items() for (dt, _m) in v if dt == "float32"):
            yield fused_fwd_sm80._ext_sm120
        if any((d, d * n) in fused_fwd_sm80.SHAPES and "train" in modes for (d, n), modes in served.items()):
            yield fused_bwd_sm80._ext_sm120
