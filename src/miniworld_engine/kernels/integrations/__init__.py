"""Adapters that run someone else's kernels behind this engine's module API.

Each adapter states what it loads, where it came from and what it refuses; none of them reimplement the upstream algorithm.
"""


def prebuild(target, _rows):
    """Every build of the adapters here for ``target`` (kernels/_prebuild.py), expanded from the requirements rows of the modules that
    launch them: the OuterProductMean rows (family ``outer_product_mean``: ``opm_train._ext``) and the MSAPairWeightedAveraging rows
    (family ``pair_weighted_averaging``: ``pwa_train._k``, ``pwa_wide``). This family has no kernel rows of its own (its kernels are
    listed under those two), so ``_rows`` is empty and the rows are taken from them here.

    OPM: for each row (width ``d_msa`` / ``d_pair``, the two normalisation orders, projection bias, mode) and each (depth, length) it
    covers, the extension the forward / backward of ``opm_train`` selects (the same predicates: ``_h100_path``, ``_fused_enabled``,
    ``_epilogue_*_enabled`` and the wide-composition routing); widths other than MiniWorld's only where ``opm_train.SHAPES`` serves
    them. PWA: ``_k(native)`` for the base width by the training-default shape (train) / inference (eval), ``pwa_wide`` pair / value
    kernels for the rows ``pwa_wide.supported`` accepts. d_hidden 64 / grouped OPM rows are the Chai kernels of ``outer_product_mean``."""
    import functools

    from miniworld_engine.kernels._prebuild import _kv, family_rows

    from . import opm_train, pwa_train, pwa_wide

    if target.arch == "sm100":
        yield functools.partial(opm_train._ext, arch="sm100")
        yield pwa_train._k100
        return
    if target.arch != "sm90":
        return
    ext = functools.partial(opm_train._ext, None, arch="sm90")
    opm: dict[tuple, dict] = {}

    def add(**kw):
        opm[tuple(sorted(kw.items()))] = kw

    base = (opm_train.CM, opm_train.CZ)
    for r in family_rows("outer_product_mean"):
        d, o = _kv(r["dims"]), _kv(r["options"])
        cm, cz = int(d["d_msa"]), int(d["d_pair"])
        if int(d["d_hidden"]) != opm_train.CH or "groups" in o or ((cm, cz) != base and (cm, cz) not in opm_train.SHAPES):
            continue
        wide = (cm, cz) != base or o.get("normalize_before_proj") == "True" or o.get("projection_bias") == "True"
        training = r["mode"] == "train"
        for s in r["depths"]:
            for n in r["lengths"]:
                path = opm_train._h100_path(s, n, training=training)
                if wide or path == "legacy":  # _OpmMath.forward -> _forward_wide (the legacy path runs the CUDA prologue)
                    add(native=True, tma=False, cm=cm, cz=cz, fused=opm_train._fused_enabled(s, n, training=training, cm=cm, cz=cz))
                else:
                    fused = opm_train._fused_enabled(s, n, training=training)
                    add(native=True, tma=path == "tma", fused=fused,
                        epi_tma=not fused and opm_train._epilogue_tma_enabled(s, n, training=training),
                        epi_half=not fused and opm_train._epilogue_half_enabled(s, n, training=training))
                if training:  # _OpmMath.backward
                    if wide:
                        add(native=True, tma=False, cm=cm, cz=cz)
                    else:
                        add(native=path != "legacy", tma=path == "tma", epi_tma=opm_train._epilogue_tma_enabled(s, n, training=True),
                            epi_half=opm_train._epilogue_half_enabled(s, n, training=True))
    for kw in opm.values():
        yield functools.partial(ext, **kw)

    base = (pwa_train.D, pwa_train.DZ, pwa_train.H, pwa_train.C)
    natives, pair, value = set(), set(), set()
    for r in family_rows("pair_weighted_averaging"):
        d = _kv(r["dims"])
        shape = (int(d["d_msa"]), int(d["d_pair"]), int(d["n_head"]), int(d["d_hidden"]))
        if shape == base:
            if r["mode"] == "eval":
                natives.add(True)  # inference runs the native kernels on H100
            else:
                natives.update(pwa_train._training_default_shape(s, n)
                               for s in r["depths"] for n in r["lengths"])
        elif pwa_wide.supported(*shape):
            pair.add(shape[1])
            value.add((shape[0], shape[3]))
    for native in sorted(natives):
        yield functools.partial(pwa_train._k, native)
    for dz in sorted(pair):
        yield functools.partial(pwa_wide._pair, dz)
    for dm, dh in sorted(value):
        yield functools.partial(pwa_wide._value, dm, dh)
