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
    them (``opm_train.TRAIN_ONLY`` widths: train rows only). PWA: ``_k(native)`` for the base width by the training-default shape (train) /
    inference (eval) -- with the ``pwa_wide`` pair kernel of its d_pair for ``pwa_train.pair_only_wide`` rows --, ``pwa_wide`` pair / value
    kernels for the rows ``pwa_wide.supported`` accepts (the BIG value build included). d_hidden 64 / grouped OPM rows are the Chai kernels of ``outer_product_mean``."""
    import functools

    from miniworld_engine.kernels._prebuild import _kv, family_rows
    from miniworld_engine.kernels.integrations import opm_train, pwa_train, pwa_wide

    if target.arch == "sm100":
        yield functools.partial(opm_train._ext, arch="sm100")
        yield pwa_train._k100
        return
    if target.arch == "sm80":
        yield from _prebuild_sm80()
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
        if r["mode"] == "eval" and (cm, cz) in opm_train.TRAIN_ONLY:   # inference at these widths is composed.inference's
            continue
        # _OpmMath.forward runs _forward_wide for the other widths, the divide-after-projection order (normalize_before_proj False) and
        # projection biases
        wide = (cm, cz) != base or o.get("normalize_before_proj", "True") == "False" or o.get("projection_bias") == "True"
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
        if shape == base or pwa_train.pair_only_wide(*shape):   # pair_only_wide: these MSA kernels + pwa_wide's pair kernels
            if shape != base:
                pair.add(shape[1])
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


def _prebuild_sm80():
    """A100: the extensions ``opm_sm80`` / ``pwa_sm80`` reach (their default selections), when a bf16 (master) OuterProductMean /
    MSAPairWeightedAveraging row is in their range (``kernels/outer_product_mean/cuda/sm80.supported``, ``.../pair_weighted_averaging/cuda/sm80.supported``).
    Each is one shape-independent extension (the sizes are launch parameters)."""
    from miniworld_engine.kernels._prebuild import _kv, family_rows
    from miniworld_engine.kernels.outer_product_mean.cuda import mask_count_ballot_sm80, mask_epilogue_sm80, pipeline_native128_sm80
    from miniworld_engine.kernels.outer_product_mean.cuda import sm80 as opm
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
        broadcast_native_sm80,
        contract_gate_fused_sm80,
        dw_native_sm80,
        glue_tuned_sm80,
        grad_reduce_fused_sm80,
        grad_reduce_vector_sm80,
        pair_partial_sm80,
    )
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80 as pwa

    def plain(o):
        return not any(k in o for k in ("groups", "count_offset"))   # projection biases: the sm_80 prologue adds them

    opm_rows = [r for r in family_rows("outer_product_mean") if r["dtype"] in ("bfloat16", "master") and plain(_kv(r["options"]))
                and opm.supported(int(_kv(r["dims"])["d_msa"]), int(_kv(r["dims"])["d_hidden"]), int(_kv(r["dims"])["d_pair"]))]
    if opm_rows:
        yield opm.ext
        from miniworld_engine.kernels.outer_product_mean.cuda import ogemm_sm80

        yield ogemm_sm80.ext   # the masked forward product (dead tokens skipped)
        yield mask_count_ballot_sm80._ext
        yield mask_epilogue_sm80._ext
        if any(r["mode"] == "train" for r in opm_rows):
            yield pipeline_native128_sm80._ext
    pwa_rows = []
    for r in family_rows("pair_weighted_averaging"):
        d = _kv(r["dims"])
        if r["dtype"] in ("bfloat16", "master") and pwa.supported(int(d["d_msa"]), int(d["d_pair"]), int(d.get("n_head", 8)),
                                                                   int(d.get("d_hidden", int(d["d_msa"]) // int(d.get("n_head", 8))))):
            pwa_rows.append(r)
    if pwa_rows:
        yield pwa.ext
        yield pwa._bmm_ext
        yield contract_gate_fused_sm80._ext
        if any(r["mode"] == "train" for r in pwa_rows):
            yield from (broadcast_native_sm80._ext, dw_native_sm80._ext, glue_tuned_sm80._ext, grad_reduce_fused_sm80._ext,
                        grad_reduce_vector_sm80._ext, pair_partial_sm80._ext)
    # ProteinMPNN (its "cuda" backend: mpnn_msg_sm80 / mpnn_edge_sm80): the four shape-independent sm_80 extensions, for a bf16 (master) row
    if any(r["dtype"] in ("bfloat16", "master") for r in family_rows("mpnn_message")):
        from miniworld_engine.kernels.mpnn_edge_tail.cuda import sm80 as edge
        from miniworld_engine.kernels.mpnn_message.cuda import sm80 as message
        from miniworld_engine.kernels.mpnn_node_message.cuda import sm80 as node
        from miniworld_engine.kernels.mpnn_relative_position.cuda import sm80 as relpos

        yield from (message._ext, node._ext, edge.ext)
        if any(r["mode"] == "train" for r in family_rows("mpnn_relative_position")):
            yield relpos._ext
