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

    from foldjet_engine.kernels._prebuild import _kv, family_rows
    from foldjet_engine.kernels.integrations import opm_train, pwa_train, pwa_wide

    if target.arch == "sm100":  # B200: opm_sm100 / chai_sm100 (opm_*sm100.cu), pwa_sm100 (pwa_*sm100.cu)
        from foldjet_engine.kernels.integrations import opm_sm100, pwa_sm100

        yield functools.partial(opm_train._ext, arch="sm100")
        yield opm_sm100._ext_wide
        yield from pwa_sm100.builds()
        from foldjet_engine.kernels.integrations import opm_fp8_b200  # the opt-in e4m3 routes (docs/b200/fp8.md): the cast / e4m3 helper extension

        yield from opm_fp8_b200.prebuild_items()
        yield from _prebuild_mpnn()  # ProteinMPNN: the sm_100a builds of the sm_80 sources (<name>_sm100, kernels/_sm120.py building_sm100)
        return
    if target.arch == "sm80":
        yield from _prebuild_sm80()
        return
    if target.arch == "sm120":  # RTX 5090: the sm_120a builds of the sm_80 OPM / PWA sources (kernels/_msa_card.py)
        from foldjet_engine.kernels import _msa_card

        yield from _msa_card.prebuild_sm120()
        yield from _prebuild_mpnn()
        return
    if target.arch != "sm90":
        return
    ext = functools.partial(opm_train._ext, None, arch="sm90")
    opm: dict[tuple, dict] = {}

    def add(**kw):
        opm[tuple(sorted(kw.items()))] = kw

    base = (opm_train.CM, opm_train.CZ)
    opm_f8 = False
    for r in family_rows("outer_product_mean"):
        d, o = _kv(r["dims"]), _kv(r["options"])
        cm, cz = int(d["d_msa"]), int(d["d_pair"])
        if int(d["d_hidden"]) != opm_train.CH or "groups" in o or ((cm, cz) != base and (cm, cz) not in opm_train.SHAPES):
            continue
        if r["mode"] == "eval" and (cm, cz) in opm_train.TRAIN_ONLY:  # inference at these widths is composed.inference's
            continue
        # _OpmMath.forward runs _forward_wide for the other widths, the divide-after-projection order (normalize_before_proj False) and
        # projection biases
        wide = (cm, cz) != base or o.get("normalize_before_proj", "True") == "False" or o.get("projection_bias") == "True"
        training = r["mode"] == "train"
        if not training and (cm, cz) == base:  # inference past depth 4096 (beyond the rows' depths): the product route's epilogue build
            add(native=True, tma=False, cm=cm, cz=cz, fused=False)
        if r["dtype"] != "float32":  # the opt-in FP8 routes (docs/fp8.md, opm_fp8: inference; docs/fp8_training.md, opm_fp8_train: training at the widths of opm_train.SHAPES): the e4m3 prologue build
            add(native=True, tma=False, cm=cm, cz=cz, fp8=True)
            opm_f8 = True
        for s in r["depths"]:
            for n in r["lengths"]:
                path = opm_train._h100_path(s, n, training=training)
                if wide or path == "legacy":  # _OpmMath.forward -> _forward_wide (the legacy path runs the CUDA prologue)
                    add(native=True, tma=False, cm=cm, cz=cz, fused=opm_train._fused_enabled(s, n, training=training, cm=cm, cz=cz))
                else:
                    fused = opm_train._fused_enabled(s, n, training=training)
                    add(
                        native=True,
                        tma=path == "tma",
                        fused=fused,
                        epi_tma=not fused and opm_train._epilogue_tma_enabled(s, n, training=training),
                        epi_half=not fused and opm_train._epilogue_half_enabled(s, n, training=training),
                    )
                if training:  # _OpmMath.backward
                    if wide:
                        add(native=True, tma=False, cm=cm, cz=cz)
                    else:
                        add(
                            native=path != "legacy",
                            tma=path == "tma",
                            epi_tma=opm_train._epilogue_tma_enabled(s, n, training=True),
                            epi_half=opm_train._epilogue_half_enabled(s, n, training=True),
                        )
    for kw in opm.values():
        yield functools.partial(ext, **kw)
    if any(r["mode"] == "eval" for r in family_rows("outer_product_mean")):  # the product route's pair-count GEMM (opm_train._count_table)
        from foldjet_engine.kernels.conditioned_transition.cuda import train as _ct_train

        yield _ct_train.owned_gemm_ext
        yield opm_train._lt_ext  # the product route's cuBLASLt heuristic-index product (opm_train._LT_ALGO)
    if opm_f8:  # the opt-in FP8 route (docs/fp8.md): the cuBLASLt e4m3 product (opm_fp8.product)
        from foldjet_engine.kernels.integrations import opm_fp8

        yield opm_fp8._lt_ext

    base = (pwa_train.D, pwa_train.DZ, pwa_train.H, pwa_train.C)
    from foldjet_engine.kernels.integrations import pwa_fp8

    natives, pair, value, f8 = set(), set(), set(), set()
    f8_train = False
    for r in family_rows("pair_weighted_averaging"):
        d = _kv(r["dims"])
        shape = (int(d["d_msa"]), int(d["d_pair"]), int(d["n_head"]), int(d["d_hidden"]))
        if shape == base or pwa_train.pair_only_wide(*shape):  # pair_only_wide: these MSA kernels + pwa_wide's pair kernels
            if shape != base:
                pair.add(shape[1])
            if r["mode"] == "eval":
                natives.add(True)  # inference runs the native kernels on H100
                f8.add((pwa_train.D, pwa_train.C))  # the opt-in FP8 route (docs/fp8.md, pwa_fp8.forward): ln_vt8 at the 8 x 32 width
            else:
                natives.update(pwa_train._training_default_shape(s, n) for s in r["depths"] for n in r["lengths"])
                f8_train = True  # the opt-in FP8 training route (docs/fp8_training.md, pwa_fp8_train: 8 x 32 widths, d_msa 64)
        elif pwa_wide.supported(*shape):
            pair.add(shape[1])
            value.add((shape[0], shape[3]))
            if r["mode"] == "eval" and (shape[0], shape[3]) in pwa_fp8.WIDE:  # the opt-in FP8 route (docs/fp8.md, pwa_fp8.forward_wide)
                f8.add((shape[0], shape[3]))
    for native in sorted(natives):
        yield functools.partial(pwa_train._k, native)
    for dz in sorted(pair):
        yield functools.partial(pwa_wide._pair, dz)
    for dm, dh in sorted(value):
        yield functools.partial(pwa_wide._value, dm, dh)
    if value:  # the backward aux kernels of the wide widths (dropout-mask pass, BIG dv permute-copy): one shape-independent build
        yield pwa_wide._aux
    for dm, c in sorted(f8):
        yield functools.partial(pwa_fp8._ext, dm, c)
    if f8_train:  # the opt-in FP8 training route: ln_vt8 natural-layout builds (default / bf16 v), the e4m3 glue and the saved-o quantizer
        from foldjet_engine.kernels.integrations import pwa_fp8_train

        yield pwa_fp8_train._vt8_nat
        yield pwa_fp8_train._vt8_nat_bf
        yield pwa_fp8_train._glue8
        yield pwa_fp8_train._o8_ext


def _prebuild_sm80():
    """A100: the extensions ``opm_sm80`` / ``pwa_sm80`` reach (their default selections), when a bf16 (master) OuterProductMean /
    MSAPairWeightedAveraging row is in their range (``kernels/outer_product_mean/cuda/sm80.supported``, ``.../pair_weighted_averaging/cuda/sm80.supported``).
    Each is one shape-independent extension (the sizes are launch parameters)."""
    from foldjet_engine.kernels._prebuild import _kv, family_rows
    from foldjet_engine.kernels.outer_product_mean.cuda import mask_count_ballot_sm80, mask_epilogue_sm80, pipeline_native128_sm80
    from foldjet_engine.kernels.outer_product_mean.cuda import sm80 as opm
    from foldjet_engine.kernels.pair_weighted_averaging.cuda import (
        broadcast_native_sm80,
        contract_gate_fused_sm80,
        dw_native_sm80,
        glue_tuned_sm80,
        grad_reduce_fused_sm80,
        grad_reduce_vector_sm80,
        pair_partial_sm80,
    )
    from foldjet_engine.kernels.pair_weighted_averaging.cuda import sm80 as pwa

    def plain(o):
        return "groups" not in o  # projection biases: the sm_80 prologue adds them; count_offset: the epilogues' divisor (FoldJet R2)

    opm_rows = [
        r
        for r in family_rows("outer_product_mean")
        if r["dtype"] in ("bfloat16", "master")
        and plain(_kv(r["options"]))
        and opm.supported(int(_kv(r["dims"])["d_msa"]), int(_kv(r["dims"])["d_hidden"]), int(_kv(r["dims"])["d_pair"]))
    ]
    if opm_rows:
        yield opm.ext
        from foldjet_engine.kernels.outer_product_mean.cuda import ogemm_sm80

        yield ogemm_sm80.ext  # the masked forward product (dead tokens skipped)
        yield mask_count_ballot_sm80._ext
        yield mask_epilogue_sm80._ext
        if any(r["mode"] == "train" for r in opm_rows):
            yield pipeline_native128_sm80._ext
    pwa_rows = []
    for r in family_rows("pair_weighted_averaging"):
        d = _kv(r["dims"])
        if r["dtype"] in ("bfloat16", "master") and pwa.supported(
            int(d["d_msa"]), int(d["d_pair"]), int(d.get("n_head", 8)), int(d.get("d_hidden", int(d["d_msa"]) // int(d.get("n_head", 8))))
        ):
            pwa_rows.append(r)
    from foldjet_engine.kernels.integrations import pwa_wide_sm80

    def wide(r):
        d = _kv(r["dims"])
        return pwa_wide_sm80.supported(int(d["d_msa"]), int(d["d_pair"]), int(d.get("n_head", 8)), int(d.get("d_hidden", 0)))

    if any(r["dtype"] in ("bfloat16", "master") and wide(r) for r in family_rows("pair_weighted_averaging")):
        from foldjet_engine.kernels.pair_weighted_averaging.cuda import wide_sm80
        from foldjet_engine.kernels.triangle_attention.cuda import sm80_bidir

        yield wide_sm80.ext
        from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90_generic

        yield sm80_bidir.glue  # its row LayerNorm (forward and backward)
        yield h100_sm90_generic._lt  # the residual out-projection (host-only cublasLt glue)
    if pwa_rows:
        yield pwa.ext
        yield pwa._bmm_ext
        yield contract_gate_fused_sm80._ext
        if any(r["mode"] == "train" for r in pwa_rows):
            yield from (
                broadcast_native_sm80._ext,
                dw_native_sm80._ext,
                glue_tuned_sm80._ext,
                grad_reduce_fused_sm80._ext,
                grad_reduce_vector_sm80._ext,
                pair_partial_sm80._ext,
            )
    yield from _prebuild_mpnn()


def _prebuild_mpnn():
    """ProteinMPNN (its "cuda" backend: mpnn_msg_sm80 / mpnn_edge_sm80): the four shape-independent sm_80 extensions, for a bf16 (master) row;
    on the RTX 5090 target their sm_120a builds (``<name>_sm120``, kernels/_sm120.py), on the B200 target their sm_100a builds (``<name>_sm100``)."""
    from foldjet_engine.kernels._prebuild import family_rows

    if any(r["dtype"] in ("bfloat16", "master") for r in family_rows("mpnn_message")):
        from foldjet_engine.kernels.mpnn_edge_tail.cuda import sm80 as edge
        from foldjet_engine.kernels.mpnn_message.cuda import sm80 as message
        from foldjet_engine.kernels.mpnn_node_message.cuda import sm80 as node
        from foldjet_engine.kernels.mpnn_relative_position.cuda import sm80 as relpos

        yield from (message._ext, node._ext, edge.ext)
        if any(r["mode"] == "train" for r in family_rows("mpnn_relative_position")):
            yield relpos._ext
