"""The B200 (sm_100a) builds of this family (``kernels/_prebuild.py``, ``prebuild(target, rows)`` with ``target.arch == "sm100"``), from the
requirements rows of the module classes whose ``"cuda"`` backend reaches them on a B200:

* ``AttentionPairBias`` (and ``PairformerBlock``'s single attention; ``integrations/attention_pair_bias_b200``): the row / pair extension
  (``apb._ext("100")``) and, per (heads, d_single) the B200 path takes (``SHAPES``, the pair widths of ``apb.PAIR_WIDTHS``), the
  ``augattn_sm100`` cores -- bf16 inference ``attn_inf -DQPAIR`` (``ApbInferenceCore``), bf16 training ``attn_fwd2 -DQPAIR`` + ``glue`` +
  ``attn_dkv`` + ``attn_dqb -DDQPART`` (``apb_forward`` / ``apb_backward``), fp32 inference ``attn_inf_tf32 -DKPERM`` (``_b200_tf32_core``)
  with the one-launch weight pack (``bo_pack_sm100``), fp32 training the TF32 forward / backward (``_b200_tf32_defs``);
* the token DiTs (``integrations/token_dit`` / ``token_dit_train``, ``conditioned_transition.cuda.token_inference``): per head layout of
  ``sm100.TDIT_HEADS`` the gated inference core (``GatedInferenceCore``: ``attn_inf`` / ``attn_inf_tf32``) and the training cores
  (``forward`` / ``backward``: ``attn_fwd2``, ``attn_dkv``, ``attn_dqb``, ``glue``; ``forward_tf32`` / ``backward_tf32``);
* the block-local atom DiTs: ``b200_atom`` / ``b200_atom_local`` (``integrations/local_dit_b200``: AtomDiTProtenix / AtomDiTOpenFold3 --
  every row kernel and local kernel ``_kernels_ready`` / ``_kernels32_ready`` load, the weight gradient ``atom_dw`` in training -- and
  Boltz-2 / RoseTTAFold3 / Chai-1 on the same block (round 3: ``local_dit_b200._run_variant``): those kernels plus the variant builds
  ``b200_atom.KERNELS_VAR`` / ``KERNELS_FUSED32_VAR``, ``var_pack``, ``conf_mask`` and, in training, ``kqln_bwd`` / ``var_rows_tf32``; the
  ``dit_variants`` B200 core ``train._b200_core`` (the calls the block refuses) reads the same local cubins). The sm_80 builds the B200
  reuses for the atom DiTs (``dit_variants``: the rows, edge, ``lattn90*`` and ``lbias`` cubins) are planned by ``dit_variants``.

No ``sm100_atom`` / ``sm100_atom_local`` build is planned for sm_100a: those packages keep the H100 ``-DATOM_HOPPER`` half (the sm90 plan);
the B200 runs their B200-only copies ``b200_atom`` / ``b200_atom_local``. The ``sm80_atom_rows`` / ``sm80_atom_local`` builds
(``local_dit_sm80_block``) are not planned either: every AtomDiTProtenix call the B200 serves takes the tcgen05 block."""

from __future__ import annotations

import functools as ft

#: the token DiT classes whose B200 cores are the augattn_sm100 ones (TokenDiTMiniWorldBiasOnly runs the bias-only family's)
TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTMiniWorld", "TokenDiTBoltz2", "TokenDiTRoseTTAFold3", "TokenDiTChai1"})
#: the atom classes on integrations/local_dit_b200 (the whole tcgen05 block)
LOCAL_TC = frozenset({"AtomDiTProtenix", "AtomDiTOpenFold3"})
#: the atom classes whose dit_variants path takes the tcgen05 window attention / pair bias (dit_variants/cuda/train._b200_core)
VARIANT_CORE = frozenset({"AtomDiTBoltz2", "AtomDiTRoseTTAFold3", "AtomDiTChai1"})
#: ESMFold2's atom block with the pair: inference and (round 3 part 7) training on integrations/local_dit_esm_b200
ESM_TC = "AtomDiTESMFold2Pair"


def _apb_geometry(r):
    """(heads, d_single, d_pair) of an AttentionPairBias row or of a PairformerBlock row's single attention."""
    from miniworld_engine.kernels._prebuild import _kv

    d = _kv(r["dims"])
    if r["module"] == "PairformerBlock":
        return int(d.get("n_head_single", 16)), int(d.get("d_single", 384)), int(d.get("d_pair", 128))
    return int(d["n_head"]), int(d["d_single"]), int(d.get("d_pair", 128))


def prebuild_sm100(target, rows):
    from miniworld_engine.kernels._prebuild import _kv
    from miniworld_engine.kernels.bias_only_dit.cuda import _dir as bo_dir
    from miniworld_engine.kernels.integrations import attention_pair_bias_b200 as apb_b200

    from . import apb, b200_atom, b200_atom_local, sm100

    calls: list = []

    def core(stem, defs=()):
        calls.append(ft.partial(sm100.cubin, stem, tuple(defs)))

    # ---- AttentionPairBias
    apb_rows = []
    pair_bwd: set = set()
    for r in rows:
        if r["module"] not in ("AttentionPairBias", "PairformerBlock"):
            continue
        H, D, dp = _apb_geometry(r)
        if (H, D) in apb_b200.SHAPES and dp in apb.PAIR_WIDTHS and H in apb.PAIR_WIDTHS[dp]:  # apb_b200.refusal
            apb_rows.append((H, D, r["dtype"], r["mode"]))
            if r["dtype"] != "float32" and dp != apb.DP:  # the tcgen05 pair -> bias pass of a wide bf16 pair (apb_b200._pair_tc)
                core("pair_fwd", (f"PF_NH={H}", f"PF_DPW={dp}"))
            if r["dtype"] == "float32" and r["mode"] == "eval" and dp % 64 == 0:  # apb_b200._pf32: the fp32 pass on tcgen05
                core("pair_fwd32", (f"PF_NH={H}", f"PF_DPW={dp}"))
            if r["dtype"] != "float32" and r["mode"] == "train" and dp == apb.DP and H not in pair_bwd:  # apb_b200._PAIR_BWD_TC
                pair_bwd.add(H)
                core("pair_bwd", sm100._pair_bwd_defs(H))
    if apb_rows:
        calls.append(ft.partial(apb._ext, "100", False))
    for H, D, dtype, mode in sorted(set(apb_rows)):
        if dtype != "float32":
            if mode == "eval":
                core("attn_inf", sm100._apb_defs(H, True, D))
                core("attn_inf", (*sm100._apb_defs(H, True, D), "KSPLIT=1"))  # ApbInferenceCore's key split (L >= 512)
                core("attn_inf", (*sm100._apb_defs(H, True, D), "PDLW=1"))  # after the fused front (_FRONT_PDL) / the GEMM (_INF_PDL)
                core("attn_inf", (*sm100._apb_defs(H, True, D), "KSPLIT=1", "PDLW=1"))  # the key split after the GEMM (apb_b200._INF_PDL)
            else:
                core("attn_fwd2", sm100._apb_defs(H, True, D))
                core("attn_fwd2", (*sm100._apb_defs(H, True, D), "PDLW=1"))  # L <= 256 (apb_b200._pdl_on)
                core("glue")
                core("attn_dkv", sm100._apb_defs(H, False, D))
                core("attn_dkv", (*sm100._apb_defs(H, False, D), "BIASNT=1"))  # the bias read as stored (apb_b200._bias_nt)
                core("attn_dkv", (*sm100._apb_defs(H, False, D), "PDLW=1"))  # L <= 256 (apb_b200._pdl_on)
                core("attn_dqb", (*sm100._apb_defs(H, False, D), "DQPART=1"))
            continue
        dh = D // H
        dhp = 48 if dh == 48 else 32  # apb_b200._geometry32
        if mode == "eval":
            core("attn_inf_tf32", (f"NHEAD={H}", f"DHP={dhp}", "KPERM=1"))
            calls.append(ft.partial(sm100.cubin, "bo_pack_sm100", (), src_dir=str(bo_dir)))  # _inference32_packs_b200
        else:
            defs = apb_b200._b200_tf32_defs(H, dh, dhp)
            for stem in ("attn_fwd_tf32", "attn_dkv_tf32"):
                core(stem, defs)
            # sm100.backward_tf32(dq_parts=True): the dQ slices per 64-key chunk
            core("attn_dqb_tf32", (*(x for x in defs if x != "DET=1"), "DQPART=1", "DET=1"))
    # ---- token DiT cores
    token = set()
    pack = False
    for r in rows:
        if r["module"] not in TOKEN:
            continue
        d = _kv(r["dims"])
        nh = int(d["n_head"])
        layout = (nh, int(d["d_single"]) // nh)
        if layout in sm100.TDIT_HEADS:
            token.add((layout, r["dtype"] == "float32", r["mode"]))
            if r["dtype"] != "float32" and r["mode"] == "eval" and r["module"] in ("TokenDiTProtenix", "TokenDiTMiniWorld"):
                pack = True
    if token and not apb_rows:
        # the token DiTs' pair bias is AttentionPairBias's one pass (bf16: apb.pair_bias / pair_bias_fwd_stats; fp32 from L 256: pair_bias32_tc)
        calls.append(ft.partial(apb._ext, "100", False))
    if pack:  # the default-option blocks' one-launch bf16 inference pack (conditioned_transition.cuda._b200_pack: bo_pack_sm100)
        calls.append(ft.partial(sm100.cubin, "bo_pack_sm100", (), src_dir=str(bo_dir)))
    for (nh, dh), f32, mode in sorted(token):
        defs = sm100._tdit_defs(nh, dh)
        if mode == "eval":
            core("attn_inf_tf32" if f32 else "attn_inf", defs)
            if f32:  # one sample: the TF32 core's query-tile-pair build (GatedInferenceCore, S = 1)
                core("attn_inf_tf32", ("QPAIR=1", *defs))
                core("attn_inf_tf32", ("QPAIR=1", "KSPLIT=1", *defs))  # few work items: the key split (sm100.tf32_ksplit)
                core("attn_inf_tf32", ("VMN=1", *defs))  # several samples: v MN-major from the q | k | v | g GEMM (token DiT part 8)
            if not f32:  # one sample: the query-tile-pair schedule (GatedInferenceCore._qpair)
                core("attn_inf", ("QPAIR=1", f"NHEAD={nh}", f"DHP={dh}", f"RSQDV={dh**-0.5!r}f"))
                core("attn_inf", (*defs, "FOLDARG=1"))  # B trunk_seeds in one launch (GatedInferenceCore.folds)
        elif f32:
            # sm100.backward_tf32(det=True): dK / dV in block order, the dQ slices per 64-key chunk, the dbias slices per sample group
            core("attn_fwd_tf32", defs)
            core("attn_dkv_tf32", (*defs, "DET=1"))
            core("attn_dqb_tf32", (*defs, "DQPART=1", "DET=1", "DBPART=1"))
        else:
            # sm100.backward(det=True): dK / dV in block order, the dQ slices per 128-key chunk
            core("attn_fwd2", defs)
            core("attn_fwd2", ("QPAIR=1", *defs))  # one sample (sm100.forward: A = 1)
            core("attn_dkv", (*defs, "DET=1"))
            core("attn_dkv", (*defs, "DET=1", "BIASNT=1"))  # one sample: the bias read as stored (token_dit_train._BIASNT_B200)
            core("attn_dqb", (*defs, "DQPART=1", "DQDET=1"))
            core("glue")  # backward: bias_transpose
    # ---- block-local atom DiTs
    atom = {(r["module"], r["dtype"] == "float32", r["mode"]) for r in rows if r["module"] in LOCAL_TC | VARIANT_CORE}
    esm = {r["dtype"] == "float32" for r in rows if r["module"] == ESM_TC and r["mode"] == "eval"}
    esm_train = {r["dtype"] == "float32" for r in rows if r["module"] == ESM_TC and r["mode"] == "train"}
    esm |= esm_train  # (training loads its dtype's inference sets first: local_dit_esm_b200.run_block)
    atom |= {("AtomDiTOpenFold3", f32, "eval") for f32 in esm}  # (its ready functions load the block-local inference sets first)
    for f32 in (False, True):
        if not any(m in LOCAL_TC | VARIANT_CORE and f == f32 for m, f, _ in atom):
            continue
        local = b200_atom_local.KERNELS_TF32 if f32 else b200_atom_local.KERNELS
        for stem in sorted({s for s, *_ in local.values()}):
            calls.append(ft.partial(b200_atom_local.cubin, stem))
        if not f32:  # the attention as a programmatic dependent launch (_load_pdl_fwd: the variant blocks, and inference of every block)
            calls.append(ft.partial(b200_atom_local.cubin, "lattn_fwd", False, ("LATTN_PDL=1",)))
        if not any(f == f32 for _, f, _ in atom):  # (every atom class here takes the block: LOCAL_TC, and VARIANT_CORE since round 3)
            continue
        if f32:  # _kernels32_ready: load_all32, load_fused32
            for stem in sorted({s for s, *_ in b200_atom.KERNELS_TF32.values()} | {s for s, *_ in b200_atom.KERNELS_FUSED32.values()}):
                calls.append(ft.partial(b200_atom.cubin_tf32, stem))
            calls.append(ft.partial(b200_atom.cubin_tf32, "fused_tf32", ("PAIR=1",)))  # (post32 on CTA pairs)
            calls.append(ft.partial(b200_atom.cubin_tf32, "fused_tf32", ("TU64=1",)))  # (round 3 part 11: 64-row units)
            for _, _, _, *defs in b200_atom.KERNELS_FUSED32.values():  # (round 3 part 14: the fp16-exact builds)
                if defs and "F16=1" in defs[0] and "RMS=1" not in defs[0]:
                    calls.append(ft.partial(b200_atom.cubin_tf32, "fused_tf32", *defs))
            for stem, _, _, *defs in b200_atom.KERNELS_TF32.values():  # (round 3 part 11: builds with defines, gemm_k64)
                if defs:
                    calls.append(ft.partial(b200_atom.cubin_tf32, stem, *defs))
        else:  # _kernels_ready: every row kernel of local_dit_b200._ROW_KERNELS
            from miniworld_engine.kernels.integrations import local_dit_b200

            for name in local_dit_b200._ROW_KERNELS:
                stem, _, _, _, *defs = b200_atom.KERNELS[name]
                calls.append(ft.partial(b200_atom.cubin, stem, *defs))
                if stem == "post_fwd":  # (round 3 part 10: post_fwd launches its BOX3 build, b200_atom._box3_kernel)
                    calls.append(ft.partial(b200_atom.cubin, stem, (*(defs[0] if defs else ()), "BOX3=1")))
            for name in ("prex_eb", "prexp_eb"):  # (round 3 part 12: inference, the pair bias inside the projection launch)
                stem, _, _, _, *defs = b200_atom.KERNELS[name]
                calls.append(ft.partial(b200_atom.cubin, stem, *defs))
        var_modes = {mode for m, f, mode in atom if m in VARIANT_CORE and f == f32}
        if var_modes:  # the variant builds (local_dit_b200._run_variant: every build of the tables is loaded, whatever the model)
            calls.append(ft.partial(b200_atom.cubin, "conf_mask"))
            if f32:
                for stem, _, _, *defs in b200_atom.KERNELS_FUSED32_VAR.values():
                    calls.append(ft.partial(b200_atom.cubin_tf32, stem, *defs))
                if "train" in var_modes:
                    calls.append(ft.partial(b200_atom.cubin, "var_rows_tf32"))
            else:
                calls.append(ft.partial(b200_atom.cubin, "var_pack"))
                for stem, _, _, _, *defs in b200_atom.KERNELS_VAR.values():
                    calls.append(ft.partial(b200_atom.cubin, stem, *defs))
                    if stem == "post_fwd":
                        calls.append(ft.partial(b200_atom.cubin, stem, (*(defs[0] if defs else ()), "BOX3=1")))
                if "train" in var_modes:
                    calls.append(ft.partial(b200_atom.cubin, "kqln_bwd"))
                    calls.append(ft.partial(b200_atom.cubin, "col_jobs"))
        if any(f == f32 and mode == "train" for _, f, mode in atom):  # the weight gradients (b200_atom.AtomDw)
            bkr, ns = (32, 2) if f32 else (16, 5)
            calls.append(ft.partial(b200_atom.cubin_tf32, "dw_atom", (f"DW_BKR={bkr}", f"NS={ns}", "DW_F32") if f32 else (f"NS={ns}",)))
    # ---- AtomDiTESMFold2Pair inference (local_dit_esm_b200._esm_ready / _esm32_ready)
    if False in esm:  # bf16: the RMS row builds, the rank-window attention, the pair bias by position and the q / k norm + rotary
        for stem, _, _, _, *defs in b200_atom.KERNELS_ESM.values():
            calls.append(ft.partial(b200_atom.cubin, stem, *defs))
            if stem == "post_fwd":
                calls.append(ft.partial(b200_atom.cubin, stem, (*(defs[0] if defs else ()), "BOX3=1")))
        for stem in ("lattn_esm", "lesm"):
            calls.append(ft.partial(b200_atom_local.cubin, stem))
        calls.append(ft.partial(b200_atom_local.cubin, "lattn_esm", False, ("LATTN_PDL=1",)))  # (round 3 part 11: esm._load_esm_pdl)
    if True in esm:  # fp32: the fused RMS row kernels around the TF32 attention
        calls.append(ft.partial(b200_atom.cubin_tf32, "fused_tf32", ("RMS=1",)))
        calls.append(ft.partial(b200_atom.cubin_tf32, "fused_tf32", ("RMS=1", "PAIR=1")))
        calls.append(ft.partial(b200_atom.cubin_tf32, "fused_tf32", ("RMS=1", "TU64=1")))  # (round 3 part 11: 64-row units)
        for _, _, _, *defs in b200_atom.KERNELS_FUSED32.values():  # (round 3 part 14: the fp16-exact RMS builds)
            if defs and "F16=1" in defs[0] and "RMS=1" in defs[0]:
                calls.append(ft.partial(b200_atom.cubin_tf32, "fused_tf32", *defs))
        for stem in ("lattn_esm_tf32", "lesm"):
            calls.append(ft.partial(b200_atom_local.cubin, stem))
        calls.append(ft.partial(b200_atom_local.cubin, "lesm", False, ("EARLY=1",)))  # (round 3 part 12: esm._load_esm_bias_early)
    if esm_train:  # training on the tcgen05 block (round 3 part 7, local_dit_esm_b200._esm_train_ready + the packs / d bq launches)
        for stem, _, _, _, *defs in (*b200_atom.KERNELS_ESM.values(), *b200_atom.KERNELS_ESM_TRAIN.values()):
            calls.append(ft.partial(b200_atom.cubin, stem, *defs))
        for defs in ((), ("F32=1",)):
            calls.append(ft.partial(b200_atom.cubin, "esm_pack", defs))
        calls.append(ft.partial(b200_atom.cubin, "col_jobs"))
        for stem in ("lattn_esm", "lattn_esm_dq", "lattn_esm_dkv", "lesm", "lbias", "lattn_esm_tf32", "lattn_esm_dq_tf32", "lattn_esm_dkv_tf32", "lbias_tf32"):
            calls.append(ft.partial(b200_atom_local.cubin, stem))
        calls.append(ft.partial(b200_atom.cubin_tf32, "dw_atom", ("NS=5",)))
        calls.append(ft.partial(b200_atom.cubin_tf32, "dw_atom", ("DW_BKR=32", "NS=2", "DW_F32")))
    return calls
