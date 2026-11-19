"""The B200 (sm_100a) builds of the bias-only token DiT (``kernels/_prebuild.py``, ``prebuild(target, rows)`` with ``target.arch ==
"sm100"``), from the requirements rows of ``TokenDiTMiniWorldBiasOnly`` (the class whose ``"cuda"`` backend runs
``integrations/bias_only_dit*.py`` on a B200; ``AttentionPairBias``'s fp32 weight pack, ``bo_pack_sm100``, is planned by augmented_attention).

The cubins are the ones the launchers ask ``augmented_attention.cuda.sm100.cubin`` for (``_sm100_kernel``), with the defines their own
selection helpers pick for each row's (samples, length) on the target's SM count -- the same calls, so the keys are the runtime's:

* inference, bf16 (``runner._step3b``): ``inf3_bf16.inf3_bf16_ready``'s fronts / tails / pair tail and its representative core, the
  step's core per (S, L) (``pick_group``, ``_pv_vb``; ``-DPDL_INF -DP_EARLY``), the one-launch weight pack and table pass (``bo_tab_sm100``);
* inference, fp32 (``runner._step3``): ``tf32.inf3_ready``'s fronts / tails (+ the pair tail ``TailTF32.cluster`` builds), the PDL core per
  (S, L) (``pick_group_tf32``), the weight pack;
* training, bf16: the forward core (``PvGateCore``, gated) and the SwiGLU GEMM with [a | b] (``GemmSwigluAB``); the backward fused
  (``bwd_fused``: ``bo_bwd_tail``, ``bo_wgrad``, and ``bo_pvdpb`` -- or, where the bias gradient runs on CTA pairs, the PDL pv dV + ``dpbx2``)
  at cond 384 / pair 128, else the per-step launches (pv dV ungated + ``dpb_sm100`` / ``dpbx2``);
* training, fp32: ``tf32.tf32_ready``'s cores and bias gradient, the gated / ungated cores per (S, L), the SwiGLU backward GEMM
  (``glu_op(bwd=True)``; the forward one is off), the bias gradient on CTA pairs where ``dpb32_pair``.

The row extensions (``bias_only_dit_rows_cuda``, ``bias_only_dit_train_rows_cuda``, ``bias_only_dit_f32_rows_cuda``) are the portable
builds (sm_80 / sm_90 / sm_100 images)."""

from __future__ import annotations

import functools as ft
import hashlib

#: the class whose B200 "cuda" backend is this family's bias-only block
CLASS = "TokenDiTMiniWorldBiasOnly"


def _cubin(stem: str, defs: tuple[str, ...], src_dir) -> ft.partial:
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    return ft.partial(sm100.cubin, stem, tuple(defs), src_dir=str(src_dir))


def prebuild_sm100(target, rows):
    from miniworld_engine.kernels._prebuild import _kv
    from miniworld_engine.kernels.conditioned_transition.cuda import _swiglu_sm100

    from . import (
        H,
        LAYOUTS,
        _dir,
        _dpb_nj,
        _head_defs,
        _pv_vb,
        _rows_ext,
        bwd_fused,
        dpb_pair_on,
        inf3_bf16,
        pick_group,
        tf32,
        train,
    )

    nsm = target.sms
    calls: list = []
    mine = [r for r in rows if r["module"] == CLASS]
    if not mine:
        return calls
    calls.append(ft.partial(_rows_ext, False))
    if any(r["mode"] == "train" for r in mine):
        calls.append(ft.partial(train._rows_ext, False))
    if any(r["dtype"] == "float32" for r in mine):
        calls.append(tf32.rows32)
    for r in mine:
        d = _kv(r["dims"])
        nh = int(d.get("n_head", H))
        da = int(d.get("d_single", 768))
        dh = da // nh
        if (nh, dh) not in LAYOUTS or da not in (768, 1024):
            continue
        hd = _head_defs(nh, dh)
        hp = 2 if dh == 32 else 1
        f32 = r["dtype"] == "float32"
        lengths, samples = r["lengths"], r["samples"]
        shapes = [(S, L) for S in samples for L in lengths]
        if r["mode"] == "eval":
            calls.append(_cubin("bo_pack_sm100", (), _dir))  # runner._pack_b200: the three-kernel step's pack in one launch
            calls.append(_cubin("bo_tab_sm100", (), _dir))  # runner._tab_b200: the one-block conditioning table pass
            if not f32:
                # inf3_bf16_ready: every tail (CL 8 / 6 and the pair tail) for x / out fp32 and bf16, every front for x fp32 and bf16, a PDL core
                for cl in (*inf3_bf16.TAIL_CLUSTERS, inf3_bf16.TAIL_PAIR):
                    for xbf, obf in ((True, True), (True, False), (False, False), (False, True)):
                        if cl == inf3_bf16.TAIL_PAIR:
                            calls.append(_cubin("bo_tail2_bf16", (f"DATT={da}", f"XBF={int(xbf)}", f"OBF={int(obf)}"), _dir))
                        else:
                            calls.append(_cubin("bo_tail_bf16", (f"CL={cl}", f"DATT={da}", f"XBF={int(xbf)}", f"OBF={int(obf)}"), _dir))
                for cl in (8, 6, 4) if da == 768 else (8, 4):  # FrontBF16.clusters
                    for xbf in (True, False):
                        calls.append(_cubin("bo_front_bf16", (f"CL={cl}", f"DATT={da}", f"XBF={int(xbf)}"), _dir))
                pv = {(1, ())}
                for S, L in shapes:  # the step's core (PvGateCore(pdl, p_early).kernel(pick_group, gate, _pv_vb))
                    vb = _pv_vb(L, S, dh)
                    pv.add((pick_group(S, L, nsm, nh, dh), () if vb == 128 else (f"VB={vb}",)))
                for sg, vdef in sorted(pv):
                    calls.append(_cubin("pv_gate_inf", (f"SG={sg}", "GATE=1", *vdef, *hd, "PDL_INF", "P_EARLY"), _dir))
            else:
                for cl in (8, 6, 4) if da == 768 else (8, 4):  # FrontTF32.clusters
                    calls.append(_cubin("bo_front_tf32", (f"CL={cl}", f"DATT={da}"), _dir))
                for cl in tf32.TAIL_CLUSTERS:
                    calls.append(_cubin("bo_tail_tf32", (f"CL={cl}", f"DATT={da}"), _dir))
                if tf32.tail_pair_on():  # TailTF32.cluster
                    calls.append(_cubin("bo_tail2_tf32", (f"DATT={da}",), _dir))
                batch = ("BATCHN",) if tf32.pv_batched() else ()
                for sg in sorted({1} | {tf32.pick_group_tf32(S, L, nsm, nh, dh) for S, L in shapes}):
                    calls.append(_cubin("pv_gate_tf32", (f"SG={sg}", "GATE=1", f"NHEAD={nh}", f"DHEAD={dh}", *batch, "PDL_INF"), _dir))
            continue
        # training
        if not f32:
            dc, dp = int(d.get("d_cond", 384)), int(d.get("d_pair", 128))
            fused = bwd_fused.bwd_fused_on() and dc == 384 and dp == 128  # integrations/bias_only_dit_train._bwd_fused
            calls.append(_swiglu_sm100(768, 1536, True))  # GemmSwigluAB: DIM 768, HMUL 2, SAVE_AB
            if fused:
                calls.append(_cubin("bo_bwd_tail", (f"DATT={da}", f"NHEAD={nh}"), _dir))
                calls.append(_cubin("bo_wgrad", (f"DATT={da}", f"WMAX={bwd_fused.wgrad_plan(da, nsm)[1]}"), _dir))
            src = hashlib.sha256()  # PvDpb's SRC_HASH (the cubin cache hashes bo_pvdpb.cu only)
            for f in ("pv_gate_inf.cu", "dpb_sm100.cu"):
                src.update((_dir / f).read_bytes())
            for A, L in shapes:
                vb = _pv_vb(L, A, dh)
                vdef = () if vb == 128 else (f"VB={vb}",)
                sg = pick_group(A, L, nsm, nh, dh)
                calls.append(_cubin("pv_gate_inf", (f"SG={sg}", "GATE=1", *vdef, *hd), _dir))  # forward: a = sigmoid(g) (P v)
                pair = dpb_pair_on(L, nh, hp, nsm)
                nj = _dpb_nj(L, nsm, nh // hp, hp)
                if fused and bwd_fused.pvdpb_on() and not pair:
                    calls.append(_cubin("bo_pvdpb", (f"SG={sg}", *vdef, f"NJ={nj}", *hd, f"SRC_HASH={src.hexdigest()[:16]}"), _dir))
                    continue
                # pv dV (fused: the PDL build; per step: the plain one), ungated, then the bias gradient (DpbKernel)
                calls.append(_cubin("pv_gate_inf", (f"SG={sg}", "GATE=0", *vdef, *hd, *(("PDL_INF",) if fused else ())), _dir))
                calls.append(_cubin("dpbx2_sm100", hd, _dir) if pair else _cubin("dpb_sm100", (f"NJ={nj}", *hd), _dir))
        else:
            batch = ("BATCHN",) if tf32.pv_batched() else ()
            pv = lambda sg, gate: _cubin(  # noqa: E731
                "pv_gate_tf32", (f"SG={sg}", f"GATE={int(gate)}", f"NHEAD={nh}", f"DHEAD={dh}", *batch), _dir
            )
            # tf32_ready(train): the ungated / gated representative cores and the bias gradient
            calls += [pv(1, True), pv(1, False), _cubin("dpb32_sm100", (f"NJ={tf32.DPB32_NJ}", f"NHEAD={nh}", f"DHEAD={dh}"), _dir)]
            calls.append(_cubin("gemm_glu_tf32", ("BWD",), _dir))  # glu_op(bwd=True); the forward GLU GEMM is off
            for A, L in shapes:
                sg = tf32.pick_group_tf32(A, L, nsm, nh, dh)
                calls += [pv(sg, True), pv(sg, False)]
                if tf32.dpb32_pair(L, nh, nsm):
                    calls.append(_cubin("dpbx2_sm100", ("TF32", f"NHEAD={nh}", f"DHEAD={dh}"), _dir))
    return calls
