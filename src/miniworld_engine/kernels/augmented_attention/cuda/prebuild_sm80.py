"""The A100 (sm_80) builds of this family's token / atom DiT attention (``kernels/_prebuild.py``, ``prebuild(target, rows)`` with
``target.arch == "sm80"``), from the requirements rows of the module classes whose ``"cuda"`` backend reaches them on an A100:

* the token DiT attention (``kernels/integrations/augattn_sm80.py``: TokenDiTProtenix, TokenDiTBoltz2, TokenDiTMiniWorld -- its per-head
  q / k RMSNorm runs inside that path): the core / glue / TF32 extensions of ``sm80.py``;
* the atom block (``kernels/integrations/local_dit_sm80_block.py``: AtomDiTProtenix): the row extension of ``sm80_atom_rows`` and the
  window-attention / pair-bias cubins of ``sm80_atom_local``.

The extensions take no model dims (one build serves every width), so a family's build is planned when some row of its classes exists.
The sm_80 AttentionPairBias builds (``apb`` with ``-DAPB_SM80``, ``integrations/attention_pair_bias_sm80.py``) are not here."""

from __future__ import annotations

#: classes whose A100 "cuda" backend runs the token attention of integrations/augattn_sm80
TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTBoltz2", "TokenDiTMiniWorld"})
#: classes whose A100 "cuda" backend runs integrations/local_dit_sm80_block
ATOM = frozenset({"AtomDiTProtenix"})
#: the window attention / pair-bias / glue sources of sm80_atom_local (one cubin each)
ATOM_LOCAL_STEMS = ("glue", "lattn_dkv", "lattn_dq", "lattn_fwd", "lbias")
#: (augattn_sm80's helper extensions -- atom_qkvg_*, atom_output_*, shared_*_sm80, atom_sample_uniform -- serve the dense atom width d = 128
#: only, which no wired class reaches: not built. The tf32_*_rne / token_grad / token_shared_scale extensions served the removed, never-wired token_padded_augattn_sm80.)


def prebuild_sm80(rows):
    import functools as ft

    modules = {r["module"] for r in rows}
    if modules & TOKEN:
        from . import sm80

        yield sm80._ext
        yield sm80._glue_ext
        if any(r["module"] in TOKEN and r["dtype"] == "float32" for r in rows):
            yield sm80._tf32_ext
    if "AttentionPairBias" in modules:   # integrations/attention_pair_bias_sm80's composed path (fp32, the wide pairs, gate_bias)
        from . import sm80

        yield sm80._ext
        yield sm80._glue_ext
        if any(r["module"] == "AttentionPairBias" and r["dtype"] == "float32" for r in rows):
            yield sm80._tf32_rna_ext
    if modules & ATOM:
        from . import sm80_atom_local, sm80_atom_rows

        yield sm80_atom_rows._ext
        for stem in ATOM_LOCAL_STEMS:
            yield ft.partial(sm80_atom_local.cubin, stem)
