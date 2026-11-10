"""The A100 (sm_80) builds of this family's token / atom DiT attention (``kernels/_prebuild.py``, ``prebuild(target, rows)`` with
``target.arch == "sm80"``), from the requirements rows of the module classes whose ``"cuda"`` backend reaches them on an A100:

* the token DiT attention (``kernels/integrations/augattn_sm80.py``: TokenDiTProtenix, TokenDiTBoltz2): the core / glue / TF32 extensions of ``sm80.py`` and the helper extensions its forward and backward launch (the shared /
  uniform-mask cores, the atom-width q k v g projections and output passes, the TF32 RNE variants, the token gradient rows);
* the atom block (``kernels/integrations/local_dit_sm80_block.py``: AtomDiTProtenix): the row extension of ``sm80_atom_rows`` and the
  window-attention / pair-bias cubins of ``sm80_atom_local``.

The extensions take no model dims (one build serves every width), so a family's build is planned when some row of its classes exists.
The sm_80 AttentionPairBias builds (``apb`` with ``-DAPB_SM80``, ``integrations/attention_pair_bias_sm80.py``) are not here."""

from __future__ import annotations

#: classes whose A100 "cuda" backend runs the token attention of integrations/augattn_sm80
TOKEN = frozenset({"TokenDiTProtenix", "TokenDiTBoltz2"})
#: classes whose A100 "cuda" backend runs the general sm_80 composition (integrations/a100_families.token: the per-head q / k RMSNorm)
GENERIC = frozenset({"TokenDiTMiniWorld"})
#: classes whose A100 "cuda" backend runs integrations/local_dit_sm80_block
ATOM = frozenset({"AtomDiTProtenix"})
#: the window attention / pair-bias / glue sources of sm80_atom_local (one cubin each)
ATOM_LOCAL_STEMS = ("glue", "lattn_dkv", "lattn_dq", "lattn_fwd", "lbias")
#: the helper extensions augattn_sm80's forward / backward import (module names under this package, each with ``_ext``)
TOKEN_HELPERS = (
    "atom_output_fused_sm80", "atom_output_tiled_sm80", "atom_qkvg_resident_sm80", "atom_qkvg_tiled_sm80", "atom_sample_uniform_sm80",
    "shared_mask_uniform_sm80", "shared_score_core_sm80", "shared_uniform_core_sm80", "tf32_qk_grad_rne_sm80", "tf32_qk_rne_sm80",
    "tf32_rne_hoisted_sm80", "tf32_rne_sm80", "token_grad_sm80",
)


def prebuild_sm80(rows):
    import functools as ft
    import importlib

    modules = {r["module"] for r in rows}
    if modules & TOKEN:
        from . import sm80

        yield sm80._ext
        yield sm80._glue_ext
        if any(r["module"] in TOKEN and r["dtype"] == "float32" for r in rows):
            yield sm80._tf32_ext
        for name in TOKEN_HELPERS:
            yield importlib.import_module(f"{__package__}.{name}")._ext
    if modules & GENERIC:   # the native rows of kernels/cuda_native.py and the norm / SwiGLU builds a100_families.token calls
        from miniworld_engine.kernels import cuda_native
        from miniworld_engine.kernels.layernorm.cuda import sm80 as norm_sm80
        from miniworld_engine.kernels.transition.cuda import fused_wide_sm80

        yield cuda_native._ext
        yield norm_sm80.ext
        yield fused_wide_sm80._ext
    if modules & ATOM:
        from . import sm80_atom_local, sm80_atom_rows

        yield sm80_atom_rows._ext
        for stem in ATOM_LOCAL_STEMS:
            yield ft.partial(sm80_atom_local.cubin, stem)
