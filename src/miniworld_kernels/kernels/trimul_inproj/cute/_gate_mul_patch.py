"""Custom quack epilogue: postact = act(A@B) ⊙ C  (external multiply), for the trimul gate.

Stock `gemm_act` applies the activation to (A@B + beta·C) — an ADDITIVE C. The trimul gate
needs y = sigmoid(x_n @ Wg) ⊙ proj — the activation on A@B, then a MULTIPLY by an external
(M,N) tensor (proj). That fuses the whole gate (GEMM + sigmoid + ⊙proj) into ONE launch,
killing the separate elementwise mul AND the gate (M,N) HBM round-trip.

We OWN one policy point (like `_bdll_patch`): `GemmActSm90.epi_visit_subtile`. The patched
version, when a C operand is present, computes `act(rD) ⊙ C` instead of `act(rD + C)`; with no
C it is unchanged (so the inference gemm_act(sigmoid) path and any other non-gated act GEMM
keep working). The C-tile load, compile, and launch are all quack's, untouched.

The method is defined here as a REAL module-level `@cute.jit` function (NOT exec'd from a
string) so cute's trace-time `inspect.getsource` can read it — exec'd code has no source file
and fails to parse. Call `apply()` (after `_bdll_patch.apply()`), then
`gemm_act(A=x_n, B=Wg, C=proj, activation="sigmoid")` returns sigmoid(x_n@Wg)⊙proj in one kernel.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import quack.gemm_act as _ga
from cutlass import const_expr
from quack.gemm_default_epi import GemmDefaultEpiMixin

_APPLIED = False


@cute.jit
def _epi_visit_subtile_actmul(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
    """Conditional epilogue:
      • act_fn present AND C present  → act(A@B) ⊙ C   (the trimul gate: C=proj)
      • otherwise (act=None+C, or no C) → STOCK quack: act(A@B + C) / (A@B + C) / act(A@B)
    So a fused gate (act⊙C) and a plain C-add GEMM (gemm_act act=None, C=… = addmm for the
    bwd dx_n fusion) coexist on the same patched GemmActSm90 without clobbering each other."""
    if const_expr(params.act_fn is not None and tRS_rC is not None):
        # gate: act(rD) ⊙ C — default WITHOUT C-add (keep alpha/rowvec), then multiply.
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, None)
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[i]) * tRS_rC[i].to(self.acc_dtype)
        else:
            for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                a0, a1 = params.act_fn((tRS_rD[2 * i], tRS_rD[2 * i + 1]))
                tRS_rPostAct[2 * i] = a0 * tRS_rC[2 * i].to(self.acc_dtype)
                tRS_rPostAct[2 * i + 1] = a1 * tRS_rC[2 * i + 1].to(self.acc_dtype)
    else:
        # stock: default applies C-add (if any), then activation (if any).
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                    tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1])
                    )
        else:
            tRS_rPostAct = tRS_rD
    return tRS_rPostAct


def apply() -> None:
    """Patch GemmActSm90.epi_visit_subtile to do act(A@B)⊙C when C is present. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    _ga.GemmActSm90.epi_visit_subtile = _epi_visit_subtile_actmul
    _APPLIED = True
