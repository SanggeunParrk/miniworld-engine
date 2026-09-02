"""Every whole-op entry, at the widths the model actually presents, on the card present.

`transition/whole_op.py`'s pre-Hopper wide-d branch handed `triton_layernorm` an
already-flattened `(M, D)` tensor. `rows_of` refuses that shape on purpose -- once a token
`(B, L, D)` and a pair `(B, L, L, D)` are both 2-D it cannot tell which key to build -- so the
branch raised `ValueError` on every call it was selected for. It had been that way silently,
because nothing exercised it: the model's `modules.Transition` carries its own copy of the arch
dispatch and never routes through the whole-op, and the tests that name `ops.transition` check
the export table and the compile surface, not a call.

That is the shape of bug this file exists for. A whole-op is the public entry point, so "does it
run at the widths krystal declares, on this GPU" is a claim worth asserting rather than assuming.
Widths come from `config/krystal/model/*.yaml`: d_pair_atom 16, d_single_atom / d_pair 128,
d_single 384, d_single_token 768, all with `transition_n: 2`.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                      # pragma: no cover - guarded by the marker
    pytest.skip("needs a CUDA device", allow_module_level=True)

from miniworld_engine import ops
from miniworld_engine.kernels.transition.reference import transition_pytorch

KRYSTAL_WIDTHS = (16, 128, 384, 768)
N_EXPAND = 2


def _weights(d: int, dtype: torch.dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    kw = {"device": "cuda", "dtype": dtype, "generator": g}
    return {
        "ln_in_weight": torch.randn(d, **kw),
        "ln_in_bias": torch.randn(d, **kw),
        "expand_a_weight": torch.randn(N_EXPAND * d, d, **kw) * 0.05,
        "expand_b_weight": torch.randn(N_EXPAND * d, d, **kw) * 0.05,
        "squeeze_weight": torch.randn(d, N_EXPAND * d, **kw) * 0.05,
        "n": N_EXPAND,
    }


@pytest.mark.parametrize("d_hidden", KRYSTAL_WIDTHS)
def test_transition_whole_op_runs_and_matches_reference(d_hidden: int) -> None:
    dtype = torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, 512, d_hidden, device="cuda", dtype=dtype, generator=g)
    kw = _weights(d_hidden, dtype)

    with torch.no_grad():
        got = ops.transition(x, **kw)
        want = transition_pytorch(
            x, kw["ln_in_weight"], kw["ln_in_bias"], kw["expand_a_weight"],
            kw["expand_b_weight"], kw["squeeze_weight"], N_EXPAND,
        )

    assert got.shape == x.shape
    cos = torch.nn.functional.cosine_similarity(
        got.flatten().float(), want.flatten().float(), dim=0,
    ).item()
    # bf16 through two GEMMs and a SwiGLU; cosine, not allclose -- the reference accumulates
    # differently and an elementwise tolerance here would only be a tolerance-tuning exercise.
    assert cos > 0.99, f"d_hidden={d_hidden}: cos={cos:.6f}"
