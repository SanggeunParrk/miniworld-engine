"""``torch.library.opcheck`` on every miniworld op, with the arguments it is really called with.

Around sixty ``fake`` implementations were hand-written when the kernels became ops. A wrong one
is not a crash: it is a shape, dtype or stride the compiler believes and the kernel contradicts,
so eager stays correct and only the COMPILED path goes quietly wrong. Numerics parity cannot see
that -- it runs eager on both sides. ``opcheck`` tests the contract itself: the schema, that no
output aliases an input or another output, that only declared arguments are mutated, and that the
fake's metadata matches the real kernel's under both fake-tensor and AOT-dispatch tracing.

Arguments come from real module runs rather than being invented, because an invented input can
satisfy a fake that the real call shape would break. Capture hooks ``CustomOpDef.__call__``: a
``TorchDispatchMode`` does NOT see these ops (a first version of this used one and captured
nothing at all), because a ``custom_op`` with a Python implementation is dispatched through the
CustomOpDef wrapper before any dispatch-mode key is consulted.

Ops this card never reaches -- the sm90/sm100 CuTeDSL paths -- are reported, not asserted on.
Their fakes are genuinely unverified and this test says so out loud instead of implying coverage
it does not have.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                      # pragma: no cover - guarded by the marker
    pytest.skip("needs a CUDA device", allow_module_level=True)

from torch._library.custom_ops import CustomOpDef

from miniworld_engine import settings
from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    AugmentedAttentionPairBias,
    ConditionedTransition,
    ImplementationType,
    MSAPairWeightedAveraging,
    OuterProductMean,
    PairformerBlock,
    PairformerConfig,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)

DEV, DT = "cuda", torch.bfloat16
L, D = 384, 128
OURS = ImplementationType.MINIWORLD


def _t(*shape):
    return torch.randn(*shape, device=DEV, dtype=DT, requires_grad=True)


def _cases():
    yield Transition(d_hidden=D, n=4, implementation=OURS), (_t(1, L, L, D),)
    yield (TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=True, implementation=OURS,
                                  p_drop=0.0), (_t(1, L, L, D),))
    yield (TriangleAttention(d_pair=D, d_hidden=128, n_head=4, starting=True,
                             implementation=OURS), (_t(1, L, L, D),))
    yield (AugmentedAttentionPairBias(d_single=384, d_cond=384, d_pair=D, n_head=16,
                                      implementation=OURS),
           (_t(1, 1, L, 384), _t(1, 1, L, 384), _t(1, L, L, D)))
    yield (ConditionedTransition(d_hidden=D, d_cond=384, n=2, implementation=OURS),
           (_t(1, L, D), _t(1, L, 384)))
    yield (AdaptiveLayerNorm(d_hidden=D, d_cond=384, implementation=OURS),
           (_t(1, L, D), _t(1, L, 384)))
    yield (OuterProductMean(d_msa=D, d_pair=D, d_hidden=32, implementation=OURS),
           (_t(1, 8, L, D),))
    yield (MSAPairWeightedAveraging(d_msa=D, d_pair=D, d_hidden=32, n_head=8,
                                    implementation=OURS), (_t(1, 8, L, D), _t(1, L, L, D)))
    yield (PairformerBlock(PairformerConfig(d_pair=D, n_block=1, p_drop=0.0),
                           implementation=OURS), (_t(1, L, L, D),))


@pytest.fixture(scope="module")
def captured():
    """One real fwd+bwd per module, recording the first call of each op with its arguments."""
    if settings.current().compile_wrap != "custom_op":
        pytest.skip(f"compile_wrap={settings.current().compile_wrap!r}: no ops to check")

    calls: dict[str, tuple] = {}
    original = CustomOpDef.__call__

    def recording(self, *args, **kwargs):
        name = getattr(self, "_qualname", None) or getattr(self, "_name", "?")
        if name.startswith("miniworld_engine::") and name not in calls:
            calls[name] = (self._opoverload, args, kwargs)
        return original(self, *args, **kwargs)

    CustomOpDef.__call__ = recording
    try:
        for model, inputs in _cases():
            model = model.to(DEV, DT)
            out = model(*inputs)
            out = out[0] if isinstance(out, tuple) else out
            out.float().pow(2).mean().backward()
            del model
            torch.cuda.empty_cache()
    finally:
        CustomOpDef.__call__ = original
    return calls


def test_capture_saw_ops(captured):
    """Guard the guard: a capture that silently records nothing would make every check vacuous."""
    assert len(captured) >= 15, (
        f"only {len(captured)} ops captured -- the hook is not seeing calls, so the opcheck "
        f"below would pass by doing nothing")


def test_every_exercised_op_satisfies_its_contract(captured):
    failures = []
    for name, (op, args, kwargs) in sorted(captured.items()):
        try:
            torch.library.opcheck(op, args, kwargs)
        except Exception as e:  # noqa: PERF203 -- every op is checked; one failure is not the end
            failures.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
    assert not failures, "op contract violations:\n  " + "\n  ".join(failures)


def test_unexercised_ops_are_reported(captured, capsys):
    """Name the ops this card cannot reach, so their fakes are known-unverified, not assumed-good."""
    registered = {f"miniworld_engine::{n}" for n in dir(torch.ops.miniworld_engine)
                  if not n.startswith("_")} - {"miniworld_engine::name"}
    never = sorted(registered - set(captured))
    with capsys.disabled():
        print(f"\n[opcheck] {len(captured)} ops verified on {torch.cuda.get_device_name(0)}; "
              f"{len(never)} never reached here (fakes UNVERIFIED):")
        for n in never:
            print(f"    {n}")
