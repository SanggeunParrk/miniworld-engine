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

Ops outside these cases are reported, not asserted on. They include alternative A6000 paths
as well as sm90/sm100 CuTeDSL paths; absence alone does not establish an architecture restriction.
The cases include inference and training, current token/atom widths, and SWA preprocessing.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                      # pragma: no cover - guarded by the marker
    pytest.skip("needs a CUDA device", allow_module_level=True)

from torch._library.custom_ops import CustomOpDef

from miniworld_engine import settings
from miniworld_engine.modules import (
    ImplementationType,
    MSAPairWeightedAveraging,
    OuterProductMean,
    PairformerBlock,
    PairformerConfig,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.swa_atom_attention.module import (
    build_attention_params,
)
from miniworld_engine.modules.swa_dit import SWADiTBlock

DEV, DT = "cuda", torch.bfloat16
L, D = 384, 128
OURS = ImplementationType.MINIWORLD


def _t(*shape):
    return torch.randn(*shape, device=DEV, dtype=DT, requires_grad=True)


def _pair_cases():
    yield Transition(d_hidden=D, n=4, implementation=OURS), (_t(1, L, L, D),)
    yield (TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=True, implementation=OURS,
                                  p_drop=0.0), (_t(1, L, L, D),))
    yield (TriangleAttention(d_pair=D, d_hidden=128, n_head=4, starting=True,
                             implementation=OURS), (_t(1, L, L, D),))
    yield (OuterProductMean(d_msa=D, d_pair=D, d_hidden=32, implementation=OURS),
           (_t(1, 8, L, D),))
    yield (MSAPairWeightedAveraging(d_msa=D, d_pair=D, d_hidden=32, n_head=8,
                                    implementation=OURS), (_t(1, 8, L, D), _t(1, L, L, D)))
    yield (PairformerBlock(PairformerConfig(d_pair=D, n_block=1, p_drop=0.0),
                           implementation=OURS), (_t(1, L, L, D),))


class _RMSRoPE(torch.nn.Module):
    """Exercise the standalone public fallback operations as well as fused SWA Q/K."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(32))

    def forward(self, x, cos, sin):
        from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm
        from miniworld_engine.kernels.rope.interface import triton_rope_3d

        return triton_rope_3d(triton_rmsnorm(x, self.weight), cos, sin)


def _cases():
    # Keep the established pair/MSA cases, and exercise their inference branches too.
    for training in (True, False):
        for model, inputs in _pair_cases():
            yield model, inputs, training
        augmentation = 48 if training else 5
        # Current production widths; atom256*A48 also reaches the >=8192-row fused tail.
        for width, cond_width, pair_width, heads, length in (
            (768, 384, 128, 16, 64), (128, 128, 16, 4, 256),
        ):
            model = DiTBlock(d_single=width, d_cond=cond_width, d_pair=pair_width,
                             n_head=heads, implementation=OURS)
            mask = torch.ones(1, length, device=DEV, dtype=torch.bool)
            mask[:, -7:] = False
            yield model, (_t(augmentation, 1, length, width),
                          _t(augmentation, 1, length, cond_width),
                          _t(1, length, length, pair_width), mask), training
        length = 256
        angles = torch.randn(1, length, 16, device=DEV)
        valid = torch.ones(augmentation, length, device=DEV, dtype=torch.bool)
        valid[:, -7:] = False
        attention_params = build_attention_params(
            angles.cos(), angles.sin(), valid, num_aug=augmentation)
        yield (SWADiTBlock(128, 128, 4, implementation=OURS),
               (_t(augmentation, length, 128), _t(augmentation, length, 128),
                attention_params), training)
        angles = torch.randn(augmentation, 128, 16, device=DEV)
        yield _RMSRoPE(), (_t(augmentation, 128, 4, 32), angles.cos(), angles.sin()), training


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
    previous = settings.configure(autotune_miss_cap=1)
    try:
        for model, inputs, training in _cases():
            model = model.to(DEV, DT).train(training)
            with torch.set_grad_enabled(training):
                out = model(*inputs)
                out = out[0] if isinstance(out, tuple) else out
                if training:
                    out.float().pow(2).mean().backward()
            del model, out
            torch.cuda.empty_cache()
    finally:
        CustomOpDef.__call__ = original
        settings.configure(**asdict(previous))
    return calls


def test_capture_saw_ops(captured):
    """Guard the guard: a capture that silently records nothing would make every check vacuous."""
    required = {"qk_norm_rope_fwd", "qk_norm_rope_bwd", "swa_gate_out_fwd",
                "swa_atom_attention_flash_window", "rmsnorm_fwd", "rmsnorm_bwd", "rope_3d"}
    if torch.cuda.get_device_capability(0) == (8, 6):
        required |= {"adaln_inference_fused", "adaln_gemm_gate", "adaln_cond_affine",
                     "adaln_dgrad_condln", "conditioned_transition_inference",
                     "conditioned_transition_composed_expand_swiglu",
                     "conditioned_transition_composed_squeeze_gate",
                     "conditioned_transition_b2b_fwd_train"}
    assert {f"miniworld_engine::{name}" for name in required} <= captured.keys()
    assert len(captured) >= 15, (
        f"only {len(captured)} ops captured -- the hook is not seeing calls, so the opcheck "
        f"below would pass by doing nothing")


#: `opcheck`'s default set includes `test_aot_dispatch_static` / `_dynamic`, which compile the op
#: forward AND BACKWARD through AOTAutograd. Except for the public FlashWindow op, these are launch wrappers inside an
#: `autograd.Function.forward`, and `kernels/_compile.py` says why `register_autograd` is
#: deliberately not used: `setup_context` can only save the op's inputs and outputs, so every
#: intermediate a backward needs (LN stats, the normalised activation) would have to become a
#: forward return. Keeping `autograd.Function` keeps `save_for_backward` free.
#:
#: So differentiating one of these ops DIRECTLY is not part of its contract, and asserting it is
#: fails 20+ ops with "no autograd formula was registered" -- a property the design does not claim.
#: The gradients are checked one level up, where they exist: `tests/numerics/test_numerical.py` compares each
#: kernel's dq/dk/dv/dbias against a torch reference through the Function.
#:
#: The compile path is NOT dropped along with it. The aot tests run with the arguments detached,
#: which is the shape these ops are really compiled in, so they still cover what they are worth
#: covering: that the fake's metadata survives a traced forward.
# FlashWindow is a public autograd op; the lower-level launch wrappers use enclosing Functions.
_AUTOGRAD_OPS = {"miniworld_engine::swa_atom_attention_flash_window"}
_GRAD_FREE = ("test_schema", "test_faketensor")
_COMPILED = ("test_aot_dispatch_static", "test_aot_dispatch_dynamic")


def _detach(value):
    """Same structure, nothing requiring grad."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (list, tuple)):
        return type(value)(_detach(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    return value


def test_every_exercised_op_satisfies_its_contract(captured):
    """Schema and fake, against the arguments the op is really called with."""
    failures = []
    for name, (op, args, kwargs) in sorted(captured.items()):
        try:
            torch.library.opcheck(op, args, kwargs, test_utils=_GRAD_FREE)
            if name in _AUTOGRAD_OPS:
                torch.library.opcheck(op, args, kwargs, test_utils=("test_autograd_registration",))
        except Exception as e:  # noqa: PERF203 -- every op is checked; one failure is not the end
            failures.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
    assert not failures, "op contract violations:\n  " + "\n  ".join(failures)


def test_every_exercised_op_survives_a_traced_forward(captured):
    """The compile half, with the inputs detached -- see the note above `_GRAD_FREE`."""
    failures = []
    for name, (op, args, kwargs) in sorted(captured.items()):
        try:
            torch.library.opcheck(op, _detach(args), _detach(kwargs), test_utils=_COMPILED)
        except Exception as e:  # noqa: PERF203 -- every op is checked; one failure is not the end
            failures.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
    assert not failures, "ops that do not survive a traced forward:\n  " + "\n  ".join(failures)


def test_autograd_registration_matches_the_public_contract(captured):
    """Only the public FlashWindow op registers autograd directly.

    All other captured ops are launch wrappers whose enclosing autograd.Function owns backward.
    A real backward proves the exception works, and detects accidental registrations elsewhere.
    """
    differentiable, unchecked = [], []
    for name, (op, args, kwargs) in sorted(captured.items()):
        grad_args, seeded = [], False
        for a in _detach(args):
            if not seeded and isinstance(a, torch.Tensor) and a.is_floating_point():
                grad_args.append(a.detach().clone().requires_grad_(True))
                seeded = True
            else:
                grad_args.append(a)
        if not seeded:
            unchecked.append(name)          # nothing to differentiate w.r.t.
            continue
        try:
            out = op(*grad_args, **kwargs)
            first = next((o for o in (out if isinstance(out, tuple) else (out,))
                          if isinstance(o, torch.Tensor) and o.is_floating_point()), None)
            if first is None:
                unchecked.append(name)
                continue
            first.sum().backward()
        except RuntimeError as e:
            if "no autograd formula" not in str(e):
                unchecked.append(f"{name} (raised something else: {str(e)[:80]})")
            continue
        except Exception as e:
            unchecked.append(f"{name} ({type(e).__name__})")
            continue
        differentiable.append(name)
    assert set(differentiable) == _AUTOGRAD_OPS, (
        f"Expected only {_AUTOGRAD_OPS} to register autograd; got {differentiable}; unchecked={unchecked}. "
        f"register_autograd is deliberately unused (setup_context cannot save the intermediates a "
        f"backward needs); if that changed, the test_utils split in this file needs revisiting.")
    # Not an assertion: an op with no float input, or one this card cannot run, is simply outside
    # what this check can say anything about. Printed so the number is visible rather than assumed.
    if unchecked:
        print(f"\n[opcheck] {len(unchecked)} op(s) not covered by the differentiability check: "
              f"{unchecked}")


def test_unexercised_ops_are_reported(captured, capsys):
    """Name ops outside these cases; absence does not imply the card cannot execute them."""
    registered = {f"miniworld_engine::{n}" for n in dir(torch.ops.miniworld_engine)
                  if not n.startswith("_")} - {"miniworld_engine::name"}
    never = sorted(registered - set(captured))
    with capsys.disabled():
        print(f"\n[opcheck] {len(captured)} ops verified on {torch.cuda.get_device_name(0)}; "
              f"{len(never)} never reached here (fakes UNVERIFIED):")
        for n in never:
            print(f"    {n}")
        print("OPCHECK_COVERAGE " + json.dumps({"verified": sorted(captured), "unexercised": never}))
