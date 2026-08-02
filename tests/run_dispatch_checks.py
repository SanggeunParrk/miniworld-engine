"""Standalone dispatch checks (no pytest dependency).

Run: ``pixi run python tests/run_dispatch_checks.py``

Mirrors tests/test_dispatch.py + tests/test_int64_offsets.py but as a plain
script, since this repo has no pytest env. Exits non-zero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)

_I = ImplementationType
_fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        _fails.append(msg)


def _builders():
    from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
    from miniworld_engine.modules.augmented_attention.module import (
        AugmentedAttentionPairBias,
    )
    from miniworld_engine.modules.conditioned_transition.module import (
        ConditionedTransition,
    )
    from miniworld_engine.modules.primitives import LayerNorm
    from miniworld_engine.modules.transition.module import Transition
    from miniworld_engine.modules.triangle_attention.bidirectional import (
        BidirectionalTriangleAttention,
    )
    from miniworld_engine.modules.triangle_attention.module import (
        TriangleAttention,
        TrianglePairAttention,
    )
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )
    from miniworld_engine.modules.triangle_multiplication.module import (
        TriangleMultiplication,
    )

    return {
        "LayerNorm": lambda i: LayerNorm(128, implementation=i),
        "Transition": lambda i: Transition(128, 4, implementation=i),
        "AdaptiveLayerNorm": lambda i: AdaptiveLayerNorm(128, 384, implementation=i),
        "ConditionedTransition": lambda i: ConditionedTransition(128, 384, 2, implementation=i),
        "AugmentedAttentionPairBias": lambda i: AugmentedAttentionPairBias(128, 384, 128, 4, implementation=i),
        "TriangleAttention": lambda i: TriangleAttention(128, 4, implementation=i),
        "TrianglePairAttention": lambda i: TrianglePairAttention(128, 4, implementation=i),
        "BidirectionalTriangleAttention": lambda i: BidirectionalTriangleAttention(128, 4, implementation=i),
        "TriangleMultiplication": lambda i: TriangleMultiplication(128, implementation=i),
        "BidirectionalTriangleMultiplication": lambda i: BidirectionalTriangleMultiplication(128, implementation=i),
    }


def main() -> int:
    print("== enum + public/internal mapping ==")
    check({b.value for b in KernelBackend} == {"pytorch", "triton", "cuda", "cute", "cuequivariance"},
          "KernelBackend has exactly the concrete backends")
    check("miniworld" not in {b.value for b in KernelBackend}, "MINIWORLD is not a KernelBackend")
    try:
        dispatch.to_kernel_backend(_I.MINIWORLD)
        check(False, "to_kernel_backend(MINIWORLD) raises")
    except InvalidImplementationError:
        check(True, "to_kernel_backend(MINIWORLD) raises")

    print("== import surface: every module builds with MINIWORLD and PYTORCH ==")
    for name, b in _builders().items():
        for impl in (_I.MINIWORLD, _I.PYTORCH):
            try:
                m = b(impl)
                ok = isinstance(m.implementation, _I) and (
                    not hasattr(m, "_backend") or isinstance(m._backend, KernelBackend)
                )
                check(ok, f"{name}({impl.value}) builds -> _backend={getattr(m, '_backend', 'n/a')}")
            except Exception as e:  # noqa: BLE001
                check(False, f"{name}({impl.value}) builds ({type(e).__name__}: {e})")
        # string coercion (Pairformer passes strings down)
        try:
            check(b("miniworld").implementation is _I.MINIWORLD, f"{name}('miniworld') coerces")
        except Exception as e:  # noqa: BLE001
            check(False, f"{name}('miniworld') coerces ({type(e).__name__})")

    print("== GPU-arch policy (mocked capability) ==")
    orig_cap = dispatch.capability
    import os
    os.environ.pop("MINIWORLD_TRIMUL_IMPL", None)
    os.environ.pop("MINIWORLD_TRIMUL_OUT_LAYOUT", None)
    try:
        for major, want_be, want_layout in [(10, KernelBackend.CUTE, "bdll_sm100"),
                                            (9, KernelBackend.CUTE, "bdll_direct"),
                                            (8, KernelBackend.TRITON, "bdll_direct")]:
            dispatch.capability = lambda device=None, _m=major: (_m, 0)
            check(dispatch.resolve_triangle_multiplication(_I.MINIWORLD) is want_be,
                  f"sm_{major}0: trimul MINIWORLD -> {want_be.value}")
            check(dispatch.trimul_out_layout() == want_layout,
                  f"sm_{major}0: out_layout -> {want_layout}")
            for res in (dispatch.resolve_transition, dispatch.resolve_triangle_attention,
                        dispatch.resolve_adaptive_layernorm, dispatch.resolve_conditioned_transition,
                        dispatch.resolve_augmented_attention):
                check(res(_I.MINIWORLD) is KernelBackend.TRITON,
                      f"sm_{major}0: {res.__name__} MINIWORLD -> triton")
            check(dispatch.resolve_layernorm(_I.MINIWORLD) is KernelBackend.CUDA,
                  f"sm_{major}0: resolve_layernorm MINIWORLD -> cuda")
        # env override wins
        dispatch.capability = lambda device=None: (10, 0)
        os.environ["MINIWORLD_TRIMUL_IMPL"] = "triton"
        check(dispatch.resolve_triangle_multiplication(_I.MINIWORLD) is KernelBackend.TRITON,
              "MINIWORLD_TRIMUL_IMPL=triton override wins over sm_100")
    finally:
        dispatch.capability = orig_cap
        os.environ.pop("MINIWORLD_TRIMUL_IMPL", None)

    print("== int64 static guard (M-index promotion in large-L kernels) ==")
    src = Path(__file__).resolve().parents[1] / "src" / "miniworld_engine" / "kernels"
    hardened = [
        ("trimul_inproj/triton/front.py", ".to(tl.int64)"),
        ("trimul_inproj/triton/back_fused.py", ".to(tl.int64)"),
        ("trimul_inproj/triton/gate_elem.py", ".to(tl.int64)"),
        ("tm1/triton/main.py", "tl.arange(0, BLOCK_M).to(tl.int64)"),
        ("tm2/triton/main.py", "tl.arange(0, BLOCK_M).to(tl.int64)"),
    ]
    for rel, snip in hardened:
        check(snip in (src / rel).read_text(), f"{rel} keeps int64 M-index")

    print("== parity: MINIWORLD LayerNorm vs PYTORCH (CUDA) ==")
    if torch.cuda.is_available():
        from miniworld_engine.modules.primitives import LayerNorm
        torch.manual_seed(0)
        x = torch.randn(2, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
        ref = LayerNorm(128, implementation=_I.PYTORCH).cuda().to(torch.bfloat16)
        mw = LayerNorm(128, implementation=_I.MINIWORLD).cuda().to(torch.bfloat16)
        mw.load_state_dict(ref.state_dict())
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(
                ref(x).float().flatten(), mw(x).float().flatten(), dim=0
            ).item()
        check(cos > 0.99, f"LayerNorm MINIWORLD vs PYTORCH cos={cos:.5f}")
    else:
        print("  skip  (no CUDA)")

    print(f"\n{'FAILED: ' + str(len(_fails)) if _fails else 'ALL PASSED'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
