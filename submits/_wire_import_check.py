"""Import every kernel module touched by the cache-prune wiring, so any syntax / bad-import /
bad-prune-construction error surfaces immediately. One-off driver (not a pytest test)."""
import importlib

MODS = [
    "miniworld_engine.kernels.triangle_attention.triton.main",
    "miniworld_engine.kernels.augmented_attention.triton.main",
    "miniworld_engine.kernels.adaln.triton.training",
    "miniworld_engine.kernels.adaln.triton.inference",
    "miniworld_engine.kernels.conditioned_transition.triton.inference",
    "miniworld_engine.kernels.conditioned_transition.triton.composed",
    "miniworld_engine.kernels.bias_only_attention.triton.gate_out",
    "miniworld_engine.kernels.layernorm_linear.triton.fused",
    "miniworld_engine.kernels.layernorm_linear.triton.stats",
    "miniworld_engine.kernels.tm1.triton.main",
    "miniworld_engine.kernels.tm2.triton.main",
    "miniworld_engine.kernels.transition.triton.fused",
    "miniworld_engine.kernels.transition.triton.main",
]

fail = 0
for m in MODS:
    try:
        importlib.import_module(m)
        print(f"  OK  {m}")
    except Exception as e:  # noqa: BLE001
        fail += 1
        print(f"  FAIL {m}: {type(e).__name__}: {e}")

assert fail == 0, f"{fail} module(s) failed to import"

# Best-effort: list the wired op-tags discovered on autotuners (proves make_cache_prune ran +
# attached its _miniworld_op tag). Introspection quirks must not mask the import result above.
try:
    from triton.runtime.autotuner import Autotuner  # noqa: E402

    seen_ops = set()
    for m in MODS:
        mod = importlib.import_module(m)
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if isinstance(obj, Autotuner):
                op = getattr(getattr(obj, "early_config_prune", None), "_miniworld_op", None)
                if op:
                    seen_ops.add(op)
    print(f"\nwired op-tags discovered on autotuners ({len(seen_ops)}):")
    for op in sorted(seen_ops):
        print(f"  - {op}")
except Exception as e:  # noqa: BLE001
    print(f"\n(op-tag discovery skipped: {type(e).__name__}: {e})")

print("IMPORT CHECK OK")
