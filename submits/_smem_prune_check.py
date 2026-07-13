"""Per-kernel smem-prune calibration on the real GPU: for each kernel whose grid we expanded,
apply its early_config_prune (RUN_AUTOTUNE=1 -> returns exactly the smem base_prune result, cache
skipped) to the FULL config grid and report kept/total. Flags over-prune (kept<=2 -> degenerate
search) and empty grids. One-off driver, no autotune run."""
import os

os.environ["MINIWORLD_RUN_AUTOTUNE"] = "1"  # make_cache_prune returns base_prune(configs) only

import importlib

import torch  # noqa: F401

# (module, autotuner attr, representative named_args the smem formula/bucket reads)
CASES = [
    ("miniworld_kernels.kernels.transition.triton.main", "transition_fwd_kernel", {"n": 4, "N": 256}),
    ("miniworld_kernels.kernels.transition.triton.main", "transition_bwd_kernel", {"n": 4, "N": 256}),
    ("miniworld_kernels.kernels.tm1.triton.main", "fused_sigmoid_gate_fwd_kernel", {"N": 128}),
    ("miniworld_kernels.kernels.tm1.triton.main", "fused_sigmoid_gate_bwd_kernel", {"N": 128}),
    ("miniworld_kernels.kernels.tm2.triton.main", "fused_sigmoid_gate2_fwd_kernel", {"N": 128}),
    ("miniworld_kernels.kernels.tm2.triton.main", "fused_sigmoid_gate2_bwd_kernel", {"N": 128}),
    ("miniworld_kernels.kernels.conditioned_transition.triton.inference",
     "_cond_transition_inference_kernel", {"BLOCK_K": 128, "D": 128}),
    ("miniworld_kernels.kernels.conditioned_transition.triton.composed",
     "_squeeze_gate_kernel", {}),
]

cap = torch.cuda.get_device_capability(0)
try:
    import triton as _t
    lim = _t.runtime.driver.active.utils.get_device_properties(0)["max_shared_mem"]
except Exception as e:  # noqa: BLE001
    lim = -1
print(f"device sm{cap[0]}{cap[1]}  max_shared_mem={lim} B ({lim/1024:.0f} KB)\n")

for mod_name, attr, named in CASES:
    mod = importlib.import_module(mod_name)
    at = getattr(mod, attr)
    full = list(at.configs)
    ecp = at.early_config_prune
    try:
        kept = ecp(list(full), named)
    except Exception as e:  # noqa: BLE001
        print(f"  {attr:38s} PRUNE-ERROR {type(e).__name__}: {e}")
        continue
    n_full, n_kept = len(full), len(kept)
    flag = "OK" if n_kept >= 3 else ("DEGENERATE(<=2!)" if n_kept >= 1 else "EMPTY!")
    # show the smem span of kept (min/max BLOCK product proxy) for a sanity feel
    def _blk(c):
        k = c.kwargs
        return (k.get("BLOCK_M"), k.get("BLOCK_N"), k.get("BLOCK_K"), k.get("BLOCK_DC"),
                c.num_warps, c.num_stages)
    print(f"  {attr:38s} kept {n_kept:3d}/{n_full:3d}  {flag}")
    if n_kept and n_kept <= 4:
        for c in kept:
            print(f"       {_blk(c)}")
print("\nSMEM PRUNE CHECK DONE")
