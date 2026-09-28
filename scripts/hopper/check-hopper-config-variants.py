"""Additional CPU-only compilation of nondefault launch settings."""
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import cutlass

from miniworld_engine.autotune.hopper_cuda_config import candidates, layernorm_candidates
from miniworld_engine.kernels.layernorm.cuda import _ext as ln_ext
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import _compile_fused
from miniworld_engine.kernels.transition.cuda import _ext

for kind in ("expand_gate", "gatebwd"):
    cfg = next(c for c in candidates(kind, 128)
               if c["warpgroups"] == 1 and c["bn"] == 64 and c["kt"] == 64
               and c["stages"] == 2 and c["min_blocks"] == 1)
    print(f"compile {kind} alternative {cfg}", flush=True)
    _ext(kind, 128, cfg)
    print("PASS", flush=True)

for config in (layernorm_candidates("compile", 128, 2)[0], {"warps": 8, "min_blocks": 1}):
    print(f"compile LayerNorm {config}", flush=True)
    ln_ext(config)
    print("PASS", flush=True)

for ws in (False, True):
    _compile_fused.__wrapped__(
        cutlass.BFloat16, cutlass.BFloat16, cutlass.BFloat16,
        "k", "k", "n", cutlass.Float32, (9, 0), (128, 128, 1, 1, True),
        warp_specialized_stats=ws,
    )
    print(f"PASS M2 warp_specialized_stats={ws}", flush=True)
