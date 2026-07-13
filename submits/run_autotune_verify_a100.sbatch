#!/bin/bash
# Verify the autotune-cache foundation on A100: import gate, build the pilot cache
# (trimul_bidir_front), then exercise the prune hook directly (no kernel launch) for
# hit / miss / MINIWORLD_RUN_AUTOTUNE. Config choice is perf-only so this only checks the
# cache MECHANISM, not numerics.
#SBATCH --job-name=mwk_autotune_verify
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=benchmarks/autotune_verify_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
echo "host=$(hostname) date=$(date)"

pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python - <<PY
import os, warnings, torch
assert torch.cuda.get_device_capability(0)[0] == 8, "need A100"

# 1) import gate (new module + pilot kernel import cleanly)
from miniworld_kernels.autotune import cache as C
from miniworld_kernels.autotune.build import build_trimul_bidir_front
from miniworld_kernels.kernels.trimul_inproj.triton import bidirectional as B
print("import ok; gpu_key =", C.gpu_key())

FULL = list(B._bidir_front_kernel.configs)
CSH  = C.config_space_hash(FULL)
print("full grid =", len(FULL), "config_space_hash =", CSH)

# 2) build the pilot cache for d=128 ONLY (fast) -> writes runtime cache
build_trimul_bidir_front(d_pairs=(128,), seq_lens=(384,), top_k=5)

# helper: synth named_args for the prune (matches triton early_config_prune call)
x = torch.empty(1, device="cuda", dtype=torch.bfloat16)
def named(d):
    h2 = 2*d; gm = B.get_seq_group(384*384)
    return {"GROUP_M": gm, "H2": h2, "K": d, "x_ptr": x}

def run(tag, d, env=None):
    if env: os.environ["MINIWORLD_RUN_AUTOTUNE"] = env
    else: os.environ.pop("MINIWORLD_RUN_AUTOTUNE", None)
    C._warned.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kept = B._bidir_front_prune(FULL, named(d))
    warned = any("miniworld.autotune" in str(x.message) for x in w)
    print(f"  [{tag}] d={d}: kept={len(kept)}/{len(FULL)} warned={warned}")
    return len(kept), warned

print("=== prune mechanism ===")
run("HIT (built d=128)", 128)        # expect kept<=5, warned=False
run("MISS (unbuilt d=256)", 256)     # expect kept==full, warned=True
run("RUN_AUTOTUNE (built d=128)", 128, env="1")  # expect kept==full (cache ignored), warned=False
print("AUTOTUNE VERIFY DONE")
PY'
echo "[exit=$?]"
