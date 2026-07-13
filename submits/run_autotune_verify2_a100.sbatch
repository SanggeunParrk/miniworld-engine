#!/bin/bash
# Verify increment 2 on A100: (1) refactored dispatch table (test_dispatch), (2) the
# transition-split cache wiring is SAFE (a d=256 forward routes to the split kernel, warns
# on the cache miss, and still runs), (3) the bidir pilot still HITs its built cache.
#SBATCH --job-name=mwk_autotune_v2
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=benchmarks/autotune_v2_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
echo "host=$(hostname) date=$(date)"

echo "### (1) dispatch table tests ###"
pixi run --frozen bash -c 'PYTHONPATH=src python -m pytest tests/test_dispatch.py -q'
echo "[dispatch exit=$?]"

echo "### (2)+(3) wiring safety + pilot hit ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python - <<PY
import warnings, torch
from miniworld_kernels.modules import Transition, ImplementationType as IT
from miniworld_kernels.autotune import build as _b, cache as C
from miniworld_kernels.kernels.trimul_inproj.triton import bidirectional as B

# (2) transition d=256 (miniworld) -> pre-Hopper split -> transition_fwd_kernel (wired prune).
# Expect: a "no tuned autotune cache" warning for transition_split_fwd AND a correct run.
m = Transition(d_hidden=256, implementation=IT.MINIWORLD).cuda().to(torch.bfloat16).eval()
x = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
C._warned.clear()
with warnings.catch_warnings(record=True) as w, torch.no_grad():
    warnings.simplefilter("always")
    y = m(x)
msgs = [str(x.message) for x in w if "miniworld.autotune" in str(x.message)]
print(f"  transition d=256 out={tuple(y.shape)} ok; autotune-warnings={len(msgs)}")
for s in msgs[:3]: print("   warn:", s[:130])
assert y.shape == (384, 256)

# (3) build the pilot cache (d=128) then confirm the prune HITs (kept<full, no warn).
_b.build_trimul_bidir_front(d_pairs=(128,), seq_lens=(384,), top_k=5)
FULL = list(B._bidir_front_kernel.configs)
xb = torch.empty(1, device="cuda", dtype=torch.bfloat16)
na = {"GROUP_M": B.get_seq_group(384*384), "H2": 256, "K": 128, "x_ptr": xb}
C._warned.clear()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    kept = B._bidir_front_prune(FULL, na)
hit = (len(kept) < len(FULL)) and not any("miniworld.autotune" in str(x.message) for x in w)
print(f"  pilot prune kept={len(kept)}/{len(FULL)} HIT={hit}")
assert hit, "pilot cache HIT expected"
print("AUTOTUNE V2 DONE")
PY'
echo "[exit=$?]"
