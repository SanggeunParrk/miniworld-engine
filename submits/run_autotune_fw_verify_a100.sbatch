#!/bin/bash
# Verify the backend-agnostic autotune framework refactor on A100:
#  (1) new APIs: as_cfg_dict / config round-trip (incl. cute-style dict + cluster tuple) /
#      select_config hit + miss  -- pure framework mechanics, no cute hardware needed.
#  (2) the two Triton builders still build valid caches through the generic tune_bucket core.
#SBATCH --job-name=mwk_autotune_fw
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/autotune_fw_verify_a100_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-kernels
echo "host=$(hostname) date=$(date)"

echo "### (1) backend-agnostic API mechanics (no GPU config needed) ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src MINIWORLD_KERNELS_CACHE_DIR=$(mktemp -d) python - <<PY
import os
from miniworld_engine.autotune import cache as C

# --- as_cfg_dict over all three shapes ---
import triton
tc = triton.Config({"BM": 128}, num_warps=4, num_stages=3)
assert C.as_cfg_dict(tc) == {"kwargs": {"BM": 128}, "num_warps": 4, "num_stages": 3}
cute = {"tile_m": 128, "tile_n": 256, "cluster": (2, 1), "pingpong": False}   # cute-style dict
d = C.as_cfg_dict(cute)
assert d["kwargs"]["cluster"] == (2, 1) and d["num_warps"] == 0, d
wrapped = {"kwargs": {"tile_m": 64}, "num_warps": 8, "num_stages": 2}
assert C.as_cfg_dict(wrapped) == wrapped, C.as_cfg_dict(wrapped)

# --- JSON round-trip: cute cluster tuple -> list -> tuple, sig matches ---
jd = C.config_to_dict(cute)
assert jd["kwargs"]["cluster"] == [2, 1], jd            # JSON-safe list on disk
assert C._sig(cute) == C._sig_from_dict(jd), (C._sig(cute), C._sig_from_dict(jd))
print("  as_cfg_dict / config_to_dict / sig round-trip OK (triton + cute-dict + cluster tuple)")

# --- config_space_hash stable across backend types ---
h1 = C.config_space_hash([cute, wrapped]); h2 = C.config_space_hash([wrapped, cute])
assert h1 == h2, "config_space_hash must be order-independent"
print(f"  config_space_hash order-independent OK ({h1})")

# --- select_config hit / miss / stale, cute pick-one path ---
gk = C.gpu_key()
op = "unit_probe_cute"
cands = [cute, {"tile_m": 256, "tile_n": 128, "cluster": (1, 1), "pingpong": True}]
csh = C.config_space_hash(cands)
# store two ranked cute configs for one bucket
C.store_ranked_configs(op, gk, "bfloat16", "K=128", [(cands[0], 0.10), (cands[1], 0.20)], csh)
best = C.select_config(op, dtype="bfloat16", bucket="K=128", candidates=cands)
assert best is not None and best["kwargs"]["tile_m"] == 128 and best["kwargs"]["cluster"] == (2, 1), best
assert "ms" not in best, best
print(f"  select_config HIT -> {best[\"kwargs\"]}")
miss = C.select_config(op, dtype="bfloat16", bucket="K=999", candidates=cands)   # unseen bucket
assert miss is None, miss
stale = C.select_config(op, dtype="bfloat16", bucket="K=128", candidates=cands + [{"tile_m": 32}])
assert stale is None, "config_space_hash change must invalidate (stale)"
print("  select_config MISS + STALE -> None (warn-once) OK")

os.environ["MINIWORLD_RUN_AUTOTUNE"] = "1"
assert C.select_config(op, dtype="bfloat16", bucket="K=128", candidates=cands) is None
print("  MINIWORLD_RUN_AUTOTUNE=1 bypasses cache OK")
print("API MECHANICS OK")
PY'
echo "[api exit=$?]"

echo "### (2) Triton builders through generic tune_bucket core ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m miniworld_engine.autotune.build --op all'
echo "[build exit=$?]"

RT="${MINIWORLD_KERNELS_CACHE_DIR:-$HOME/.cache/miniworld_engine}/autotune"
echo "### rebuilt runtime caches ###"
find "$RT" -name '*.json' -print 2>/dev/null
echo "AUTOTUNE FW VERIFY DONE"
