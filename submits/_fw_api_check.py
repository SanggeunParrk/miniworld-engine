"""One-off verification of the backend-agnostic autotune framework APIs (run on A100 via
sbatch). NOT a pytest test — a throwaway driver next to the submit scripts. Exits non-zero on
any failed assertion so the sbatch log shows the failure."""
import os

import triton

from miniworld_engine.autotune import cache as C

# --- as_cfg_dict over all three config shapes -----------------------------------------------
tc = triton.Config({"BM": 128}, num_warps=4, num_stages=3)
assert C.as_cfg_dict(tc) == {"kwargs": {"BM": 128}, "num_warps": 4, "num_stages": 3}
cute = {"tile_m": 128, "tile_n": 256, "cluster": (2, 1), "pingpong": False}   # cute-style dict
d = C.as_cfg_dict(cute)
assert d["kwargs"]["cluster"] == (2, 1) and d["num_warps"] == 0, d
wrapped = {"kwargs": {"tile_m": 64}, "num_warps": 8, "num_stages": 2}
assert C.as_cfg_dict(wrapped) == wrapped, C.as_cfg_dict(wrapped)

# --- JSON round-trip: cute cluster tuple -> list on disk -> tuple, sig matches ---------------
jd = C.config_to_dict(cute)
assert jd["kwargs"]["cluster"] == [2, 1], jd            # JSON-safe list on disk
assert C._sig(cute) == C._sig_from_dict(jd), (C._sig(cute), C._sig_from_dict(jd))
print("  as_cfg_dict / config_to_dict / sig round-trip OK (triton + cute-dict + cluster tuple)")

# --- config_space_hash order-independent + backend-mixed ------------------------------------
h1 = C.config_space_hash([cute, wrapped])
h2 = C.config_space_hash([wrapped, cute])
assert h1 == h2, "config_space_hash must be order-independent"
print("  config_space_hash order-independent OK ({})".format(h1))

# --- select_config hit / miss / stale, cute pick-one path -----------------------------------
gk = C.gpu_key()
op = "unit_probe_cute"
cands = [cute, {"tile_m": 256, "tile_n": 128, "cluster": (1, 1), "pingpong": True}]
csh = C.config_space_hash(cands)
C.store_ranked_configs(op, gk, "bfloat16", "K=128", [(cands[0], 0.10), (cands[1], 0.20)], csh)
best = C.select_config(op, dtype="bfloat16", bucket="K=128", candidates=cands)
assert best is not None, "expected a hit"
assert best["kwargs"]["tile_m"] == 128 and best["kwargs"]["cluster"] == (2, 1), best
assert "ms" not in best, best
print("  select_config HIT -> {}".format(best["kwargs"]))
miss = C.select_config(op, dtype="bfloat16", bucket="K=999", candidates=cands)   # unseen bucket
assert miss is None, miss
stale = C.select_config(op, dtype="bfloat16", bucket="K=128", candidates=cands + [{"tile_m": 32}])
assert stale is None, "config_space_hash change must invalidate (stale)"
print("  select_config MISS + STALE -> None (warn-once) OK")

os.environ["MINIWORLD_RUN_AUTOTUNE"] = "1"
assert C.select_config(op, dtype="bfloat16", bucket="K=128", candidates=cands) is None
print("  MINIWORLD_RUN_AUTOTUNE=1 bypasses cache OK")
print("API MECHANICS OK")
