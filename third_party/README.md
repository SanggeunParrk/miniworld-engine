# third_party: external code used as comparison baselines

Tracked files are metadata only (README / INVENTORY, UPSTREAM.json with the pinned revision and sha256, fetch scripts, patches). The upstream
trees are fetched by each entry's `fetch.sh` into `<entry>/upstream/` (git-ignored); nothing here is part of the wheel and nothing is vendored.

| entry | what | license | used by |
|---|---|---|---|
| `anthropic/` | Anthropic uplifting-biomolecular-modeling kits (`anthropic` backend) | Apache-2.0 | E1 / E2 / E3 |
| `fast_trimul/` | `fast_trimul` 3.0.6: CuTe DSL fused triangle multiplicative update (fp16) | Apache-2.0 | E5 `fast_trimul` |
| `gpumode_trimul/` | arseniivanov/trimul: GPU MODE TriMul competition Triton kernels (A100 / H100 / B200) | MIT | E5 `gpumode` |
| `geometric-nn-ops-trimul/` | AMD ROCm/geometric-nn-ops 1.0.0 `triangle_multiplicative_update` (Triton forward) | Apache-2.0 | E5 `geonn` |

(The triangle-attention entries of E5 are on the `third-party-attn` branch; `geometric-nn-ops-trimul/` fetches the same upstream commit as
its geometric-nn-ops entry, for the other function of the same package.)

Installing on a card: `third_party/<entry>/fetch.sh [TARGET]` (idempotent, sha256 checked); the E5 adapters look in `$PAPER_EXT_PATH`
(default `~/paper-ext`) and in `third_party/<entry>/upstream`, after the engine environment's site-packages.
