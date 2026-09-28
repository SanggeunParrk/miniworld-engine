# experiments

Research runners and their records. Nothing here is imported by the installed package or
selected by default dispatch; a result becomes production only when it is moved into
`src/miniworld_engine/` with its tests.

| directory | what it is |
|---|---|
| [transition_fused/](transition_fused/README.md) | development record of the fused H100 Transition (now wired in via `kernels/transition/cuda/fused_sm90a.py`) |
| [trimul_b7b12/](trimul_b7b12/README.md) | explicit H100 bidirectional TriMul B7–B12 training runner |
| [trimul_k1k3_inference/](trimul_k1k3_inference/README.md) | K1/K3 native inference overlay, served through `TRIMUL_NATIVE_BUILD_DIR` |
| [token_dit_fused/](token_dit_fused/README.md) | fused token DiT inference step |
| [token_dit_overlap/](token_dit_overlap/README.md) | measurements of what remains beyond the fused DiT v6 step |
| [token_dit_train/](token_dit_train/) | token DiT training core (source of `kernels/augmented_attention/cuda`) |

## Archived

Removed from the tree on 2026-09-27 to keep the checkout navigable. Everything is still in
history; the tag `archive/pre-tidy-20260927` is the last commit that contains it:

```sh
git show archive/pre-tidy-20260927:experiments/trimul_training_v2/README.md
git archive archive/pre-tidy-20260927 experiments/trimul_training_v2 | tar -x -C /tmp/restore
```

| path at the tag | what it was |
|---|---|
| `experiments/trimul_training_v2/` | 2.0.0 TriMul training research capsule: 6,625 run files from Sept 17–23 (`runs/`), `verify_release.py`. Its selected kernels are vendored in `src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/` (see `PROVENANCE.json`). |
| `experiments/legacy_branches/` | snapshots of superseded local branches taken during the 2026-09-27 consolidation |
| `experiments/legacy_h100_runtime/` | superseded runtime files from the research attention checkout (`MERGE.json` provenance) |
| `docs/records/release-2.0.0/retired-caches/` | tuned-cache JSON retired at 2.0.0 |
| `docs/records/local-patches-20260917/incompatible-caches/` | caches incompatible with the 2026-09-17 local patches |
| `docs/records/h100-cache-267166/measurements.tar.gz`, `stale-native/` | raw H100 measurement shards and stale native entries of that build |
| `docs/records/archive/` | CPU-built and stale A5000 cache artifacts |
| `diag.py` | one-off MPNN edge-tail debugging snippet |
