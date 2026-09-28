# Archived experiments

`experiments/` was removed from the tree in 2.2.0: every result worth keeping now lives in
`src/` (the fastest variant only), and the research runners were history. Nothing is lost —
each capsule is still in git:

| where | contains |
|---|---|
| tag `archive/experiments-20260928` | the last tree with `experiments/`: `transition_fused`, `trimul_b7b12`, `trimul_k1k3_inference`, `token_dit_fused`, `token_dit_overlap`, `token_dit_train` |
| branch `wip/main-20260928` | the Sept 27–28 local-H100/Vast research: `trimul_large_d_vast`, `transition_wide_fusion`, `transition_shapes_vast`, `triattn_baseline_20260927`, `triattn_compare_20260927`, `triattn_local_d128`, `trimul_config_audit_20260927`, `trimul_d128_config_20260927` |
| tag `archive/pre-tidy-20260927` | `trimul_training_v2` (2.0.0 training capsule, 6,625 run files), `legacy_branches`, `legacy_h100_runtime`, retired cache blobs |

```sh
git show archive/experiments-20260928:experiments/transition_fused/README.md
git archive archive/experiments-20260928 experiments/trimul_k1k3_inference | tar -x -C /tmp/restore
git show wip/main-20260928:experiments/trimul_large_d_vast/LATENCY_TABLE.md
```

Where each capsule ended up in `src/`:

| capsule | production code |
|---|---|
| `transition_fused`, `transition_shapes_vast` | `kernels/transition/cuda/fused_sm90a.py`, `fused_wide_sm90a.py` |
| `trimul_k1k3_inference` | `kernels/trimul_inproj/cuda/h100_inference.py` (byte-identical K1/K3 overlay sources) |
| `trimul_b7b12`, `trimul_training_v2` | `kernels/trimul_inproj/cuda/h100_training.py` (D128 B1/B7) |
| `trimul_large_d_vast` | `kernels/trimul_inproj/cuda/h100_wide_training.py` (D256/384/512 training) |
| `token_dit_train` | `kernels/augmented_attention/cuda/` |
| `token_dit_fused`, `token_dit_overlap` | not yet ported (quack-GEMM dependent); see the v2.2.0 pending list in `CHANGELOG.md` |
| `transition_wide_fusion` | not ported (explicit 1.03x D384/512 candidate) |

`kernels/trimul_inproj/cuda/h100_sources/PROVENANCE.json` and the `.cu` provenance headers name
paths inside these capsules; resolve them against the tag or branch above.
