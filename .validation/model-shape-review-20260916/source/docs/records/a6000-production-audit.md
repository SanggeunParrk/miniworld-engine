# Final A6000 production audit — 2026-09-11

This is the current A6000 qualification record. It supersedes earlier DiT timing and
build-status claims; older dated records preserve the measurements made at those revisions.
The AdaLN forwarding correction also affects standalone augmented-attention composition;
its older standalone timings have not been remeasured here and are historical.

## Corrected defects

- AugmentedAttention forwards its implementation selection to internal AdaLN. Token and
  ordinary atom DiT now execute the requested composed MiniWorld path.
- Build workers preserve Slurm CUDA_VISIBLE_DEVICES mappings and reject mixed GPU models.
  Incomplete alternative-driver work causes a nonzero exit while valid results are preserved.
- Capture shards carry GPU/compiler provenance. Resume and merge reject foreign/stale work;
  merge checks current kernel source/key identity before ranking measurements.
- Generated dispatch plans use a writable user cache (MINIWORLD_PLAN_CACHE_DIR/XDG override).
  Source identity excludes development notes so wheel and checkout identities agree.
- Tuned-cache storage remains in the installation. Build preflight rejects read-only targets
  before GPU work; use a writable virtual environment or checkout to build/merge caches.
- SciPy is a core dependency because default module initialization needs it. The actual wheel
  constructs an ordinary PyTorch DiT block with optional backends and benchmark imports blocked.
- Exact wheel inventory checks caught nine obsolete modules in stale setuptools build output.
  The clean wheel includes current Python, kernel sources, grids and caches, excludes notebook
  and generated plan history, and is tested outside the checkout. The verified canonical plan
  is bundled and reused from an isolated installation. CI now gates these checks.
- Known-stale A5000 caches were archived with hashes. Their optimized-cache qualification needs
  a fresh build; they were not relabelled or shipped as current results.

## Complete-block measurements

| Block | Mode | PyTorch ms | MiniWorld ms | Speedup |
|---|---|---:|---:|---:|
| Token DiT (384 tokens) | inference | 1.0332 | 0.7199 | 1.435x |
| Token DiT (384 tokens) | training | 18.8426 | 18.1504 | 1.038x |
| Ordinary atom DiT (4096 atoms) | inference | 9.8673 | 4.9644 | 1.988x |
| Ordinary atom DiT (4096 atoms) | training | 173.8015 | 134.7389 | 1.290x |

All rows use one physical A6000, depth 1, BF16 under native mixed-affine policy, TF32 OFF,
mask probability .125, and actual torch.compile. Pair LayerNorm affine stays FP32;
AdaLN condition LayerNorm affine is BF16. For the atom constructor, native dtype parity was checked
for all 23 parameters in both implementations. Inference uses A5 with manual CUDA Graph;
training uses A48 without graphs, forward+backward including input/condition/pair/parameter
gradients and no optimizer. These blocks have no dropout. There are three native timer
samples per process. Torch, NumPy and Python RNGs are seeded; linear weights are nonzero,
with identical parameter digests across implementations. Cache misses abort and tuned-cache
files stay unchanged during measurement. Ordinary atom DiT uses pair-bias attention, not SWA.

## Validation

Current dispatch identity: `13670d32da86126ad0f471b7c08919da7847c64dd4fa13709941796944f19546`.
Required A6000 cache keys: **1008/1008**, no missing/invalid entries.
Alternative driver inputs: **772/772**, each shard contains its named target's actual measurements.
A final standalone build reuses the completed shards and exits successfully after both distributed workers finish.

Independent nonzero DiT accuracy covers token 384 A5/A48 and atom 4096 A5/A48, using
seeds 17/39 and official PyTorch BF16/FP32 references. The initial 224 tensor comparisons
included atom 512 training; an additional 108 comparisons close the full atom 4096 A48
training case. MiniWorld executes full A48 compiled forward+backward. To fit memory, the
reference executes independent A4 slices through the official module, accumulates shared
pair/parameter gradients in FP32, and concatenates single/condition gradients without
changing loss scaling. All comparisons pass the existing 3% relative-L2 band.

For full atom training, maximum gradient relative L2 is 2.279% against BF16 and 1.646%
against FP32; maximum output relative L2 is 0.331%. Every input/parameter gradient is finite and
nonzero. The run reports zero cache misses and no cache writes. Its MiniWorld peak allocation
is 26.51 GiB; the chunked reference is a numerical oracle, not a memory-performance baseline.

Gate summaries:

- cpu: 2696 passed, 31 skipped, 216 deselected, 22 warnings in 194.03s (0:03:14)
- gpu: 226 passed, 5 skipped, 2731 deselected, 79 warnings in 344.91s (0:05:44)
- ruff: All checks passed!
- types: All checks passed!
- dev-audit: stats: {'autotuners': 83, 'registered': 83, 'captured': 0}
- final-cache-cpu: 729 passed in 27.61s
- final_cli: 44 passed; scoped Ruff and ty passed after the final progress-reporting fix
- gpu_supplement: 3 passed: available-device cases previously skipped for dtype/GPU count

The full GPU suite covers numerical, compilation, graph, masking and dropout contracts.
The supplementary run executes the three available A6000 cases skipped in the single-GPU
BF16 suite: two-GPU DDP compilation and the FP32-only numerical/determinism cases. Of
those five skipped cases, only the SM90/SM100-only cases remain unexecuted.
The expanded custom-op suite additionally passes five tests covering 42 operations' schema,
fake metadata, static/dynamic AOT contracts, and the FlashAttention wrapper's registered
backward. It newly checks 16 operations from the earlier unreached list. Other precisions,
alternative paths and GPU backends retain an explicit scope in the
[opcheck coverage record](a6000-opcheck-scope.md); this is not certification of every registration.

The installable artifact is `dist/miniworld_engine-1.0.0-py3-none-any.whl`.
Its hash, inventory, import and construction evidence are embedded in the JSON record.

## Scope and repository layout

Build coverage certifies declared Triton keys and usable cached candidates. It does not prove
an independently optimal tile for every physical workload. Six CuTe tuner families remain
outside the shared builder; H100/B200 native execution and numerical qualification are not
claimed. See [the build command contract](../operations/dispatch-cache.md#what-build-all-verifies).

Read-only wheels can consume shipped caches, but tuned-cache build/merge requires a writable
installation. Automatic approval review rejected a proposed cache overlay because changing
cache selection, merge and publication together could select stale or incorrect kernels.
That change was not executed; the validated alternative preserves cache semantics and fails
at build preflight with an actionable error.

Team-GM scratch artifacts (1,067 entries) live in `.scratch/a6000-2026-09/`, and 166 old
engine-root logs/scripts/plans were preserved in `.scratch/root-history/`. Both relocations
have manifests. Another 32 historical plan files (23.6 MB) were preserved in
`.scratch/production-audit/plan-history/`; the current verified plan occupies the canonical
`src/miniworld_engine/kernels/registry_kernel.csv` and units JSON. Final worklogs are under `.scratch/production-audit/`; curated records stay
in `docs/records/`, runtime assets in `src/`, and the wheel in `dist/`.

[Machine-readable evidence](a6000-production-audit.json).
