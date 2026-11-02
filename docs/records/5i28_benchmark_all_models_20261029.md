# The 5I28 benchmark: every model, FoldForge on the 2.4.0 candidate engine against the Anthropic kits (2026-10-29)

The protocol is in `benchmarks/5i28.md`:
- 5I28 x1-x16, one H100 80GB HBM3;
- the real MSA and 4 distinct real templates on both sides;
- one complete model forward, CUDA-synchronized;
- warm = median of the warm forwards.

The ratio is kit / FoldForge (above 1 = FoldForge faster). Protenix-v2 is in `protenix_v2_ff_vs_kit_realmsa_20261029.md` (same protocol).

Harness: `~/miniworld-engine-scratch/bench5i28/`:
- one `run.sh N` per kit;
- `ff/run.sh`, `ff/run_rf3.sh` for FoldForge;
- `queue.sh`, an N-GPU work queue;
- `collect.sh`, which reads every number below straight from the kit logs and the FoldForge run JSONs;
- the output of `collect.sh` is in `final_table.txt`.

Slurm jobs: 39192, 39204, 39205, 39214, 39216, plus one srun for FoldForge RF3 x1.

## Inputs (both sides)

- **Sampling:** 200 steps, 5 samples, bf16.
- **Trunk passes:** 4 everywhere, checked per run against the kit:
  - Protenix v1, OpenDDE: `-c 4`.
  - AF3-JAX and AF3-Torch, Boltz-2, Chai-1, OpenFold3: `num_recycles 3`, i.e. 3 + 1.
  - RF3: `n_recycles 4`, which counts total passes.
  - ESMFold2: `num_loops 3`, which runs `max(1, n_loops + 1)` = 4 steps.
- **MSA:** LMDB key P0113359, 6789 rows plus the query (6790).
- **Templates:** 5SYD_A, 6KOL_A, 5YT7_A, 6TFD_A.
  - FoldForge `load_templates` keeps one hit per sequence, as AF3 and Protenix do.
  - The kits take the same 4 in their own input forms. Every kit log shows 4 real templates.
  - RF3 takes one coordinate set per chain: 5SYD_A on both sides. FoldForge's RF3 family turns input templates off by default; it was turned on for this run.
  - ESMFold2 has no template input.
- **Kit modes:** fast mode everywhere, cu130 container. Kit timers:
  - Protenix v1, OpenDDE, OpenFold3: `PHASE total_s`.
  - Boltz-2: `PHASE fwd_s`.
  - AF3-Torch: `ITEM forward_s`.
  - AF3-JAX: the per-seed inference time.
  - Chai-1: a forward-only span; the kit's FORWARD includes CIF writing.
  - RF3, ESMFold2: a synchronized timer around the model forward.

## Result (seconds, warm medians)

| model | n=1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| Protenix v1 kit / FF | 1.138 / 0.422 | 1.315 / 0.604 | 2.191 / 1.288 | 6.262 / 3.922 | 14.476 / 9.117 | 31.658 / 17.751 |
| kit/FF | **2.70** | **2.18** | **1.70** | **1.60** | **1.59** | **1.78** |
| OpenDDE kit / FF | 4.321 / 0.645 | 3.770 / 1.277 | 7.077 / 3.578 | 23.840 / 13.379 | 59.233 / 34.161 | OOM / OOM |
| kit/FF | **6.70** | **2.95** | **1.98** | **1.78** | **1.73** | - |
| ESMFold2 kit / FF | 0.308 / 0.254 | 0.564 / 0.443 | 1.503 / 1.052 | 3.878 / 3.280 | 9.069 / 7.237 | OOM / 12.835 |
| kit/FF | **1.21** | **1.27** | **1.43** | **1.18** | **1.25** | - |
| AF3-JAX kit / FF | 1.040 / 0.511 | 1.570 / 0.695 | 2.900 / 1.381 | 7.465 / 3.982 | 15.120 / 9.215 | 27.360 / 17.620 |
| kit/FF | **2.04** | **2.26** | **2.10** | **1.87** | **1.64** | **1.55** |
| AF3-Torch kit / FF | 1.044 / 0.511 | 1.304 / 0.695 | 1.952 / 1.381 | 5.400 / 3.982 | 11.753 / 9.215 | 22.131 / 17.620 |
| kit/FF | **2.04** | **1.88** | **1.41** | **1.36** | **1.28** | **1.26** |
| Boltz-2 kit / FF | 0.694 / 0.441 | 1.183 / 0.716 | 2.691 / 1.590 | 8.152 / 5.066 | 17.906* / 12.027 | OOM / 23.049 |
| kit/FF | **1.57** | **1.65** | **1.69** | **1.61** | **1.49** | - |
| Chai-1 kit / FF | 3.855 / 0.486 | 3.861 / 0.922 | 8.315 / 2.614 | 22.150 / 11.133 | 43.848* / 29.763 | 74.383* / 62.686 |
| kit/FF | **7.93** | **4.19** | **3.18** | **1.99** | **1.47** | **1.19** |
| OpenFold3 kit / FF | 0.877 / 0.550 | 1.351 / 0.700 | 2.364 / 1.401 | 6.098 / 4.002 | 13.096 / 9.330 | 24.172 / 17.794 |
| kit/FF | **1.59** | **1.93** | **1.69** | **1.52** | **1.40** | **1.36** |
| RF3 kit / FF | 2.275 / 0.457 | 3.397 / 0.691 | 4.537 / 1.414 | 9.122 / 4.171 | 17.301 / 9.529 | 29.636 / 18.011 |
| kit/FF | **4.98** | **4.92** | **3.21** | **2.19** | **1.82** | **1.65** |

`*` the kit ran one warm forward at this size. Several kits run two warm forwards at n >= 2 (Boltz-2, OpenFold3, RF3, AF3-Torch n >= 8);
there the median is their mean.

FoldForge is faster than every kit at every size both ran. OOM:
- OpenDDE n=16 on both sides.
- ESMFold2 kit n=16 (20 GiB allocation).
- Boltz-2 kit n=16 (upstream `predict_step` caught the OOM).
- FoldForge OpenDDE n=12 first OOMed on fragmentation (11 GiB reserved but unallocated). Its re-run on a fresh GPU fitted.

## FoldForge code and engine

**Engine:** the working tree on f92f9280 plus the 2.4.0 changes (`release2313/cachewt`), with the H100 native caches re-measured in the
warm CUDA-graph regime.

**Prebuilt bundle:**
- Early runs used the 2.3.12 bundle. Kernels missing from it were compiled once, before the cold forward: OpenDDE n=1 and Boltz-2 n=1/2.
- Later runs used the 2.4.0 candidate bundle `release240/prebuilt`.
- Timed forwards compiled nothing.

**FoldForge tree, by family.** Protenix v1, OpenDDE, ESMFold2, Boltz-2 and OpenFold3 ran on `~/FoldForge` (working tree with another session's
uncommitted changes) plus the template dedupe. AF3, Chai-1 and RF3 ran on a HEAD worktree (b876f39) plus the dedupe, because the
working tree's load-time size gate refuses the on-disk AF3 and Chai-1 blobs. Those blobs are the same files the earlier comparisons used,
from another converter version. In that worktree only:
- the AF3 and Chai-1 lock sizes match the on-disk blobs;
- the RF3 family's `use_input_templates` is on;
- RF3's template distogram edges are built on the device. Building them on the host copies a host tensor in during CUDA graph capture.
  RF3 templates had never been exercised.

**Trunk passes in HEAD:**
- HEAD's AF3 counts additional passes, so it ran recycles 3 for 4 passes.
- HEAD's run JSON reports `trunk_passes` = recycles + 1 for every family. Chai-1 and RF3 therefore show 5, but run 4: their recycles
  count total passes.

**Discarded runs:**
- Runs with the wrong trunk-pass count: ESMFold2 2 passes (recycles 2, as the u1029 runs), AF3 5 passes.
- FoldForge RF3 runs without the template.

## Kit-side differences in the inputs (as the kits take them)

- **OpenFold3:** re-aligns the CIF-direct templates with kalign and drops 6TFD_A, a 423-residue fusion chain (seq-id x coverage 0.085 < 0.1).
  The 6tfd mmCIF was trimmed to its resolved hit range (label 26-134); then all 4 slots are real.
- **Boltz-2:** aligns templates gap-free. Its coverage is 5syd 86, 6kol 49, 5yt7 114 and 6tfd 11 residues, against FoldForge's
  88 / 120 / 114 / 101. Same slot count.
- **Chai-1:** re-aligns with kalign. 6TFD maps 127 query positions; FoldForge maps 101.
- **Exit codes:** the Chai-1 kit ends with exit 1, and the Boltz-2 kit n=12 with exit 3. In both cases the kit's gate refuses after the
  forwards completed, so the timings stand.
