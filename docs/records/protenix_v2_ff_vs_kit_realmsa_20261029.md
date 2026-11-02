# Protenix-v2: FoldForge on the 2.4.0 candidate engine against the Anthropic kit (fast), the 5I28 benchmark (2026-10-29)

The 5I28 benchmark as `benchmarks/5i28.md` defines it: 5I28 x1-x16, the real MSA and 4 distinct real templates on both sides.

Supersedes:
- the comparison numbers of `protenix_v2_kit_fwd_aligned_20261029.md`, whose kit ran without the MSA and the templates;
- a first real-MSA run of this day. That run fed the kit a 4-hit a3m holding only 2 distinct sequences: the 4 hits were 5syd_A, 5syd_B,
  6kol_A and 6l9s_A, and the chains of one entry share a sequence. Protenix keeps one hit per sequence, so the kit ran 2 real
  templates and 2 empty slots, and its `template_dedupe` lever evaluated the 2 identical empty slots once: 3 template stacks per
  recycle. FoldForge ran the same 4 hits without deduplication: 4 stacks over 2 distinct structures. Logs:
  `~/miniworld-engine-scratch/kitv2real/logs/run1_4hits/`.

Timing definition (both sides):
- One complete model forward (trunk passes + diffusion sampler + confidence head), CUDA-synchronized before and after.
- warm = median of the warm forwards.
- FoldForge reports `model_seconds_warm`; the kit reports `PHASE ... fwd_s`. The kit's `Model forward time` line also covers
  post-forward handling and is not used.

Inputs (identical sequence, 4 recycles, 200 steps, 5 samples, bf16, one H100 80GB HBM3), job 39183:
- MSA: `~/data/BioMolDB_20260224/a3m_16k.lmdb` key P0113359, 6790 rows (both logs: `msa 6790` / `N_msa 6790`).
- Templates: 5SYD_A, 6KOL_A, 5YT7_A, 6TFD_A.
  - FoldForge: `template.lmdb` key P0113359, `template_n: 4`. The LMDB holds 20 records, 6 distinct sequences; `load_templates` now
    keeps one per hit sequence, as AF3 and Protenix do, and loads those 4.
  - Kit: all 20 hits in one a3m (`kitv2real/in/5i28_templates_hits20.a3m`, mmCIFs from `~/data/cif_raw`). Its log shows
    `hits=20 real=4/4 dropped=0` and `Found 4 templates` for every n.
- FoldForge:
  - Code: `~/FoldForge` (with the template dedupe) on the engine working tree f92f9280 + the 2.4.0 changes. The H100 native caches
    were re-measured in the warm CUDA-graph regime.
  - Run: compile + CUDA graph + bucketing, scope trunk, 0 compiles in the timed runs, 3 warm forwards.
- Kit:
  - Command: `protenix_v2 run.sh pred --config h100 --mode fast` (`ACTIVE mode=fast`).
  - Seeds 0-3 in one process, warm = seeds 1-3.

## Result (seconds, warm medians)

| N | tokens | kit fast `fwd_s` | FoldForge | FoldForge / kit |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.666 | 0.465 | 0.70 |
| 2 | 256 | 1.015 | 0.765 | 0.75 |
| 4 | 512 | 2.242 | 1.918 | 0.86 |
| 8 | 1024 | 7.519 | 6.829 | 0.91 |
| 12 | 1536 | 17.719 | 17.211 | 0.97 |
| 16 | 2048 | 41.716 | 32.980 | 0.79 |

FoldForge's warm forwards spread under 0.5 % at every n (128 tokens: 0.462-0.465 s). FoldForge is faster at every size; the margin is
smallest at 1536 tokens.
