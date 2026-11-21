# E5: other public kernels (all operator classes, all candidates; branch `e5-all`)

Every public kernel that serves one of the Pairformer operators, as an external backend of the measurer, next to the PyTorch reference, cuEquivariance and the engine's `cuda`:
one config, one registry (`externals.py`), one collector (`collect.py`), one result file (`E5_results.md`). Nothing under `src/` changes: `externals.install(names)` adds the
backend name to the module classes (class attribute `backends`) and attaches `_supports_<name>` / `_forward_<name>`; a case a library does not serve is recorded `unsupported`
(or `failed` / `skipped` with the error) with the reason, never re-routed.

Scope (all cards): AF3 setup only, L = 512 / 1024 / 2048, eval and train, 20 timed iterations (warmup 5, L2 flush, CUDA-graph timing), 3 repeats (a different job each).

| class | shape (`requirements_e5_all.csv`) | candidates (`--ext` names) |
|---|---|---|
| TriAttn | TriangleAttention starting, d_pair 128, 4 heads of 32 | `ds4sci`, `trifast`, `gnops`, `of3tri`, `megafold`, `bionemo_ir`, `turbopf` (+ context `cueq`) |
| TriMul | TriangleMultiplication outgoing, d_pair 128 | `fast_trimul`, `gpumode`, `geonn` (+ context `cueq`) |
| AttentionPairBias | d_single 384 / 16 heads (float32 and master), d_single 768 / 16 heads x 48 channels (AF3 token DiT; master) | `gnops`, `of3tri`, `megafold` |
| OPM | OuterProductMean d_msa 64, d_pair 128, d_hidden 32, depth 1024 | `bionemo_ir` |
| PWA | MSAPairWeightedAveraging d_msa 64, d_pair 128, 8 heads, depth 1024 | `ligo_pwa`, `bionemo_ir` |
| Transition | d_hidden 128 n 4 (pair), d_hidden 64 n 4 (MSA, depth 384) | `megafold_tr`, `bionemo_ir` |

Not a class here: LayerNorm (no candidate has a standalone LayerNorm op: BioIR / MegaFold fuse it into LN + projection kernels); FlashBias is not applicable (dense pair bias).

## Files

- `externals.py`: the ONE registry. `AVAILABLE` = all backend names, `CLASS_OF` = classes each serves; `install(names)` for `run_measure.py --ext <any>`. Inline: `ds4sci`, `trifast`. TriMul adapters:
  `ext_fast_trimul.py`, `ext_gpumode_trimul.py`, `ext_geonn_trimul.py`. Pairformer-operator adapters: `ext_gnops.py`, `ext_of3tri.py`, `ext_megafold.py`, `ext_megafold_tr.py`, `ext_bionemo_ir.py`,
  `ext_turbopf.py`, `ext_ligo_pwa.py` (shared plumbing `ext_attn_common.py`).
- `requirements_e5_all.csv`, `cfg_e5_all.yaml` (speed), `cfg_e5_all_acc.yaml` (accuracy vs the float64 gold). Row numbers 1..12 are those of the former `requirements_ext_attn.csv` (row 1 = TriAttn as in the
  first E5 run), row 13 = TriMul (it was row 1 of the former `requirements_trimul_e5.csv`; the collector matches on the case content, not on the row number, so every old record still matches).
- `h100.sbatch`, `a100.sbatch`, `rtx5090.sh`: the stage `e5` (`REP=1..3 paper_common/chain.sh 6 E5_other_kernels/h100.sbatch`); `NAME / CFG / KIND / EXT / EXTRA` select other stages, e.g. the accuracy stage
  `NAME=e5acc CFG=cfg_e5_all_acc.yaml KIND=accuracy`. `acc_trimul.py`: the TriMul accuracy harness (gradients included). `collect.py`, `report.py`.
- `third_party/` (repo root): metadata and `fetch.sh` of every upstream (one `geometric-nn-ops` entry serves `geonn` and `gnops`). On a card the TriMul adapters read `$PAPER_EXT_PATH` (default `~/paper-ext`) and
  `third_party/<entry>/upstream`; the Pairformer-operator adapters read `$PAPER_EXT_ATTN/<name>` (default `~/paper-ext-attn`).
- `README_trimul.md`, `ext_attn_results.md`: the notes and per-card results of the two third-party agents (history; their per-record tables are superseded by `E5_results.md`).

## Run

    cd benchmarks/paper
    python paper_common/run_measure.py E5_other_kernels/cfg_e5_all.yaml --kind speed --out E5_other_kernels/records/<card>/e5.r1.jsonl --ext <names> [--module TriangleMultiplication --backend cuda cueq fast_trimul]
    python E5_other_kernels/collect.py --card h100        # -> E5_h100.csv (all record families of the card)
    python E5_other_kernels/report.py                     # -> E5_results.md

## Record families and who owns which cell

`records/<card>/<stage>.r<k>.jsonl` (+ `.meta.json`: engine version, commit, node, settings incl. iterations). One row per (operator class, mode, L, dtype, backend) in `E5_<card>.csv`; the `stack` column says
which family a row comes from and `iterations` how many timed iterations it has.

| family | origin | owns |
|---|---|---|
| `ext` | the first E5 run (H100 campaign branch `paper-results-h100`, A100 `paper-results-a100`, 5090 `paper-results-5090`), 3.0.6 | TriAttn: the context rows `pytorch` / `cuda` / `cueq`, and `ds4sci`, `trifast` |
| `ext_attn` | branch `third-party-attn` | AttentionPairBias, OPM, PWA, Transition (context `pytorch` / `cuda` included: the family's own run) and the TriAttn candidates `gnops`, `of3tri`, `megafold`, `bionemo_ir`, `turbopf` |
| `ext_trimul`, `fill_l2048_train` | branch `third-party-trimul` | TriMul candidates and the TriMul `pytorch` / `cuda` context (the 5090 train L = 2048 fill run has 20 iterations) |
| `e5` | this integration (small runs, see below) | preferred over every other family when present |

When a cell exists in several families the order is: TriAttn `e5`, `ext`, `ext_attn`; TriMul `e5`, `fill_l2048_train`, `ext_trimul`; any other class `e5`, `ext_attn`; within it, a family whose record is `ok` with 20
iterations wins over one with another iteration count, then any `ok`, then the last non-ok record. The records of the old 100-iteration campaign (`old_it100/`) are not used. Exception found in the
inputs: the 5090 `ext_trimul` records were taken with 100 iterations on a wider grid (the third-party agent reused them); the cells of this scope are re-measured here at 20 iterations (`e5`), and the 100-iteration
records would only be used for a cell the new run lacks (the `iterations` column and a note in `E5_results.md` say so).

## What this integration measured newly (small, justified by a missing record)

- TriMul `cueq` (the cuEquivariance TriMul was not in any 20-iteration E5 record) together with `cuda`, 3 repeats, on H100 and A100; on the 5090 all TriMul backends (`cuda`, `cueq`, `fast_trimul`, `gpumode`, `geonn`), 3 repeats, 20 iterations.
- Accuracy against float64 of `ds4sci` and `trifast` on TriAttn (and `cuda` as the engine reference), stage `e5acc`, L = 512, eval and train, on each card (`ds4sci` on the 5090 does not load: speed records say `failed`).

## Corrections found while adding the accuracy records

1. `ds4sci` scaled q twice. The first E5 run's adapter multiplied q by `d^-0.5` before `DS4Sci_EvoformerAttention`, which already applies that scale (float64 reference on random inputs: relative rms error
   0.66 with the pre-scale, 2e-3 (bf16) / 2.5e-4 (fp16) without; module accuracy run: eval 0.19, gradients 0.91, RED, on H100 and A100). `externals.py` is fixed; the accuracy records of the buggy version are kept under
   `records/<card>/ds4sci_double_scale_bug/`, and the re-run is stage `e5ds4sci` (3 repeats).
2. `ds4sci` is not CUDA-graph capturable. DeepSpeed's evoformer_attn kernels launch without a stream argument (attention_cu.cu:118, attention_back.cu:164), i.e. on the legacy default stream, while the measurer captures
   a graph on a side stream: the replay does not contain the attention kernels (and the parameter gradients are 0). EVERY `ds4sci` graph timing (first run and `e5ds4sci`) is therefore invalid (H100 train L = 2048:
   "43 ms" is an empty graph; eager: about 725 ms against the engine's 83 ms). `graph_check.py` (below) found it mechanically and checks all candidates; the corrected values are eager timings.

## CUDA-graph validity check (`graph_check.py`, records `graphcheck*.r1.jsonl`)

For every (backend, case) except `pytorch`: same step as the measurer; (a) two graph replays give the eager output, (b) train: parameter gradients non-zero and of the eager norm, (c) replay time >= half of the eager GPU kernel
time, (d) the largest eager kernels appear in the replay profile (torch.profiler). A failing cell is `invalid_graph` (reason in the CSV; the graph time kept in `ms_graph_invalid`), excluded from the graph columns and tables; its eager
timing (`ms_eager`, CUDA events, same warmup / iterations / L2 flush, compared with the engine's eager time) is the corrected value (`e` in `E5_results.md`). `collect.py` applies it; section 0 of `E5_results.md` lists the verdict per backend and card.
Run: `chain.sh 12 E5_other_kernels/gc_h100.sbatch` (`gc_a100.sbatch`, 5090 through gpu.sh).

## Installing ds4sci / trifast (into a separate path, never the engine env)

    pip install --target $HOME/paper-ext --no-deps trifast platformdirs psutil hjson py-cpuinfo ninja msgpack
    pip install --target $HOME/paper-ext --no-deps deepspeed          # 0.20.0; the op JIT-builds its CUTLASS extension on first use (CUTLASS_PATH = CUTLASS >= 3.1 headers, nvcc, ninja; h100.sbatch sets them)

`$PAPER_EXT_PATH` (default `~/paper-ext`) is appended to `sys.path` after the engine environment, so it never shadows an engine dependency. ds4sci's CUTLASS kernels target sm_80 / sm_90 (A100, H100);
on the RTX 5090 the extension does not load (recorded `failed`). trifast is Triton (any card).

## Reproduce on a card (summary)

    # 1. libraries (once per card; nothing touches the engine environment)
    for d in fast_trimul gpumode_trimul geometric-nn-ops; do third_party/$d/fetch.sh ~/paper-ext/$d-tree; done   # TriMul trees: $PAPER_EXT_PATH or third_party/<d>/upstream (default target)
    for d in openfold3 megafold bionemo-ir ligo-af3 turbopairformer; do DEST=$HOME/paper-ext-attn/$d third_party/$d/fetch.sh; done   # turbopairformer: build_local.sh prebuilds H100 extensions
    # trifast, deepspeed: pip --target $HOME/paper-ext (see above)
    # 2. speed: 3 repeats, a different job each (H100: gc/E5 sbatch through chain.sh; A100 sbatch; 5090 through ~/mwe/gpu.sh)
    cd benchmarks/paper; for r in 1 2 3; do REP=$r paper_common/chain.sh 6 E5_other_kernels/h100.sbatch; done
    # 3. CUDA-graph validity (all candidates), then tables
    paper_common/chain.sh 12 E5_other_kernels/gc_h100.sbatch
    python E5_other_kernels/collect.py --card h100 && python E5_other_kernels/report.py

Record files (`E5_other_kernels/records/<card>/`): `ext.r*` (first E5 run), `ext_attn.r*`, `ext_trimul.r*`, `fill_l2048_train.r1` (the three families merged from the branches), `e5.r*` (this integration: TriMul cueq / 5090 TriMul at 20 iterations), `e5ds4sci.r*`
(ds4sci re-run after the scale fix; graph timing invalid), `e5acc.r*` / `ext_attn_acc.r1` / `acc/ext_trimul_acc.<card>.jsonl` (accuracy against float64), `graphcheck*.r*` (validity records: verdict, graph and eager times,
reasons), `ds4sci_double_scale_bug/`, `graphcheck_v0/`, `graphcheck_v1/` (superseded: records of a buggy accuracy adapter and of two earlier versions of the checker that failed on train cases; not used). `collect.py` writes `E5_<card>.csv`.

## Mixed settings in the tables (stated, not hidden)

- 5090 TriMul: the `ext_trimul` records of the third-party run have 100 iterations on a wider grid; the cells of this scope were re-measured here at 20 iterations (`e5`), so the CSV rows carry `iterations` 20; the
  `fill_l2048_train` run (20 iterations) is the older fill for train L = 2048.
- Context rows (`pytorch`, `cuda`) of a class come from the family named in the owner table above; they are not re-measured in every family, so a candidate / engine ratio mixes two jobs on the same card (same day, same build).
- The A100 jobs of one chain can share a node with other jobs of the same chain (different GPUs).
- Graph validity could not be established for cells where the checker ran out of memory (5090, 32 GB: the checker holds the eager step, the graph pool and the replay copies) or crashed; they stay `ok` with `graph_valid = check error` in the CSV.
- (d) of the validity check (profile) is advisory: CUPTI sometimes omits kernels of a replayed graph (the engine's own APB backward kernel on the 5090 while replay time = eager time and the gradients match); the verdict rests on (a)-(c).
