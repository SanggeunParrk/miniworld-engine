# miniworld-engine 2.5.12

Base: 2.5.11 (fe5dbb5a4), release candidate `rc/bidir-2.5.12`. H100 only, the bidirectional modules: (1) `BidirectionalTriangleAttention` (and the single-direction bias-only
`TriangleAttention(use_self_attention=False)`, which runs on the same kernels): host tails fused into kernels, **bit-identical**; (2)
`BidirectionalTriangleMultiplication` D64 / D128 inference: measured cuBLASLt contraction table (**bit-identical**), the port K3 at D128
(**not bit-identical, inside the engine accuracy standard**), and the launch-table cells of calls without a mask (**bit-identical**). Module
`version` is not bumped anywhere; only `BidirectionalTriangleMultiplication.backend_versions["cuda:h100_sm90"]` 5 -> 6. The TriMul launch
table `kernels/trimul_inproj/configs/h100_sm90.json` changes (version 4 -> 6), which the judge counts for every class launching a
`trimul_inproj` kernel on the H100 (re-measured, see Judge). A100 / RTX 5090: no version or token change.

## Changes

| # | Change | Outputs | `backend_versions["cuda:h100_sm90"]` | Measured (CUDA-graph replay, same process, cool) |
|---|---|---|---|---|
| 1 | Bidir TriAttn inference: one-pass softmax-logits kernel (widen / transpose / mask), planes GEMM `[Wv; Wg; Wb] y^T` through the fastest bit-equal cuBLASLt algorithm (calibrated once per shape against `torch.mm`, outside capture; skipped when L L > 2^20), fused tail (gate + out-projection + residual in one mma.sync kernel, 16-channel chunks in a 4-deep cp.async ring), fp32 -> bf16 weight cast + concat in one launch (`triattn_bidir_sm90.cu`, `integrations/bidir_triattn_h100.py`) | bit-identical | - | d_pair 128, L128-768: 0.73-0.83x |
| 2 | Bidir TriAttn training: the fused tail also writes the saved u; du GEMM fused into the gate backward (`gate_bwd_fused`), the softmax gradient's last step in one kernel (`softmax_bwd_planes`), fp32 dyln GEMM written in place | bit-identical (output, dx, every parameter gradient) | - | fwd + bwd L128-768: 0.84-0.90x |
| 3 | Bidir TriMul D64 / D128 inference: the two contractions through `_h100_bmm.contract` with the measured cuBLASLt table `_h100_bmm_lt_bidir.LT_BIDIR` (178 entries, algorithm 66, no split-K, each entry bitwise equal to the default pick), passed by the bidirectional call only (`contract(..., table=LT_BIDIR)`; the single-direction contractions keep `LT_CONFIGS`) | bit-identical | - | part of row 4 / 5 below (D64 L1024-1536 0.97-0.98x alone) |
| 4 | Bidir TriMul D128 inference, L >= 160: the port K3 (`h100_inference.PORT_K3`, default on; `MINIWORLD_TRIMUL_BIDIR_PORT_K3=0` switches it off): the packaged K1 in its save-statistics l2 form (5 + 5 new K1 instances in `tmn_sm90_unit.cuh`, masked / unmasked), the contractions, then `h100_b200style.rowpad_k3` (LayerNorms folded into the weights, LN_in statistics from K1) | **changed**, within the standard (below) | 5 -> 6 | D128 L256-2048 0.95-0.99x |
| 5 | TriMul launch table: the 36 `trimul_fwd_sm90_cuda` cells of the bidirectional D64 / D128 route without a mask (`DIR=0, MASK=0`, B 1 / 2, L128-2048; no MASK=0 key existed, every no-mask call ran the kernel default and warned "no table entry"); candidates from `kernels/_pretune.py`, ranked by an in-graph A/B of the module (the pretuner's eager ranking was up to 4 % slower than the default in graph replay) | bit-identical (every candidate equals the default's bits) | - (table version 4 -> 6) | L128 0.87-0.94x, L2048 0.98x (D64), else within +-0.5 % |
| 6 | Left out of the merged history: `BIDIR_RESIDUAL_LT` (opt-in cuBLASLt residual out-projection: not the same bits, no default use), the bidir-k3gap host memo / diag macros (`TMN_K1_DIAG`, `TMN_K3_DIAG`, `TMN_DEV_ABL`) / k1b / k1redesign / ILV / PP / SW history. `tmn_kernels.cuh` is byte-identical to 2.5.11 (the 8 anchor-patch chains of the wide-training, b200style backward d256 / d384 and dpl families untouched) | - | - | - |

## Accuracy (engine standard, `measure/accuracy.py`: per tensor max|x - gold| / max|gold| vs fp64 PyTorch, tol = max(err_pytorch, 8e-3))

Port K3 (row 4), registry row 95 (`BidirectionalTriangleMultiplication` d_pair 128, every length, eval + train), port K3 on vs off: no
light changes (eval L128-768 GREEN on and off; training does not reach the route, its 10 GREEN / 2 YELLOW records are the same as before).
Eval max error on -> off: L256 4.28e-3 / 4.65e-3, L384 4.94e-3 / 4.83e-3, L512 4.94e-3 / 4.65e-3, L640 4.58e-3 / 4.94e-3, L768 4.76e-3 /
5.03e-3 (L128 below the minimum length: same route). Wider check (registry inputs, fp64 gold, per tensor; "vs 2.5.11" = against the 2.5.11
route's output):

| case (D128, eval) | max / rms vs fp64, RC | max / rms vs fp64, 2.5.11 | RC vs 2.5.11 max / rms | light |
|---|---|---|---|---|
| L160 key mask | 4.69e-3 / 2.86e-3 | 4.74e-3 / 2.91e-3 | 4.90e-3 / 2.47e-3 | GREEN |
| L256 key mask | 4.28e-3 / 2.84e-3 | 4.65e-3 / 2.89e-3 | 4.74e-3 / 2.47e-3 | GREEN |
| L300 key mask | 4.97e-3 / 2.83e-3 | 5.16e-3 / 2.88e-3 | 4.67e-3 / 2.47e-3 | GREEN |
| L512 key mask | 4.94e-3 / 2.79e-3 | 4.65e-3 / 2.84e-3 | 4.31e-3 / 2.47e-3 | GREEN |
| L512 B2 key mask | 4.94e-3 / 2.79e-3 | 5.14e-3 / 2.84e-3 | 4.31e-3 / 2.47e-3 | GREEN |
| L768 key mask | 4.76e-3 / 2.75e-3 | 5.03e-3 / 2.80e-3 | 4.55e-3 / 2.47e-3 | GREEN |
| L1024 key mask | 4.72e-3 / 2.72e-3 | 4.83e-3 / 2.77e-3 | 4.39e-3 / 2.47e-3 | GREEN |
| L1536 key mask | 4.40e-3 / 2.68e-3 | 4.60e-3 / 2.73e-3 | 4.20e-3 / 2.47e-3 | GREEN |
| L256 no mask | 4.85e-3 / 3.04e-3 | 5.07e-3 / 3.09e-3 | 4.67e-3 / 2.69e-3 | GREEN |
| L512 B2 no mask | 4.42e-3 / 2.97e-3 | 5.00e-3 / 3.03e-3 | 4.35e-3 / 2.68e-3 | GREEN |
| L768 no mask | 4.67e-3 / 2.93e-3 | 5.01e-3 / 2.99e-3 | 4.51e-3 / 2.68e-3 | GREEN |
| L1536 no mask | 4.30e-3 / 2.84e-3 | 4.79e-3 / 2.91e-3 | 4.18e-3 / 2.68e-3 | GREEN |
| L2048 no mask | 4.48e-3 / 2.81e-3 | 4.84e-3 / 2.87e-3 | 4.31e-3 / 2.68e-3 | GREEN |

rms error lower in 13 / 13, max error lower in 12 / 13 (L512: 4.94e-3 against 4.65e-3, 0.62 of tol). The difference to 2.5.11 is ~1 bf16
ulp of the output (2.5e-3 rms relative).

## Bit-identity checks (H100, the RC tree vs the 2.5.11 tree fe5dbb5a4 in separate processes, sha1 per tensor)

85 cases (registry rows, own masks or none, B1 / 2; training: output, input and every parameter gradient): single-direction TriMul inference
D64-D512 and training D64-D512, single-direction TriAttn (q.k: 128:4:32, 64:2:32; bias-only both nodes, eval + train), OuterProductMean (AF3 /
Boltz order), bidirectional TriAttn (L128-384, B1 / 2, mask / none, eval + train), bidirectional TriMul D64 (L256-2048, mask / none) and
D256-D512 (eval + train), D128 training and L128 inference: **72 / 72 identical**. The 13 D128 inference cases at L >= 160 differ (port K3,
table above). The in-process "main" route used for the speed table (RC tree with its switches at the 2.5.11 route) equals the 2.5.11 tree's
bytes on all 85 cases.

## Speed (main = 2.5.11 route, same process, CUDA-graph replay)

H100 80GB HBM3, **power-capped at 700 W**: sustained load throttles the SM clock 15-18 %, so cool and sustained are reported separately and
are not comparable to the benchmark pages. Both routes in one process, the same module and inputs; "main" = the RC tree with its switches at
the 2.5.11 route (bidir TriAttn `LEGACY` torch ops, no `LT_BIDIR`, port K3 off, the 2.5.11 launch table), checked bitwise against the 2.5.11
tree. cool = median of 4 rounds of (2 s idle, ~20 ms of replays), order alternating; sustained = 2 s idle, 1 s burn, mean over the next
0.5 s. fp32 master weights, bf16 pair, no mask, B1. ms.

| Module | L | main cool | RC cool | ratio cool | main sus | RC sus | ratio sus |
|---|---|---|---|---|---|---|---|
| BidirTriAttn 128, infer | 128 | 0.0705 | 0.0583 | 0.827 | 0.0712 | 0.0587 | 0.825 |
| | 256 | 0.1874 | 0.1444 | 0.770 | 0.1881 | 0.1446 | 0.769 |
| | 384 | 0.3934 | 0.2943 | 0.748 | 0.3952 | 0.2958 | 0.749 |
| | 512 | 0.6632 | 0.4824 | 0.727 | 0.6649 | 0.4922 | 0.740 |
| | 640 | 1.0324 | 0.7584 | 0.735 | 1.0345 | 0.7771 | 0.751 |
| | 768 | 1.5114 | 1.1243 | 0.744 | 1.5157 | 1.1676 | 0.770 |
| BidirTriAttn 128, train fwd+bwd | 128 | 0.2435 | 0.2198 | 0.903 | 0.2423 | 0.2172 | 0.896 |
| | 256 | 0.5866 | 0.5096 | 0.869 | 0.5886 | 0.5124 | 0.871 |
| | 384 | 1.1537 | 0.9907 | 0.859 | 1.1556 | 0.9932 | 0.860 |
| | 512 | 1.9284 | 1.6241 | 0.842 | 1.9297 | 1.6374 | 0.849 |
| | 640 | 3.0577 | 2.6314 | 0.861 | 3.0628 | 2.6486 | 0.865 |
| | 768 | 4.4225 | 3.7880 | 0.857 | 4.4482 | 3.8370 | 0.863 |
| BidirTriMul D64 infer | 256 | 0.0541 | 0.0542 | 1.002 | 0.0542 | 0.0542 | 1.000 |
| BidirTriMul D64 infer | 512 | 0.1997 | 0.1991 | 0.997 | 0.2056 | 0.2052 | 0.998 |
| BidirTriMul D64 infer | 768 | 0.4512 | 0.4461 | 0.989 | 0.4761 | 0.4636 | 0.974 |
| BidirTriMul D64 infer | 1024 | 0.8607 | 0.8332 | 0.968 | 0.9189 | 0.8943 | 0.973 |
| BidirTriMul D64 infer | 1536 | 2.2726 | 2.2586 | 0.994 | 2.4583 | 2.4156 | 0.983 |
| BidirTriMul D64 infer | 2048 | 4.6862 | 4.5973 | 0.981 | 5.0659 | 4.9433 | 0.976 |
| BidirTriMul D128 infer | 256 | 0.1165 | 0.1149 | 0.986 | 0.1185 | 0.1186 | 1.001 |
| BidirTriMul D128 infer | 512 | 0.4010 | 0.3928 | 0.980 | 0.4352 | 0.4287 | 0.985 |
| BidirTriMul D128 infer | 768 | 0.9024 | 0.8715 | 0.966 | 1.0381 | 1.0139 | 0.977 |
| BidirTriMul D128 infer | 1024 | 1.7345 | 1.6528 | 0.953 | 1.9609 | 1.9063 | 0.972 |
| BidirTriMul D128 infer | 1536 | 4.5360 | 4.3955 | 0.969 | 5.2545 | 5.1064 | 0.972 |
| BidirTriMul D128 infer | 2048 | 9.3239 | 9.1735 | 0.984 | 10.7073 | 10.4226 | 0.973 |
| BidirTriMul D256 infer (unchanged route: control) | 256 | 0.2606 | 0.2604 | 0.999 | 0.2759 | 0.2761 | 1.001 |
| BidirTriMul D256 infer (unchanged route: control) | 512 | 0.9448 | 0.9453 | 1.000 | 1.0588 | 1.0582 | 0.999 |
| BidirTriMul D256 infer (unchanged route: control) | 768 | 2.1540 | 2.1536 | 1.000 | 2.4103 | 2.4275 | 1.007 |
| BidirTriMul D256 infer (unchanged route: control) | 1024 | 4.0823 | 4.0823 | 1.000 | 4.5894 | 4.6985 | 1.024 |
| BidirTriMul D256 infer (unchanged route: control) | 1536 | 10.5218 | 10.5312 | 1.001 | 12.1284 | 12.0910 | 0.997 |
| BidirTriMul D256 infer (unchanged route: control) | 2048 | 21.1296 | 21.1333 | 1.000 | 23.9397 | 24.0111 | 1.003 |
| BidirTriMul D128 train | 256 | 0.4932 | 0.4941 | 1.002 | 0.4972 | 0.4972 | 1.000 |
| BidirTriMul D128 train | 384 | 0.9530 | 0.9533 | 1.000 | 0.9958 | 0.9954 | 1.000 |
| BidirTriMul D128 train | 512 | 1.6410 | 1.6450 | 1.002 | 1.7594 | 1.7576 | 0.999 |
| BidirTriMul D128 train | 768 | 3.6335 | 3.6240 | 0.997 | 4.0226 | 4.0572 | 1.009 |


D256 (unchanged code) is the protocol control: cool 0.999-1.001, sustained 0.997-1.024 (the spread of the sustained figure under the power cap). TriMul D128 training is not on any changed route: 0.997-1.002 cool.

## Judge (version-based tokens)

* Requirements CSV: three new `*_sm90` rows for the bidir TriAttn kernels (`triangle_attention_bidir_logits_sm90`, `_tail_sm90` forward,
  `_gate_bwd_fused_sm90` backward). Kernels entering the list owe nothing by the judge's rule: `judge owed` after the TriAttn commits = 0 on
  every card (BidirectionalTriangleAttention and TriangleAttention tokens carried; their outputs are bit-identical).
* H100 owed after the bumps: 890 accuracy + 890 speed cases, all `table:trimul_inproj 4 -> 6` (every class launching a `trimul_inproj`
  kernel, cuda and triton backends: TriangleMultiplication, TriangleMultiplicationRF3, BidirectionalTriangleMultiplication, PairformerBlock,
  Transition), 60 + 60 of them also `adapter:cuda 5 -> 6` (BidirectionalTriangleMultiplication). Re-measured with `tools/slurm/cycle.sh`
  (one cycle, 6 groups, 2670 tokens issued). A first cycle at table version 5 was superseded by the graph re-rank of change 5.
* Accuracy lights old -> new, all re-measured cells: GREEN -> GREEN 1185, YELLOW -> YELLOW 117, skipped -> skipped 33; **no light changed,
  RED 0**. Errors changed on the port K3 cells (BidirectionalTriangleMultiplication d_pair 128 eval cuda, L256-768: new / old 0.92-1.06,
  all GREEN) and by <= 0.1 % on Triton training cells (run-to-run atomics). Speed (speed-up vs PyTorch, new / old): median 1.000 in every
  class (Bidir TriMul range 0.93-1.04: the judge's cases are L <= 768 with a mask, where the gain is 1-3 %, inside the token spread).
* `judge owed` after the merge: H100, A100, RTX 5090 accuracy and speed owed 0, pending 0.

## Tests and checks

* GPU, one 2-GPU work-queue job (one file per free GPU), 12 items: `test_bidir_triattn_h100.py` (254 passed, 24 skipped),
  `test_trimul_bidir_contraction_port_k3.py` (23), `test_accuracy_smoke.py -k "Triangle or trimul or triattn or TriMul"` (34),
  `test_determinism.py` (same filter, 8), `test_trimul_h100_k3_pipe.py` (12), `test_trimul_inference_workspace.py` (23),
  `test_bias_only_mask_skip.py` (9), `test_weight_cast_cache_h100.py` (12), `test_h100_aux_int64.py` (23), `test_h100_drop_scale.py` (31):
  pass. `test_trimul_big_plane.py`: 3 pass, `test_triton_training_past_2_31_elements[False-14]` (Triton backend, untouched) out of memory in
  the queue; it passes on its own on both 2.5.11 and this branch (as in 2.5.10).
* CPU: `test_requirements` (1167), `test_prebuild_plan` (4), `test_judge` (20), `test_modules_cpu` (497), `test_imports` (922) pass. `ruff
  check`, `ruff format --check`, `pyright` clean on the 10 touched Python files. `_prebuild --check`: no new finding vs 2.5.11 on sm90a /
  sm80 / sm120 (the renamed `gate_bwd_fused_k` makes the new CSV row match its kernel).
* Kernels from a copy of the 2.5.11 release prebuild cache (content-keyed; changed artifacts built once), not the stale kernels wheel of
  the shared environment.

## Packaging: prebuilt artifacts whose keys change

Plan vs 2.5.11 (`_prebuild --list`, same artifact counts: sm90 513, sm80 72, sm120 64; `--check`: no new finding on any arch, 0 not built in
the RC cache):

* **sm90** (`miniworld-engine-kernels-sm90-cu129`): `triattn_h100_bidir` -> **`triattn_h100_bidir2`** (new name: the extension now links
  cuBLASLt; the 2.5.11 name is not carried, the 2.5.11 engine pins its own wheel) and the 7 `trimul_h100/generated/unit` cubins
  (`tmn_sm90_unit.cuh`: the save-statistics K1 instances of the port K3) re-keyed. Everything else keeps its key.
* **sm80** (`-sm80-cu129`): `triattn_bidir_sm80` re-keyed (`triattn_bidir_sm90.cu` is compiled for sm_80 too; its new kernels are not
  launched there: the A100 path keeps the torch ops, so outputs and tokens are unchanged). Built (nvcc, CPU node): OK.
* **sm120** (`-sm120-cu129`): `triattn_bidir_sm120` re-keyed, same reason. Built: OK.

## Release build (2.5.12)

* Incremental prebuild from the 2.5.11 release cache (GCC 12.4 host compiler): sm90 8 built (`triattn_h100_bidir2` + the 7 re-keyed
  `trimul_h100/generated/unit` cubins), sm80 1 (`triattn_bidir_sm80`), sm120 1 (`triattn_bidir_sm120`); 0 failed, 0 absent. Artifacts:
  sm90 513, sm80 72, sm120 64 (as 2.5.11).
* Manifest names vs the 2.5.11 upload: sm90 `triattn_h100_bidir` -> `triattn_h100_bidir2` (the only name change); sm80 / sm120 names
  unchanged (re-keys only). Carried byte-for-byte as before: `triattn_h100_core_h16_v2` (sm90, from 2.5.5).
* Installed-wheel check (the four artifacts unpacked, no repo / CUTLASS / nvcc, empty JIT cache): 513 / 513 sm90 artifacts resolved, 0
  compiled (also with the system libstdc++ preloaded); capture API and init presets OK; `test_trimul_bidir_contraction_port_k3.py` 23
  passed, `test_bidir_triattn_h100.py` 254 passed / 24 skipped, `test_restored_triton_backends.py` 63 passed, 0 builds; accuracy smoke
  (TriangleAttention, TriangleMultiplication, BidirectionalTriangleAttention, BidirectionalTriangleMultiplication) RED 0, 0 builds.
* PyPI sha256 (= `dist/SHA256SUMS`): engine wheel `8db14a95...`, sdist `aa3fcbf6...`, sm90 `67a6dd29...`, sm80 `597957b1...`, sm120
  `44fc8688...`.
