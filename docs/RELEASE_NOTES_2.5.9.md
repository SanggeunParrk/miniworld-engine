# miniworld-engine 2.5.x release candidate (draft, local branch `rc/2.5.x-tri`)

Base: 2.5.8 (`origin/main` d3e071883). H100 only, TriangleAttention and TriangleMultiplication inference. Every change below is
**bit-identical** to the previous default route (module outputs compared with `torch.equal`); no FP8 route, no accuracy change. Module
`version` is not bumped anywhere; only `backend_versions["cuda:h100_sm90"]` (so A100 / RTX 5090 / anthropic / cueq / triton / pytorch tokens stay valid).

## Changes

| # | Change | Bit-identical check | `backend_versions["cuda:h100_sm90"]` | Measured gain |
|---|---|---|---|---|
| 1 | Front Y16 (fp16-exact one-kernel front for d_pair 64 x 32-channel heads beyond L768, `FRONT_Y16`) | `tests/gpu/test_triattn_h100_front_y16.py` (judge tokens, owed 0) | TriangleAttention 8 (AF3 8, RF3 10) | module +8.5..12.4 % at L1024 (64:x:32) |
| 1b | dh16 core (`TA_SW=2`, `TA_N24=1`, `TA_K16=1`: software-pipelined sub-step, PV as m64n24, K in 16 channels) | judge tokens | counted in the same bump (8) | -11.6 % core time at L2048 (microbench); 1-2 % on flat-logit module inputs |
| 2 | `ln_bias_fwd` / `ln_bias_blocks`: the unfused fp16 front (d_pair 128 < C <= 512, e.g. 384 x 12 x 32) makes the pair bias inside the LayerNorm kernel (`mma.sync`, the cuBLAS fp16 bias GEMM bits); `FRONT_LNB` default on; row-blocked path (L >= 2048) uses `ln_bias_blocks` with per-warp rows and no block barrier | `tests/gpu/test_triattn_h100_front_lnb.py` (L384 / 1024 / 2048, both nodes, key mask); front2 `bitid_mod` / `bitid_rows` scripts (starting / ending node, flip, mask, rows blocks 1 and 2): all identical | TriangleAttention 9 (AF3 9, RF3 11; the base class bump) | module +0.5..1.7 % (starting node, L256-1536), within noise at L2048; ending node does not take it |
| 3 | K3 pipeline macros `K3_XSPF`, `K3_XEARLY`, `K3_DESC`, `K3_EPI2` in the D64 / D128 any-L inference K3 (`h100_b200style.K3_EXTRA`, prebuild recipe follows it); bf16 only, no FP8 plane / K1 / cuBLASLt code | `tests/gpu/test_trimul_h100_k3_pipe.py` (D64 / D128, L200 / 768 / 1000, both directions, key mask); route A/B at L256-2048; 18/18 real Protenix / Boltz-2 / AF3 dumps (tiled to L768) | TriangleMultiplication 8, TriangleMultiplicationRF3 4, BidirectionalTriangleMultiplication 5 | K3 alone 1.00-1.13x; module D64 L1024-2048 +2.4..3.3 %, D128 L768-1024 +2.5..2.9 %, D128 L1536-2048 within noise, L <= 512 neutral |

## Not included

* Front2 experiments `MW_F3_CL` / `MW_F3_WGS` (dc0cadfd2) and `MW_F3_EPIPIPE` (c270289d5): default-off, labelled experiments, left out
  (they change `front_fused3_sm90.cu` for no default behaviour).
* K3 variants `K3_WRES`, `K3_MC`, and the other `K3_*` flag-gated experiments of the fp8 tree beyond the four macros above (the macros live in
  `k3.cu` with the commit "K3 pipeline variants", `K3_WRES` is in that file but default off and unused).
* Everything FP8 (planes, K1 e4m3, cuBLASLt fast accumulation): goes to 3.0.0.
* TriMul on H100 stays bf16 only.

## Judge (version-based tokens)

* H100: 668 cases owed after the bumps (cuda 7->8: 178, 8->9: 310, 10->11: 60, 3->4: 60, 4->5: 60); remeasured with `tools/slurm/cycle.sh`
  (2 cycles, 2712 tokens issued), **owed 0**.
* A100 and RTX 5090: owed 0 (unchanged; only `cuda:h100_sm90` versions moved).

## Known pre-existing failures (identical on `origin/main` d3e071883)

* `tests/cpu/test_requirements.py::test_class_versions_are_human[...]`: 30 cases (cuda:5090_sm120 versions from the RTX 5090 port).
* `ruff check`: 4 errors; `pyright`: 9 errors (bias_only_dit_sm80.py, _sm120_builds.py, tests/gpu/test_transition_sm80_rows.py). Not touched here.

## Open

* The experimental commit history on this branch includes the interrupted front2 agent's uncommitted `ln_bias_blocks` work, now verified.
* `origin/ln-speed-1.05` (LayerNorm row kernel, A100 / 5090 tokens) merges cleanly into this branch (see session report); no file is touched by both
  (checked with `git merge-tree`, no conflicts).
