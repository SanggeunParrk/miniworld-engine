# H100 Transition against anthropic, 2026-10-15

Measurement record (not edited after the fact). Final kernels of the worktree `wt-h100-transition2` (branch `h100/transition-n2`, uncommitted),
H100 SXM 80GB, driver 575.51.03, torch 2.13.0+cu129, CUDA 12.9, `bf16`, inference, module path `Transition(D, n, implementation="miniworld")`,
CUDA graph of one call replayed (graph-launch cost, about 1.1 us, is inside every number on both sides). anthropic through
`integrations/anthropic.py`, `MINIWORLD_ANTHROPIC_ROOT=/home/psk6950/ext/uplifting-biomolecular-modeling`, weights bf16.

## Why this record supersedes the 2026-10-14 D128 numbers

The 2026-10-14 D128 timings of the fused forward came from a kernel whose fragment-LayerNorm path was wrong (fp64 relative error 7.9e-3 at n = 2,
1.2e-2 at n = 4; generic 1.7e-3). ptxas assigned the loop-carried RS-wgmma A registers (`af[4..7]`, R168 .. R181 in the SASS) to the swiglu temporaries
inside the rolled hidden-chunk loop; the data were right (variant: same fragments written to the xn tile and fed through the SS path gave 1.1e-3) and
`fence_regs` did not help. Unrolling the chunk loop fixes it. `tests/numerics/test_transition_fused_sm90a_gpu.py::
test_inference_matches_fp64_on_whole_persistent_tiles` pins it (n = 2 / 4, 128 / 2048 / 16 896 rows, < 3e-3). Error now 1.66e-3 at 128, 2048,
16 896, 147 456 rows, n = 2 and 4; 0 / 200 non-identical replays at 768 / 3072 / 16 896 / 49 152 rows.

## Rows tried, anthropic (us; rows 768 / 3072 / 12 288 / 49 152 / 147 456)

| n | D | row | us |
|---|---|---|---|
| 2 | 64 | v2 | 5.3 / 5.4 / 7.0 / 14.7 / 35.6 |
| 2 | 64 | lnl | 5.6 / 5.7 / 7.1 / 14.2 / 37.8 |
| 2 | 64 | af3_fused | 5.5 / 5.6 / 7.0 / 14.2 / 37.8 |
| 2 | 64 | pf | 7.0 / 7.2 / 7.5 / 18.4 / 54.8 |
| 2 | 128 | lnl | 9.4 / 9.7 / 12.7 / 34.4 / 101.8 |
| 2 | 128 | v2 | refused (`cell_128x256_not_configured`) |
| 2 | 128 | pf | 14.3 / 14.6 / 15.3 / 41.4 / 131.1 |
| 2 | 256 | torch_swiglu | 26.0 / 35.8 / 94.7 / 394 / 1108 |
| 2 | 256 | lnl | 32.8 / 32.7 / 63.5 / 205 / 593 |
| 2 | 256 | pf | 42.1 / 42.6 / 43.8 / 135.7 / 395.8 |
| 2 | 384 | pf | refused (`no-cell ... 384x768`) |
| 4 | 64 | v2 | 6.9 / 7.0 / 9.4 / 20.1 / 54.0 |
| 4 | 64 | lnl | 7.4 / 7.6 / 9.8 / 21.9 / 59.3 |
| 4 | 64 | af3_fused, pf | refused |
| 4 | 128 | lnl | 14.8 / 15.1 / 19.8 / 55.6 / 166.9 |
| 4 | 128 | v2 | 13.9 / 14.2 / 19.2 / 54.3 / 156.9 |
| 4 | 128 | pf | 23.4 / 23.7 / 24.5 / 69.0 / 212.9 |
| 4 | 256 | torch_swiglu | 28.2 / 46.6 / 142 / 534 / 1516 |
| 4 | 256 | lnl | 59.6 / 59.0 / 116.4 / 360 / 1043 |
| 4 | 256 | pf | 76.2 / 76.6 / 77.7 / 233.5 / 682.9 |
| 4 | 384 | pf | 77.4 / 159 / 460 / 1839 / 5407 |

n = 2 D512 and n = 4 D512: no anthropic row serves them.

## Ours (default dispatch, us; same rows; final build, run after the runtime ping-pong gate)

| n | D | us |
|---|---|---|
| 2 | 64 | 4.4 / 4.6 / 6.2 / 11.8 / 27.8 |
| 2 | 128 | 8.6 / 8.9 / 9.7 / 22.9 / 69.1 |
| 2 | 256 | 10.4 / 13.8 / 25.7 / 80.4 / 221.5 |
| 2 | 384 | 11.8 / 20.7 / 58.6 / 190.2 / 549.8 |
| 2 | 512 | 13.6 / 24.8 / 80.9 / 294.4 / 891.6 |
| 4 | 64 | 5.6 / 5.8 / 8.0 / 16.6 / 42.2 |
| 4 | 128 | 9.4 / 12.0 / 13.8 / 35.3 / 102.8 |
| 4 | 256 | 12.0 / 18.0 / 41.7 / 130.4 / 374.0 |
| 4 | 384 | 17.1 / 29.5 / 95.8 / 305.5 / 905.9 |
| 4 | 512 | 20.4 / 40.0 / 142.4 / 486.2 / 1505.4 |

Every cell with an anthropic number is faster; smallest margins n = 2 D128 3072 rows 8 % (8.9 / 9.7) and 768 rows 9 %, n = 2 D64 12 288 rows 11 %.
With 50 calls per graph (launch cost amortised; `mod_inf_p.py` / `anth_p.py`) the D128 margins are n = 2 7.4 - 29 % (768 .. 12 288 rows) and n = 4
10 - 37 %.

## Training (fwd + bwd, us; ours / PyTorch torch.compile, CUDA graph)

n = 4, rows 768 / 3 072 / 12 288 / 49 152 / 147 456:

| D | ours | PyTorch compile |
|---|---|---|
| 64 | 22.6 / 29.7 / 47.9 / 98.8 / 227 | 39 / 58 / 95 / 306 / 646 |
| 128 | 24.1 / 39.6 / 71.0 / 190 / 516 | 46 / 71 / 170 / 483 / 1282 |
| 256 | 34.8 / 65.9 / 198 / 632 / 1767 | 58 / 117 / 349 / 860 / 2326 |
| 384 | 47.7 / 108 / 324 / 1227 / 3568 | 67 / 148 / 509 / 2597 / 7040 |
| 512 | 61.6 / 143 / 462 / 1934 / 5600 | 106 / 254 / 956 / 3560 / 9984 |

1.32x (D256 147 456 rows: 1767 / 2326) up to 3.0x. n = 2 ours: see `docs/gpus/h100/transition/transition.md`.

Regression caught here: the inference ping-pong switches had leaked into the training forward build (D128 n = 4, 147 456 rows: 639 us vs 520 us
without). Inference-only now, and only for at most two tiles per CTA at run time.

## Not measurable here

* `esm_t16` (D256): the kit's cubin is built with NVRTC 13.0; loading it on driver 575 segfaults (exit 139 in a fresh interpreter).
* `flash_sm90a` (D256 n = 4 only): shipped binary is torch 2.13 + cu130. Rebuilding for cu129 works with `flash_sm90a/csrc/build.py`, but the sealed
  recipe pins CUTLASS 4.2.0 headers (`build_flash_prebuilt.py` refuses others); only 4.8.0 is on this machine and the compute nodes are offline.
  A build with 4.8.0 headers was made and is not a legitimate measurement; it was not used.
* `esm_fused_exact` loads: 60.4 / 60.6 us at 768 / 3072 rows (n = 4, D256), ours 12.1 / 18.0.
