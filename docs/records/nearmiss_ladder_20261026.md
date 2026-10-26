# Near-miss rows of the performance ladder on H100: before / after (2026-10-26)

Branch `perf/final-nearmiss` (on `integ/ladder-all` `e6a635d7`). Ladder rule: hand CUDA < portable (Triton) < PyTorch; a row is a near-miss at ratio 0.95-1.0.
Measured with `benchmarks/runners/ladder.py run --repeats 3` (compile + CUDA graph or no compile + graph, the faster regime per rung; PyTorch = fastest of compiled / eager-graph / eager)
on one H100 80GB HBM3 (node02), torch 2.13.0+cu129. Before = `e6a635d7`, after = this branch (final run `~/miniworld-engine-scratch/fin-near/out_final`; the other runs of the same
code are quoted where they differ).

## Result

| row | mode | ms before (CUDA / portable) | ms after (CUDA / portable) | CUDA / portable before | after | verdict |
|---|---|---|---|---|---|---|
| Transition D512 L384 (bf16) | training | 5.920 / 5.935 | 5.607 / 5.954; 5.622 / 5.911; 5.789 / 6.072 (three runs) | 0.997 | 0.942, 0.951, 0.953 | moved 4-5 %, **margin still ~0.95** |
| pair_embed_ca B16 L128 | inference | 0.057 / 0.060 | 0.055 / 0.059 | 0.958 | 0.934 | fixed (L256 0.908 -> 0.882, L384 0.895 -> 0.868) |
| DiT variants Chai-1 token fp32 | training | CUDA rung did not build (see below) | 5.847 / 6.297 | n.a. | 0.929 | already below 0.95 once the build is fixed (0.925 in the earlier record) |
| AttentionPairBias 8x48 pair256 | training | 0.335 / 0.369 (ladder: portable / PyTorch 0.872) | unchanged | 0.908 | 0.912 | not a near-miss in this tree: the ladder takes the faster regime of the portable rung (no compile + graph 0.369 ms); the 0.955 of the attention record is the compiled-only regime (0.4056) |
| pool_atoms 24576 | inference | 0.051 / 0.056 | unchanged | 0.906 | 0.905 | already <= 0.95, untouched |
| token_pair_init B16 L256 / L384 | inference | 0.207 / 0.216, 0.441 / 0.448 | unchanged | 0.958, 0.985 | 0.957, 0.970 | **not moved** |

## What changed

* **DiT variant CUDA rows did not build in the integrated tree** (`dit_variants_train_rows_sm90a`: `more than one instance of overloaded function "transpose_k"`): `train_rows.cu` does `using namespace tdr;` and
  the conditioned-transition branch added `tdr::transpose_k` (`token_dit_common.cuh`) with the same name. Every `dit_*` / `local_*` training and fp32 inference row therefore fell off its CUDA rung
  at `e6a635d7` (`n.a.: cuda: RuntimeError: Error building extension`). The file's own kernel (bounds-checked for sizes that are not a multiple of 32) is renamed `dv_transpose_k`; the 32 + 34 + 31 DiT variant GPU tests pass.
* **Transition D512 (n = 4) training keeps the forward's a / b at the 384 x 384 pair** (`kernels/transition/cuda/fused_wide_sm90a.py`, `_D512_SAVED_SIDES = (768, 384)`): so far only the 768 x 768 pair did, every other
  size ran the recompute path (the gate kernel recomputes a, b with two more GEMMs). With a / b kept the forward is `ln_fwd` + `fwd_split` (writes out, h, a, b), the gate backward is the dh GEMM + SwiGLU backward only, the compact
  `ln_bwd` follows. Model-level fwd+bwd in a CUDA graph (fp32-free, bf16 params): 5.335 -> 5.176 ms; ladder 5.920 -> 5.607-5.789 ms (the spread is the node, the portable rung moves with it). Cost: a, b, h = 1.8 GB more
  retained activations per D512 Transition layer at that pair size (the 768 x 768 pair already pays 4x that). The sub-switches were A/B-ed at 384 x 384 and the existing L768 combination wins: without the TMA ring 5.41, without the split forward 5.93, without the compact ln_bwd 5.17 (equal).
  Error against an fp64 run of the same inputs (bf16, 1 x 384 x 384 x 512, two seeds): new / PyTorch bf16 = 0.68 (out), 0.78 (dx), 0.69-0.71 (ln weight), 0.65-0.72 (ln bias), 0.85 (expand_a), 0.89 (expand_b), 0.70 (squeeze) -- all <= 1.5x; the
  recompute path was 0.1-0.5e-3 lower (a / b rounded to bf16 once before the backward), the two paths are not bitwise equal. New test: `test_d512_default_training_retains_ab_at_384_and_768_pairs`.
* **`pair_embed_ca` forward-only glue** (`kernels/pair_embed_geom/interface.py`): without a gradient the call no longer goes through `_PairEmbedFn` (which wrote the backward's int32 codes, 4 MB at B16 L128, and cast the table in
  two kernels: `.t().to(bf16)` then `.contiguous()`); the new opaque op `pair_embed_ca_fwd` writes no codes, and the table is one transpose + cast kernel (`empty().copy_(w.t())`). Profiler, B16 L128: 2.1 + 3.2 us of glue -> 3.7 us,
  `pg_fwd_ca` 49.3 -> 48.8 us. Both rungs share this glue (the portable rung takes the same op), so the gain on the ratio is the removed codes write and glue relative to a 55-59 us step. New test: forward-only call bitwise equal to the training forward (with and without the fused LayerNorm).

## Not moved

* **token_pair_init B16 L256 0.957 and L384 0.970.** Glue (profiler, L256): four D2D copies of `torch.stack` 1.7 us each, the int32 cast 3.3 + 2.2 us, `torch.cat` 3.1 us = 15 us of 207; the kernel is 180 us for 0.54 GB (3.0 TB/s, ~90 % of the
  HBM peak). The glue is shared by both rungs, and removing a shared constant g moves the ratio from (c)/(p) to (c - g)/(p - g) = 0.958 -> 0.955: not worth a change. The remaining gap is the kernel itself against the portable kernel
  at 85-90 % of DRAM; the earlier record already tried store policies and CTA shapes.
* **Transition D512 training is at 0.942-0.953, not clearly below 0.95.** The step is 5.12 ms of kernels (profile): `expand_saved` 1248 us (0.62 TFLOP, plus 1.8 GB of h / a / b writes), `gate_saved` 1221 us (reads a, b 1.2 GB, writes
  dA|dB 1.2 GB, 0.31 TFLOP: memory-bound floor ~0.8 ms), cuBLAS dW 382 + 669 us and dx 704 us (880-925 TFLOP/s), squeeze 481, `ln_bwd` 234, `ln_fwd` 112. The two hand kernels have 20-30 % of headroom each (epilogue / mainloop
  overlap), which is a kernel rewrite, not a switch. Harness extras around the step (grad accumulation, input handling) are another ~0.5 ms shared with the portable rung.
* **DiT Chai-1 fp32 token training** needed no kernel work (0.925-0.929 against 0.95); the SwiGLU-into-GEMM-epilogue idea (saves reading `ab`, ~0.1 ms of 5.8) was not started.
* **AttentionPairBias pair256 training**: see the table; no change made.

## Tests (H100, whole files to the end, no failures)

Transition: `sm90/test_transition_{wide,fused,generic_h16,n2}_sm90a_gpu.py` 72 + 23 + 2 + 24, `any/test_transition_{generic_sm90a,residual,layout,fp32_portable,fp16_halves}` 47 + 12 + 27 + 12 + 2;
pair embed: `sm90/test_pair_embed_geom{,_train}_gpu.py` 4 + 17, `sm90/test_pair_embed_adaln_gpu.py`, `any/test_pair_embed_{geom,adaln}_portable_gpu.py` 30 + 14, `sm90/test_pair_ops_portable_vs_cuda_gpu.py` 16;
DiT variants: `sm90/test_dit_variant_{train_,}h100_gpu.py` 32 + 34, `any/test_dit_variants_portable_gpu.py` 31; `any/test_token_pair_init_gpu.py` 12.
`tests/cpu/registry` has 2 failures + 2 errors (`test_no_stale_caches`, `test_key_flags_are_built_on_both_sides`, `test_a_ladder_offers_what_wins`) identically at `e6a635d7` (A5000 / A6000 / H100 autotune caches stale from other branches).
`dev derive --arch sm86` was re-run (`registry_kernel.units.json`; the sweep page is unchanged).
