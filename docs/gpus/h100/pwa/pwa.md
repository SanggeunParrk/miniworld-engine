# MSAPairWeightedAveraging on H100 (sm90)

Kernel-level status of `MSAPairWeightedAveraging` on H100; the module-level summary is in [h100.md](../h100.md) and the dispatch
contract in [dispatch.md](../dispatch.md). bf16, batch 1; the columns are (Length, MSA depth). The B200 page is
[../../b200/pwa/pwa.md](../../b200/pwa/pwa.md): the wide path below is the B200 structure written for wgmma.

Dispatch: `integrations/pwa_train.py` (MiniWorld width: `csrc/pair3.cu`, `ln_vg.cu`, `pwa_fwd2/3.cu`, `pwa_gateout.cu`, `pwa_glue3.cu`,
`dgv_bwd.cu`) and `integrations/pwa_wide.py` (every other width: `csrc/pair_wide.cu`, `csrc/pwa_wide.cu`). Served when
`implementation=miniworld` (or `anthropic`), bf16, batch 1, a token key mask (or none), inference and training alike; the MSA residual
and the module's row dropout (`drop_msa`) are applied in the gate / out pass. `settings.pwa_train` / `pwa_infer` turn the paths off.
Every other call runs the module's PyTorch statements on H100 (no portable-kernel fallback, see [dispatch.md](../dispatch.md)); other cards run the
portable portable path (which refuses d_hidden < 16: those calls run the PyTorch statements there too).

| width (d_msa, d_pair, heads x d_hidden) | model | path | length / depth |
|---|---|---|---|
| 64, 128, 8 x 32 | MiniWorld | `pwa_train.py` | L a multiple of 128, even S, N <= 1024 |
| 128, 256, 8 x 16 | ESMFold2 | `pwa_wide.py` | any L <= 1024 (padded to 16), any S |
| 128, 256, 8 x 8 | Protenix-v2 | `pwa_wide.py` | same |
| 128, 384, 8 x 8 | OpenDDE | `pwa_wide.py` | same |

The module has no pair-LayerNorm beta (its contribution is a per-head constant on the softmax logits, so the output never depended on it
and its gradient is 0): the kernels take zeros, a checkpoint's `ln_pair.bias` is dropped on load.

## Wide path fusion

| stage | forward | backward |
|---|---|---|
| F1 / B5 | `pair_fwd`: pair LN + proj_z + key mask + softmax over j -> w [H, N, N] (mma.sync, d_pair a compile-time width) | `pair_bwd`: softmax backward -> proj_z backward -> LN backward -> dz, dWb, dgamma_z |
| F2 | `ln_vg`: y = LN(m) (+ statistics), v = y Wv^T head-major [H, N, S*C] | |
| GEMM | cuBLAS `o = w . v` | cuBLAS `dv = w^T . do`, `dw = do . v^T`, and dWo / dWg / dWv |
| F4 / B1 | `gate_out`: gate = y Wg^T, `o * sigmoid(gate)` from registers into the out projection, dropout, residual | `gate_bwd`: do' = dout Wo, gate recomputed: go, do (head-major), dgp |
| B4 | | `dm_bwd`: dy = [dgp \| dv] [Wg; Wv], LayerNorm backward + residual gradient -> dm, dgamma / dbeta |

All kernels are built per (d_msa, d_hidden) / d_pair. Per-width warp / CTA counts of the pair kernels: `pwa_wide.PAIR_CFG`
(measured: d_pair 128 8 x 2, 256 4 x 3, 384 6 x 1).

## Completion

Columns (Length, MSA depth); each row below is one width, all CUDA + cuBLAS at every listed cell. 성능 확인 is the maintainer's.

| width | inference (S 1024 / 2048 / 4096, L 128..768) | training (S 1024 / 2048 / 4096, L 128..768) | 성능 확인 |
|---|---|---|---|
| 64 / 128 / 8 x 32 (MiniWorld) | CUDA | CUDA | ✗ |
| 128 / 256 / 8 x 16 | CUDA | CUDA | ✗ |
| 128 / 256 / 8 x 8 | CUDA | CUDA | ✗ |
| 128 / 384 / 8 x 8 | CUDA | CUDA | ✗ |

## Speed of light and other implementations

Measured 2026-10-13 ([record](../../../records/h100_opm_pwa_wide_20261013.md)), CUDA graph, S 1024, ms. Floor = sum over this fusion's
kernels of `max(bytes / HBM, FLOPs / TC)` at 3.04 TB/s and 703 TFLOP/s. Anthropic = `opt_core.ops.msa_pwa` as shipped (forward only;
does not compile at d_hidden 8); Triton = the portable path (refuses d_hidden 8); cuEquivariance ships no kernel.

| width, L | infer PyTorch | infer Triton | infer Anthropic | infer ours | SoL | train PyTorch | train Triton | train ours | SoL |
|---|---|---|---|---|---|---|---|---|---|
| 128/256/h16, 128 | 0.650 | 0.144 | 0.349 | **0.134** | 60 % | 1.577 | 0.754 | **0.588** | 45 % |
| 128/256/h16, 384 | 2.228 | 0.446 | 1.389 | **0.393** | 66 % | 5.017 | 2.201 | **1.605** | 52 % |
| 128/256/h16, 768 | 5.390 | 1.076 | 3.944 | **0.888** | 67 % | 12.022 | 4.874 | **3.421** | 59 % |
| 128/256/h8, 128 | 0.533 | — | — | **0.099** | 70 % | 1.290 | — | **0.432** | 46 % |
| 128/256/h8, 384 | 1.918 | — | — | **0.279** | 81 % | 4.261 | — | **1.208** | 53 % |
| 128/256/h8, 768 | 4.755 | — | — | **0.622** | 80 % | 10.345 | — | **2.557** | 60 % |
| 128/384/h8, 128 | 0.549 | — | — | **0.104** | 68 % | 1.335 | — | **0.451** | 45 % |
| 128/384/h8, 384 | 2.058 | — | — | **0.305** | 78 % | 4.624 | — | **1.317** | 52 % |
| 128/384/h8, 768 | 5.312 | — | — | **0.701** | 78 % | 11.821 | — | **2.945** | 57 % |

The MiniWorld width (64 / 128 / 8 x 32): [h100_pwa_20261008.md](../../../records/h100_pwa_20261008.md) (training L384 / L768 at 94 / 85 %
of its floor, inference 71-74 %, measured with different ceilings).

## Open

- Wide training at 45-60 % of the fusion floor: head-major intermediates (v, o, do, dgp, dv) round-trip HBM in six kernels plus three
  cuBLAS calls. `dm_bwd` (0.69 ms against about 0.5), `gate_bwd` (0.58, memory bound) and the `dv` copy into the natural-layout `dgv`
  are 2-5 % of a step each; `pair_bwd` is down to 0.34 / 0.64 ms (d_pair 256 / 384).
- Training of L512 / L640 at the MiniWorld width fails with `invalid argument` after a qualified shape in the same process (see the
  2026-10-08 record); the wide path was not affected in the runs above.
