# Fused Transition forward and backward on B200 (sm_100a)

The pair Transition `y = x + W_s(silu(W_a·LN(x)) · (W_b·LN(x)))` at D = 128, H = 4D = 512, bf16, as **one forward kernel** and
**one backward kernel** (plus a small partial-sum reduction) written for Blackwell: tcgen05 MMA with accumulators in tensor memory,
TMA, 2-CTA clusters. The fusion and the numerics contract (what is fused, what is saved, where values are rounded, in which order
the hidden chunks accumulate) are those of the H100 kernels in `../transition_fused/`; the mapping to hardware is new, because
Blackwell has no register-operand MMA.

Status: research capsule. The kernels are **not wired into the engine** (on B200 the engine runs the Triton Transition), and
they are launched from Python through `drv.py` (cubin + `cuda.bindings`), not a torch extension.

## Result

B200 (148 SMs, 1000 W power limit), CUDA-graph replay median, `bench_all.py` → `records/table-v22.json`:

| L384 | inference | training (fwd + bwd) |
|---|---:|---:|
| PyTorch eager | 429.0 µs | 1079.5 µs |
| torch.compile | 202.9 µs | 613.6 µs |
| Anthropic v2 (best Anthropic row; forward only) | 108.4 µs | n/a |
| **fused sm_100a** | **58.7 µs** | **275.8 µs** |

| L768 | inference | training |
|---|---:|---:|
| PyTorch eager | 1653.5 µs | 3932.8 µs |
| torch.compile | 740.7 µs | 2203.1 µs |
| Anthropic v2 | 496.8 µs | n/a |
| **fused sm_100a** | **215.6 µs** | **1127.5 µs** |

Accuracy vs the fp32 autograd module: out 2.25e-3, dx 3.1e-3, weight gradients 3.5-4.1e-3 — the same as the contract emulated in
torch; bit-reproducible run to run.

### Speed of light on this card

The card runs at its 1000 W cap under any sustained load, so time per call = energy per call / ~985 W (v5, v7). Dense bf16 cuBLAS
sustains only 1.30 PFLOP/s (clock peak 2.38). SoL is therefore stated as the design's tensor work (28·M·D·H for a training step)
at the bf16 cuBLAS rate **measured back to back in the same regime** (`step16_sustained.py`; v16 §1 explains why burst vs
sustained must not be mixed):

| training step | sustained | SoL |
|---|---:|---:|
| L384 | 309.5 µs | **67 %** |
| L768 | 1165.0 µs | **72 %** |

SoL90 is out of reach for this fusion on this card with the bf16 contract: the forward with LayerNorm, SwiGLU and weight reloads
all removed (a skeleton that computes nothing correct) is 46.7 µs, 79 % of the 1.58 PF-basis SoL (v12); the backward skeleton is
176 µs vs 149.5 µs (v13). The non-tensor energy is data movement (LayerNorm, TMEM round trips of `[a|b]` / h, weight streaming,
~7 % each), not MUFU math. The relaxed-precision e4m3 path (v16-v18) reached 96 % on the time basis with 3-7 % error vs fp32 and
is kept as a record only (v19).

## How it works

**Forward** (`src/tfwd2.cu`, v8; `fwd_op.FusedFwd2`). Persistent, one CTA per SM, 128-row tiles, the two CTAs of a 2-CTA cluster in
lockstep. Per hidden chunk j (eight of 64 units): the leader issues the expand `[a|b] = xn [Wa_j; Wb_j]^T` as one M = 256
`cta_group::2` product (the leader holds Wa_j, the peer Wb_j — each SM streams half of the weights) into a TMEM buffer; two SwiGLU
warpgroups (ping-pong over two `[a|b]` buffers, 16-column pipelined TMEM loads) form `h = bf16(silu(a) b)` and write it back to
TMEM; the squeeze `acc += h Ws_j^T` is an A-from-TMEM product accumulating across all eight chunks. h never touches shared or
global memory. LayerNorm of the next tile runs on its own warpgroup and reproduces the sm_90a reduction tree, so `xn` / `rstd` /
`c1` are bit-comparable to the H100 kernel; inference skips the `xn` save at run time.

**Backward** (`src/tbwd.cu`, v9 + v21/v22; `bwd_op.FusedTrain`, R = 9). Two CTA roles in one launch, 2-CTA clusters:
- 8 × R **DW** CTAs keep one hidden slice's Ws / Wa / Wb resident, recompute dh / a / b for every tile, run the gate into shared
  memory and accumulate `[dWa_s; dWb_s]` and `dWs_s` in TMEM. xn and dy arrive as separate multicast streams (v9).
- The remaining **DX** CTAs stream the weight chunks (multicast over the pair), recompute dh_j / `[a|b]_j` into TMEM, write
  `[dA|dB]` back over `[a|b]` as bf16, accumulate `d_xn` with an A-from-TMEM product, and run the LayerNorm backward + residual
  from TMEM.
- Gates and the LayerNorm-backward epilogue in packed fp32x2 (`GATE2`, `EPIF2`; bit-identical / same rounding points).
- No atomics; `transition_bwd_reduce` sums the per-CTA partials.

## Layout

| path | what |
|---|---|
| `src/tfwd2.cu`, `src/tbwd.cu`, `src/sm100.cuh` | default kernels and the sm_100a helpers (tcgen05, TMA, mbarrier) |
| `src/tfwd.cu`, `src/tfwd3.cu` | 1-CTA forward (v6) and three-accumulator forward (v11), kept for comparison |
| `src/tbwd2.cu`, `src/tbwdx.cu` | backwards without a recompute, exchanging gate outputs through L2 (v14: `[dA\|dB\|h]` to DW; v20: `[dA\|dB]` to DX), not adopted |
| `src/tfwd8.cu`, `src/tbwd8.cu`, `src/tbwd8x.cu` | e4m3 record (v16-v18) |
| `src/*_bench.cu`, `src/*_test.cu` | hardware micro-benchmarks (MMA rate/latency, MUFU, L2→smem, multicast, DSMEM, TMEM load shapes) |
| `fwd_op.py`, `bwd_op.py`, `fwd8_op.py`, `bwd8_op.py`, `drv.py`, `common.py` | host side: tensor maps, launches, inputs |
| `bench_all.py` | the comparison table (PyTorch, torch.compile, Anthropic rows, ours) |
| `bench_fwd.py`, `bench_bwd.py`, `bench8.py` | correctness vs the contract / fp32 and timing |
| `energy*.py`, `step*_sustained.py`, `ab_*.py` | NVML energy, same-regime SoL, alternating A/B |
| `trace_*.py`, `span*.py`, `prof*.py`, `ncu_*.py` | clock64 role traces, per-role exit times, nsys / ncu helpers |
| `rounds/vN.md` | one file per round: profile, hypothesis, change, validation, table |
| `records/table-vN.json`, `records/experiments/*.patch` | tables per round; measured-but-rejected variants |

## Running it (B200 box)

```bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950
source $W/pixi_env.sh && cd $W/miniworld-engine/experiments/transition_fused_sm100
./build.sh tfwd2 && ./build.sh tbwd            # sm_100a cubins in build/, ptxas register/spill summary printed
$W/gpuq/gpuq run -g auto -- pixi run --frozen python bench_bwd.py --length 384 --repl 9
$W/gpuq/gpuq run -g auto -- pixi run --frozen python bench_all.py --lengths 384 768 --save records/table-vN.json
```

Before any timing, check `nvidia-smi --query-compute-apps` for other tenants' jobs: under the shared power cap they change every
number (v11 discarded two hours of contended runs). ncu needs root on this box (`RmProfilingAdminOnly`); it runs through `gcsudo`
and locks the SM clock (~0.94 GHz), so read ratios, not times (v21).

## Rounds

| round | change | L384 inference / training |
|---|---|---:|
| v1 | bring-up of the H100 fusion on tcgen05 | 86.7 / 392.4 µs |
| v2 | converged MMA issue (no ELECT waterfall), split issuers and rings, two SwiGLU warpgroups | 61.9 / 341.0 |
| v3 | backward gate warpgroups, register plan, split DX rings | — / 333.6 |
| v4 | 2-CTA clusters (multicast inputs / weights), DX gate split by columns | — / 311.0 |
| v5 | power-capped ceiling measured; poly-ex2, x-in-TMEM, forward multicast (not adopted) | 62.0 / 312.6 |
| v6 | pipelined SwiGLU TMEM loads | 57.3 / 307.4 |
| v7 | energy per call; Newton rcp, back-off (not adopted) | — |
| v8 | 2-CTA `cta_group::2` forward | 58.4 / — |
| v9 | DW input as separate xn / dy streams, R = 9 | 58.7 / 291.3 |
| v10-v11 | DX buffer restructurings, quad-layout epilogue, three-accumulator forward (not adopted) | — |
| v12-v13 | energy attribution; forward and backward skeletons miss SoL90 | — |
| v14-v15 | exchange backward, LN fold, 4-CTA clusters, cluster reduce-scatter (not adopted) | — |
| v16-v18 | e4m3 + exchange backward: 46.9 / 213.9 µs, 96 % time-basis SoL, 3-7 % error | record only |
| v19 | decision: bf16 stays the default | — |
| v20 | bf16 exchange backward (tbwdx): energy unchanged, not adopted | — |
| v21 | first ncu profile (no saturated unit); fp32x2 gates, bit-identical | 58.7 / 286.1 |
| v22 | fp32x2 DX LayerNorm-backward epilogue | 58.7 / 275.8 |
| v23 | re-baseline on v2.2.0; reduce-scatter / paired-sigmoid ablations (not adopted); DW zero-tile fix; ptxas 13.1 vs 12.9; engine wiring | 58.9 / 276.1 |

Other widths and small L (D = 64 / 256 / 384 / 512; `bench_infer_all.py` for inference at L128-768 vs torch.compile and Anthropic):

| round | change |
|---|---|
| w1-w2 | D64 fused forward + backward, D256 fused forward; split path (LN / expand + SwiGLU / squeeze GEMM) for D >= 256 |
| w3-w4 | gate ablations; saved-a / b ("no recompute") path, faster than torch.compile at D384 / D512 |
| w5-w6 | xn-resident SwiGLU, fused d_xn + LN backward: slower; the D >= 384 step sits at its energy floor |
| s1 | small L: cluster-scope releases off the critical path (fused D64-256 faster at every L, D64 L128 9.7 -> 8.5 µs); SwiGLU item schedule up to L384 for D384 / D512; all 30 (D, L) inference combos beat torch.compile and Anthropic (`records/infer-all-s1.json`) |

## Open items

- Wired into the engine on branch `feat/transition-sm100a` (v23 §5, s1 §9): `kernels/transition/cuda/fused_sm100a.py` (D128),
  `fused_wide_sm100a.py` (D64 / 256 / 384 / 512), kernels in `kernels/transition/cuda/sm100/`, page
  `docs/gpus/b200/transition/transition.md`.
- Every width is wired into the engine on `feat/transition-sm100a` (s1 §9): the engine's `sm100/` sources are GENERATED from `src/`
  by `export_engine.py <engine root>`; re-run it after changing a kernel here (and re-check the SASS, s1 §9).
- Build with CUDA 13.x: 12.9's ptxas makes the backward ~8 % slower (v23 §4).
- Re-measured under the engine's v2.2.0 env (torch 2.13 / cu129): unchanged (v23 §1).
- Remaining levers named by the rounds: the DX LayerNorm-backward epilogue is on the latency path (10 µs sustained, v22 / v23) and
  spills; its dgamma / dbeta reduce-scatter is worth ≤ 5.5 µs (v23 §2); more epilogue warps need the loader roles packed into fewer
  warps (v18 §2).
