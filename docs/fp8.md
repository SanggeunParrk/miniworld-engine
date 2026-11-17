# FP8 routes (H100, opt-in, engine 3.0.0 line)

Everything here is opt-in and off by default. With no switch set, every route is bit-identical to 2.5.9 (checked, see "Verification" in the release notes of the stage).
No registry row, tolerance or `backend_versions` entry exists for FP8: the registry dtype vocabulary is `float32|master` only, and the FP8 planes / e4m3 V are internal
intermediates of the `master` route. The numerics harness (ours vs PyTorch) would flag the ~3.8 % update error; that decision belongs to the owner (see "Open decisions").

## TriMul forward: e4m3 planes (`h100_b200style_fp8`, details in `docs/h100/trimul.md`)

Scope: one sample, bf16 inference, D128 / D256 (hidden = pair), one direction or the bidirectional block (`BidirectionalTriangleMultiplication`, from 2026-11-16, branch `fp8/bidir-tm`). B > 1, D64 / D384 / D512, training stay bf16.

Enable (environment, read at import, set before `import miniworld_engine`):

| env | effect |
|---|---|
| `MINIWORLD_TRIMUL_FP8=calib` | e4m3 a / b planes between K1 and the contraction (cuBLASLt e4m3 batched GEMM, bf16 out), static per-module scales calibrated on the first eager call (16x headroom) |
| `MINIWORLD_TRIMUL_FP8_GUARD=lazy` | no host sync (required with CUDA graphs); `sync` (default) redoes an overflowing call itself; `off` |
| `MINIWORLD_TRIMUL_FP8_FASTACC=1` | cuBLASLt fast accumulation (1.07-1.08x of the contraction at L >= 1024) |
| `MINIWORLD_TRIMUL_FP8_HEADROOM`, `_SCALE` | scale headroom (16) / `tensor` (default) or `channel` scales |
| `MINIWORLD_TRIMUL_FP8_K1=1` | EXPERIMENTAL e4m3 K1 compute (6.9 % update error): never a default |
| `MINIWORLD_TRIMUL_FP8_MINL=<n>` | replaces the per-width / direction min-L gates (below them the bf16 default route runs): D128 out 144, D128 in 192, D256 out 224, D256 in 144, bidirectional D128 256, bidirectional D256 1024; 0 = no gate |

The K3 pipeline flags are the 2.5.9 bf16 ones (`h100_b200style.K3_EXTRA`); the FP8 route uses that single hook (no private copy).
The experiments `K3_MC` (2-CTA multicast, D256), the K1 `WTMA` / `PIPE` variants and the TriAttn core macros `TA_F8SW` / `SWN` / `QM` / `K` / `TAU10` were removed from the code (no gain): see "Tried, no gain".

### Guard contract (for FoldForge)

K1 reports the max scaled plane value; the e4m3 cast saturates above 448, and unguarded saturation gave 35 % update error (guarded 3.8 %, bf16 floor 0.6 %).
With `GUARD=lazy` the caller must, once per forward and outside any graph:

1. `peak = h100_b200style_fp8.overflow_peak()` (one host sync, device max since the last read);
2. if `peak > 448`: `h100_b200style_fp8.rescale_all(peak)` (in place, captured graphs follow) and redo the forward (RNG restored; at most 3 times).

The patch that wires this into FoldForge is a file only, not applied: `~/miniworld-engine-scratch/fp8/trimul/foldforge_patch/trimul-fp8-guard.patch` (README beside it).
A capture with no calibration takes the bf16 route (the first two eager warm-up calls of FoldForge's `_Graph` calibrate).

## TriAttn forward: FP8 PV, per-model whitelist (`h100_sm90`)

`triattn_h16_sm90.cu` built with `-DTA_F8PV=1` (P as e4m3 bits straight from the logits, V as e4m3 `V^T` hi plane, or hi + lo with `-DTA_F8HL=1`; row sums from the quantized P).
Not a general route. Use the whitelist API:

```python
from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
if hs.enable_fp8_pv_for("protenix1"):   # True only for whitelisted models; sets FP8_PV, FP8_PV_FUSED, FP8_VSCALE_AUTO (+ FP8_PV_UNFUSED / the S gate for "unfused" models)
    ...                                 # d_pair 128 inference: the front writes V^T planes (tri_front_vt); other fronts: standalone V^T pass, single pass, static scale
# after every forward: if hs.take_overflow(): hs.rescale_all(); redo
hs.disable_fp8_pv()
```

`FP8_PV_WHITELIST` maps a model to the call-site kinds whose end-level accuracy was checked: `{"protenix1": {"fused"}, "opendde": {"unfused"}}`; Boltz-2 is absent on purpose (9c90, see limits).
`enable_fp8_pv_for(model, hl=False)`: `hl=True` writes the hi + lo planes (output error about half, see below). Default (non-FP8) route: bit-identical to 2.5.9 (10 shapes / orientations / head counts compared as bytes; branch `fp8/vt`, whose default builds of `front_fused3_sm90.cu` and `triattn_mod_sm90.cu` changed in source (the V^T code is `#ifdef` / a template instance): 14 more shapes d_pair 64 / 128 / 256 / 384, start / end, flip, masked, L256..1536, as bytes against main ec11768fc). The FP8 route with front-made planes equals the standalone-pass FP8 route as bytes (hi: 64:4:32 / 64:2:32 / 64:4:16 L256..1536 with a padded key mask, trans; hi + lo L256..1024; 256:8:32 L256..2048, trans).

Three kinds of call sites (branch `fp8/vt`: the first two make the planes in the front, the pass only remains where the front is cuBLAS):
- **fused** (d_pair 128, 4 x 32): the front writes the e4m3 V^T planes itself (static per-weight-pack scale, power of two above 4 x the calibration V absmax, saturation flag).
- **front-made** (d_pair 64: 4 x 32, 2 x 32, 4 x 16; d_pair 256 x 8 x 32; `FP8_FRONT_VT`, default on inside the FP8 route): the one-kernel fronts write the planes the same way. d_pair 64: `front_fused3_sm90.cu` built with `-DMW_F3_VTW=1|2`
  (bf16 front up to L768 / 2048 compact, fp16-exact Y16 front beyond; the V block of the projection is quantized from the fp16-rounded V, a lane pair swaps bytes so each lane stores one 32-bit word of 4 keys,
  then 16-byte global stores); d_pair 256: the group-2 CTA of `tri_front_wide` (`VTW` template, hi plane only; hi + lo keeps the standalone pass). The planes are bit for bit the standalone pass's
  (same fp16 V, same static scale, same cvt): the module output equals the standalone route as bytes. The first eager call of a site still runs the pass (it takes the static scale).
- **unfused with cuBLAS front** (d_pair 384 x 12 x 32, and any shape without a one-kernel front): fp16 V from the cuBLAS q | k | v | g GEMM, one `vt_quant` pass makes the planes. With `FP8_VSCALE_AUTO` the pass uses a per-site static scale
  (calibrated on the first eager call; single pass, no amax pass, same `take_overflow` / `rescale_all` contract). `vt_quant` serves up to 16 heads (12-head d_pair 384 had silently run the slow torch reference).
S gates: standalone pass `FP8_PV_UNFUSED_MIN_S` (32-channel heads 1024, 16-channel heads 512; x 2 with hi + lo); front-made planes `FP8_PV_UNFUSED_FRONT_MIN_S` per (d_pair, heads, channels): 64:4:32, 64:2:32, 64:4:16 from L256, 256:8:32 from L768
(x 2 with hi + lo). Below the gate the fp16 core runs (bit-identical to the default). The scale tensors derived from the static scale (1 / scale, scale per sample) are cached and updated in place by `rescale_all`: a call launches no torch glue kernel for them.

### Module time of the merged state (branch `fp8/main`) against the 2.5.9 default

ms per sample; percent change of the FP8 route against the 2.5.9 default (negative = FP8 faster); interleaved, 9 rounds (median of 8 calls per round), one H100 per job, random weights, flat logits, 2.5.9 default measured twice per round as the
same-cubin control (control column: the control against the default, i.e. the noise of that row; differences below ~3 % are noise). Routes exactly as `enable_fp8_pv_for` sets them: `fused` = whitelist kind "fused" (d_pair 128: the front writes the e4m3 V^T planes),
`shipped` = kind "unfused" with the S gates (below its gate the fp16 core runs, so the row reads 0 +- noise), `no gate` = unfused route with every S gate removed (the cost / gain of the FP8 core where the gate keeps it off). Hi plane. Raw rows: `~/miniworld-engine-scratch/fp8/main/res/mod_*.jsonl`.

| config (d_pair:heads:channels) | route | L256 | L512 | L768 | L1024 | L1536 | L2048 | control range |
|---|---|---|---|---|---|---|---|---|
| 128:4:32 (Protenix v1, Boltz-2, AF3) default ms | | 0.098 | 0.439 | 1.209 | 2.565 | 7.861 | 17.32 | |
| 128:4:32 | fused | -6.8 % | -10.0 % | -11.7 % | -11.2 % | -15.9 % | -17.2 % | -0.3 .. +2.5 % |
| 128:4:32 | shipped (gate 1024) | 0.0 % | +0.3 % | +0.8 % | -5.9 % | -12.0 % | -13.3 % | |
| 128:4:32 | no gate | +1.8 % | 0.0 % | -3.8 % | -6.9 % | -12.2 % | -13.4 % | |
| 256:8:32 default ms | | 0.218 | 1.060 | 2.896 | 6.022 | 17.78 | 38.67 | |
| 256:8:32 | shipped (front-made planes from L768) | +0.6 % | +0.3 % | -4.4 % | -4.9 % | -10.9 % | -14.2 % | -1.2 .. +2.8 % |
| 256:8:32 | no gate | +1.9 % | -1.1 % | -7.0 % | -7.6 % | -11.6 % | -13.6 % | |
| 384:12:32 (OpenDDE) default ms | | 0.395 | 1.833 | 4.963 | 10.07 | 28.94 | 64.01 | |
| 384:12:32 (cuBLAS front, pass) | shipped (gate 1024) | +0.4 % | +1.6 % | +0.3 % | -4.6 % | -9.5 % | -11.5 % | -0.3 .. +0.8 % |
| 384:12:32 | no gate | +0.3 % | +2.2 % | -3.1 % | -5.5 % | -9.3 % | -11.0 % | |
| 64:4:16 default ms | | 0.0875 | 0.404 | 1.095 | 2.385 | 7.187 | 15.91 | |
| 64:4:16 (front-made, N24 core) | shipped (from L256) | -6.3 % | -14.6 % | -18.8 % | -21.4 % | -23.8 % | -21.2 % | -1.9 .. +0.2 % |
| 64:4:16 | no gate | -6.3 % | -14.6 % | -18.8 % | -20.5 % | -22.0 % | -22.0 % | |
| 64:4:32 default ms | | 0.103 | 0.457 | 1.233 | 2.546 | 7.775 | 17.44 | |
| 64:4:32 (front-made) | shipped (from L256) | -10.5 % | -11.1 % | -9.4 % | -8.4 % | -13.5 % | -16.9 % | -0.4 .. +2.5 % |
| 64:2:32 default ms | | 0.0627 | 0.258 | 0.678 | 1.397 | 4.022 | 9.072 | |
| 64:2:32 (front-made) | shipped (from L256) | -4.3 % | -6.0 % | -8.2 % | -7.1 % | -9.3 % | -14.0 % | -0.1 .. +3.0 % |

Where the gain comes from: the F8 core is 24-28 % faster than the 2.5.9 core on real activations (table below); the V^T planes cost a standalone pass (0.01-1 ms, 56-95 % of the byte floor; launch bound below L512) that ate 40-55 % of that gain at module level
until the fronts that are ours (d_pair 64, d_pair 256 x 8, d_pair 128) wrote the planes themselves (the pass removed: 64:4:32 L1024 -4.6 -> -8.4 %, L2048 -13.9 -> -16.9 %). The cuBLAS-front site (384:12:32) keeps one single-pass static-scale `vt_quant` (74 % of the byte floor; a one-kernel K = 384 front
would gain about 2.5 % of the module and is a second numerics path: not done). The S gates (the "shipped" rows) come from the earlier per-round measurements; the "no gate" rows above are the re-measure and agree with them (the gate only matters where the row is above +-noise: 128:4:32 below L1024, 256:8:32 / 384:12:32 below L768 / L1024).
History in one line each: round 2 (V as hi / lo planes, fused d_pair 128 front), round 3 (static per-site scale, `vt_quant` for 16 heads, S gate, whitelist per call-site kind), round 4 (front-made planes for d_pair 64 / 256 x 8; `TA_F8N24` for 16-channel heads: PV as m64n24, bit-identical).
hi + lo costs more (128:4:32 fused +2 .. -10 %, unfused +16 .. -5 % depending on config and L): hence the gate is twice as high for it.

### TriMul FP8 planes, module time against the 2.5.9 default (2026-11-15, branch `fp8/tm2`; D128 / D256, one direction, one sample, bf16 inference; `MINIWORLD_TRIMUL_FP8=calib`, lazy guard)

Protocol: the module (`TriangleMultiplication`, `master_weights=True`), 9 interleaved rounds (default, default control, FP8), median; primary column = CUDA-graph replay of 8 calls (what FoldForge runs; GPU time), second block = eager (8 calls back to back). Same-cubin control within +-1 % (graph). H100 SXM, power-capped at ~690 W under sustained load.

Graph replay, FP8 vs default (negative = FP8 faster); L >= 256 is above every gate:

| D | direction | L256 | L512 | L768 | L1024 | L1536 | L2048 | default ms at L2048 |
|---|---|---|---|---|---|---|---|---|
| 128 | outgoing | -9.0 % | -15.3 % | -14.7 % | -18.5 % | -21.3 % | -24.9 % | 6.13 |
| 128 | incoming | -10.1 % | -14.9 % | -16.5 % | -17.4 % | -23.0 % | -23.2 % | 5.97 |
| 256 | outgoing | -3.9 % | -6.6 % | -5.9 % | -5.1 % | -13.6 % | -16.4 % | 13.23 |
| 256 | incoming | -4.1 % | -6.7 % | -4.4 % | -6.1 % | -14.2 % | -16.3 % | 13.33 |

Eager (host-bound below L768: the module call costs 0.31-0.44 ms of host time, the GPU 0.07-0.24 ms; run-to-run +-5 % at L256 / 512): D128 out -6.3 / -7.9 (repeat; first run +9.3) / -14.8 / -19.2 / -20.5 / -28.8 %, D128 in -6.3 / -5.2 / -16.6 / -17.4 / -23.6 / -25.0 %,
D256 out -19.4 / -6.8 / -5.9 / -4.1 / -11.3 / -16.8 %, D256 in -20.2 / -7.0 / -5.3 / -4.0 / -15.9 / -16.5 % (same L columns).

What the earlier table (D128 incoming +7.4 / +8.9 % at L256 / 512, D256 -3 .. -5 % at L512-1024) was: an EAGER measurement.
1. Below L768 the eager module call is host-bound for BOTH routes (stalled-GPU enqueue time 0.33-0.35 ms = the eager wall, GPU 0.07 / 0.24 ms), so the eager column compares host work. The FP8 incoming route did 20-30 us more host work per call than the default
   (the LN-statistics transpose of the incoming direction: `view`/`transpose`/`contiguous`/`copy_`, plus an `aten::zeros` for the amax buffer that `setdefault(.., torch.zeros(1))` evaluates on every call): +17 us at L256 (0.356 vs 0.339 ms).
2. Both are gone (see "Fixes"); host time of the FP8 route is now 0.30-0.33 ms, below the default's 0.33-0.44 ms.
3. In graph replay (no host cost) the FP8 route already won at every L >= 256 before the fix (D128 in -3.9 / -12.0 %, D256 -3.7 .. -7.6 %); the fix gave another -6.5 % (L256) / -2.9 % (L512) of the D128 incoming module.

Fixes (output bytes identical to `fp8/main`: 20 / 20 FP8 cases, 40 / 40 default cases): K1 stores the LN_in statistics at the swapped pair token when it reads the pair transposed (`K1Params.stats_t`, `TMN_K1_F8` builds only; the host transpose of `xs`,
one elementwise kernel of 3.1 / 5.5 / 13.9 us at L256 / 512 / 1024 and four host ops, is gone); the amax buffer is allocated once per module. The remaining extra kernel launches of the FP8 route are the cuBLASLt e4m3 contraction and its 0.8 us workspace `Memset` (the only kernels the default route has no twin for).

Per-stage breakdown (profiler device time, single call; graph = module graph-replay time per call; %SOL = max(byte floor at 3.05 TB/s, tensor floor at the 989 / 1979 TF peak) / measured; in brackets against the cuBLAS-measured peak 748 / 1431 TF; incoming, outgoing is within 1-2 %):

| D | L | stage | default ms (%SOL) | FP8 ms (%SOL) |
|---|---|---|---|---|
| 128 | 256 | K1 / contraction / K3 | 0.0276 (60) / 0.0144 (115) / 0.0274 (61) | 0.0245 (46) / 0.0110 (100) / 0.0227 (73) |
| 128 | 256 | module: kernel sum / graph / %SOL | 0.0694 / 0.0671 / 72 | 0.0589 (+0.0007 other) / 0.0603 / 66 |
| 128 | 512 | K1 / contraction / K3 | 0.0863 (77) / 0.0743 (89) / 0.0855 (78) | 0.0797 (56) / 0.0507 (87) / 0.0753 (89) |
| 128 | 512 | module | 0.2461 / 0.2429 / 81 | 0.2065 / 0.2067 / 75 |
| 128 | 1024 | K1 / contraction / K3 | 0.3405 (78) / 0.3850 (72, 95) / 0.2784 (96) | 0.2837 (63) / 0.2453 (72, 78) / 0.2808 (95) |
| 128 | 1024 | module | 1.0065 / 0.9846 / 81 | 0.8106 / 0.8131 / 77 |
| 128 | 2048 | K1 / contraction / K3 | 1.392 (77) / 2.925 (76, 100) / 1.153 (93) | 1.178 (61) / 1.700 (65, 90) / 1.156 (92) |
| 128 | 2048 | module | 5.473 / 5.971 / 80 (graph 73) | 4.034 / 4.583 / 72 (graph 63) |
| 256 | 256 | K1 / contraction / K3 | 0.0646 (54, 71) / 0.0353 (93) / 0.0513 (65) | 0.0717 (48, 64) / 0.0216 (102) / 0.0531 (62) |
| 256 | 256 | module | 0.1540 / 0.1531 / 66 | 0.1472 / 0.1468 / 61 |
| 256 | 512 | K1 / contraction / K3 | 0.2284 (61, 80) / 0.1460 (90) / 0.1803 (74) | 0.2414 (58, 76) / 0.0979 (90) / 0.1835 (72) |
| 256 | 512 | module | 0.5577 / 0.5596 / 72 | 0.5236 / 0.5223 / 69 |
| 256 | 1024 | K1 / contraction / K3 | 0.8712 (64, 84) / 0.7406 (75, 99) / 0.7274 (73) | 0.9168 (61, 80) / 0.5051 (70, 76) / 0.7231 (73) |
| 256 | 1024 | module | 2.342 / 2.466 / 70 (graph 67) | 2.146 / 2.316 / 67 (graph 62) |
| 256 | 2048 | K1 / contraction / K3 | 2.839 (78, 104) / 6.344 (70, 93) / 3.112 (68) | 3.081 (72, 95) / 3.663 (61, 84) / 3.126 (68) |
| 256 | 2048 | module | 12.30 / 13.33 / 72 (graph 66) | 9.871 / 11.16 / 67 (graph 59) |

(graph minus kernel sum at D256 L1024 / L2048 and D128 L2048: sustained replay runs at the power-capped clock; measured SM clock / power during 60 replays of L1024: D256 default 1390 MHz, FP8 1462-1505 MHz, both 690 W; D128 default 1610-1628, FP8 1488-1500 MHz.)

Reading it, with the removal ablations (CUDA-graph module time, same-cubin control within 0.2 %, 11 rounds, `scripts/abl.py` of the work dir):
- Stages add up: removing the cuBLAS contraction (no call) takes 14.6 / 53.1 / 270 us off the D128 incoming module at L256 / 512 / 1024 (its profiler time 11.1 + 0.8 Memset / 51.9 / 259), 24.5 / 89 / 544 us at D256 L256 / 512 / 1024: the contraction is a serial stage, nothing overlaps with it.
- The overflow guard (K1 amax: `atomicMax` per warp) costs nothing measurable: with the K1 amax pointer removed the module time moves by -1.0 .. +1.6 % at D256 (L256 / 512 / 768 / 1024, noise) and 0.0 .. -0.2 % at D128 (profiler K1 time D256 L768 -9 % does not show in the module time).
- D256 gains little because the contraction is only 30 % of the module (0.73 of 2.34 ms at L1024) and e4m3 speeds it up 1.47x (0.74 -> 0.51 ms; the bf16 contraction already runs at 75 % / 99 % of the cuBLAS peak, 1.39 GHz): the saving is at most 0.23 ms of 2.34 (9.8 %), and the FP8 K1 is 2-6 % SLOWER than the bf16 K1 at D256 (L512 0.241 vs 0.228 ms, L1024 0.917 vs 0.871): the net is -5 .. -7 %.
- D128 gains more because the planes are 1/3 of the K1 write bytes (K1 -6 .. -17 %), the contraction share is larger (38 % at L1024, e4m3 1.57x) and K3 reads the same bytes.
- At small L the whole D128 module is 60-67 us: the GPU time is launch- and latency-bound (K1 / K3 at 46-73 % of their byte floor at L256), the FP8 saving there is the planes' bytes in K1 and the contraction (3.3 us), and 4.6 us of K3 time that comes from the direct K3 (`k3_static`) the FP8 route uses being faster than the packaged D128 `tmn_k3` at L256, not from FP8.

Min-L gates (below them the bf16 default route runs unchanged; `MIN_L` in `h100_b200style_fp8.py`, `MINIWORLD_TRIMUL_FP8_MINL=<n>` overrides, 0 = none). Graph replay, FP8 vs default, same-cubin control within +-1 %:

| L | 64 | 96 | 112 | 128 | 144 | 160 | 176 | 192 | 224 | 300 | 384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D128 out | +6.4 | +2.4 | +1.3 | +1.3 / +0.8 | -5.4 | -8.1 | -10.1 | -7.7 | -6.3 | -15.7 | -13.1 |
| D128 in | +8.1 | +3.4 | +2.7 | +1.7 / +2.6 | +7.0 | +2.6 | +0.3 | -8.9 | -7.6 | -18.6 | -14.2 |
| D256 out | +1.7 | -1.3 | +3.4 | -3.0 / -4.4 | +1.8 | +0.1 | -0.9 | +1.1 | -5.8 | +0.3 | -5.6 |
| D256 in | -1.4 | +0.3 | +3.4 | -1.9 / -2.7 | -10.1 | -9.9 | -9.7 | -0.5 | -7.5 | -0.7 | -4.8 |

Gates: D128 outgoing L >= 144, D128 incoming L >= 192, D256 outgoing L >= 224, D256 incoming L >= 144. D256 outgoing below L224 is within +-3 % on the GPU (20 us more host launch work than the default in eager mode); D256 incoming gains 10 % at L144-176 because the default incoming route is 14 % slower than its outgoing there (FP8 in = out), is flat at L192 and below L144. At L300 (row-padded K1 / K3) D256 is flat (+0.3 / -0.7 %), inside the gated range: no further gate. Two-sample cells (L128, 192) agree to 1 %; the gate lines sit between measured L, not at an interpolated threshold (L between 176 and 192 for D128 incoming, 128 and 144 for D128 outgoing: untested).

### TriMul FP8 planes, bidirectional block (2026-11-16, branch `fp8/bidir-tm`, on `fp8/main` + `fp8/tm2`; D128 / D256, one sample, bf16 inference, `MINIWORLD_TRIMUL_FP8=calib`, lazy guard)

Route (`h100_b200style_fp8.run(..., direction=0)`): the block's two directions are two of the single-direction FP8 routes on the matrix halves (rows `[:d]` outgoing, `[d:]` incoming of `to_left`, `to_left_gate`, `to_right`, `to_right_gate`): K1 (the single-direction FP8 tile, the incoming one on the transposed pair with the LN statistics stored at the swapped token) -> cuBLASLt e4m3 batched contraction of `d` channels, each writing its channels of one bf16 `tri [2 d, n, n]` -> one K3 over the 2 d channels (LN_out is joint over them, as in the default) with the per-channel scale product folded into the projection operand (`K3_CSCALE=1`: the two directions have different scales; the single-direction route keeps the scalar `K3_CSCALE=2`). Scales: per direction and per tensor (a, b), calibrated on the first eager call (16x headroom) into `cache["bd"][1 | 2]`; `refresh` / `rescale_all` / the lazy guard (`overflow_peak`, one amax buffer shared by the two K1) follow the same contract as the single direction. A capture with no calibration, and the guard's retry-exhausted fallback, run the bf16 default route of the block (`_update_inference_body`, not the single-direction port). The K1 planes of the two directions are two launches (the default has one K1 over 2 d hidden channels), which is why D256 gains less than D128: the FP8 K1 is tensor-bound and recomputes LN_in and re-reads z for the second direction.

Graph replay of the module, 11 interleaved rounds, median (default, default control, FP8, FP8 control; controls within 1 %); update error = rms(y - y_fp32) / rms(y_fp32 - x) against the fp32 PyTorch module on random inputs, mask with 3 masked tokens:

| D | L | default ms | FP8 ms | FP8 / default | update error FP8 (bf16 default 0.62 % at D128, 0.59 % at D256) | peak scaled plane value |
|---|---|---|---|---|---|---|
| 128 | 144 / 160 / 192 / 224 | 0.061 / 0.063 / 0.072 / 0.101 | 0.061 / 0.060 / 0.075 / 0.100 | 1.00 / 0.95 / 1.04 / 0.99 | 3.78-3.79 % | 17-19 |
| 128 | 256 / 384 / 512 / 768 | 0.113 / 0.232 / 0.407 / 1.012 | 0.107 / 0.216 / 0.369 / 0.859 | 0.949 / 0.932 / 0.907 / 0.848 | 3.78-3.79 % | 19-20 |
| 128 | 1024 / 1536 / 2048 | 1.964 / 5.363 / 10.79 | 1.609 / 3.937 / 7.750 | 0.819 / 0.734 / 0.719 | 3.79 % | 20 |
| 256 | 256 / 384 / 512 / 768 | 0.257 / 0.560 / 1.007 / 2.379 | 0.260 / 0.581 / 1.040 / 2.417 | 1.012 / 1.038 / 1.033 / 1.016 | 3.78-3.79 % | 19-20 |
| 256 | 1024 / 1536 / 2048 | 4.561 / 11.84 / 24.64 | 4.446 / 10.48 / 20.26 | 0.975 / 0.885 / 0.822 | 3.79 % | 20-22 |

Same error and peak as the single direction (3.8 %, scale peak far below 448); eager (8 back-to-back calls): D128 L512 / 1024 / 2048 1.026 / 0.829 / 0.726 of the default, D256 L512 / 1024 / 2048 1.003 / 0.971 / 0.824; host enqueue per call +0.06-0.07 ms at D128 (0.43-0.46 vs 0.37-0.40 ms), +0.00-0.01 ms at D256. Gates (`MIN_L[(d, 0)]`): D128 256 (L144-224 inside +-4 % noise, no gain), D256 1024 (L256-768 are 1-4 % slower: the second K1 launch); the `cuda` backend refuses D256 L % 128 != 0 at B = 1 for the default too, so those lengths never reach the route.

Where the FP8 module time goes (graph replay, per-kernel profiler time, D128 L1024 / L2048, D256 L1024 / L2048): K1 (two launches) 0.64 / 2.42, 1.93 / 8.11 ms; e4m3 contraction (two launches) 0.54 / 3.52, 1.04 / 7.30 ms; K3 (2 d channels) the rest. Removal ablation of the contraction (`res/abl_*.jsonl`): D128 L1024 1.60 -> 1.06 ms, L2048 7.77 -> 4.30; D256 L1024 4.21 -> 3.22, L2048 19.95 -> 12.30 (the contraction is serial: its removal removes its time; removing the amax output of K1 changes nothing within noise: 1.5 % at D128 L1024). Against the e4m3 tensor floor the contraction runs at 85-87 % of 1431 TF (D128 L2048 3.52 ms vs 3.07, D256 L2048 7.30 vs 6.15): nothing left to gain there, the cuBLASLt algorithm sweep of the single direction found no faster pick. K1 runs at 61 % (D128 L2048) .. 73 % (D256 L2048) of its bf16 tensor floor at 748 TF (two launches recompute LN_in).

Scale floor: `pow2_scale` (calibration scale of a channel from its amax) returned a scale from `amax.clamp_min(1e-30)`, i.e. 2^-99 .. 2^-107 for a channel whose amax is a denormal-sized ~0 (exactly 0 already gave 1): such a scale saturates the e4m3 cast on any later input (seen in the OpenDDE end-level runs). It is now floored at 2^-30 (amax 4e-7); channels above are unchanged (the single-direction route's outputs are bit-identical: 20 / 20 FP8 cases against `fp8/tm2`). The TriAttn per-head V scale (`vscale_from_amax`, `h100_sm90.py`, 2^-99 for a head with amax 0) is the same pattern; it belongs to the TriAttn branch and was not touched here.

### The F8 core alone on real activations

The FP8 core alone on REAL activations (q, k, v, bias of the dumps of Protenix v1 / Boltz-2 / AF3 / OpenDDE, 6 layers x start / end of 5I28, tiled to L1024 / L2048; ms per sample, mean over 12 layers):

| model (config) | L | default | hi core | hi core + V^T pass | hi + lo core | output error vs fp16 default, hi / hi + lo (mean / max) |
|---|---|---|---|---|---|---|
| Protenix v1 / Boltz-2 / AF3 (128:4:32) | 1024 | 1.79 | -24.4 .. -24.7 % | -11.4 % (dynamic pass) | -19 % | 2.1-2.4 / 2.7 %, 1.0-1.2 / 1.5 % |
| same | 2048 | 13.95 | -27.8 .. -28.2 % | -21.7 % (dynamic pass) | -21 % | 2.3-2.4 / 2.9 %, 1.2-1.5 / 1.8 % |
| OpenDDE (384:12:32) | 1024 | 5.50 | -25.3 % | -10.5 % (dynamic pass) | -19.5 % | 2.4 / 2.6 %, 0.95 / 1.2 % |
| OpenDDE | 2048 | 41.6 | -26.3 % | -18.5 % (dynamic pass) | -16.9 % | 2.6 / 2.8 %, 1.2 / 1.7 % |

(the real-activation cores gain more than the flat random ones: -25..-28 % vs -13..-16 %.) Error vs fp32 (reference of the 16 sampled rows, all 24 layer-targets per model): fp16 default 0.17-0.18 %, hi 2.0-2.3 %, hi + lo 0.85-1.2 % (mean).


### FP8 core, where the time goes and the 16-channel PV (2026-11-15, branch `fp8/core`)

Removal ablation of the F8 core (`TA_ABL` bits, time only, results wrong; one part removed per variant, interleaved rounds, 7 rounds, same-cubin control within +-2 %; percent of the F8 core time saved,
H100): the F8 core has NO MUFU at all (P is made by two HFMA2 + one HMNMX2 + a PRMT per four probabilities: `F8conv`), so there is no exp share and `TA_POLY` does not apply to it.

| removed | hv 4:32 L1024 / L2048 | flat 4:32 L1024 / L2048 | real (Protenix v1 / AF3 / Boltz-2) 4:32 L1024 / L2048 | hv 4:16 L1024 / L2048 |
|---|---|---|---|---|
| PV wgmma (4 x m64n40k32 per sub-step) | 28.5 / 27.1 % | 26.2 / 30.7 % | 23-27 / 25-29 % | 23.2 / 21.4 % |
| QK^T wgmma | 19.3 / 19.3 % | 10.1 / 14.6 % | 4.5-7.4 / 2.5-5.2 % | 11.4 / 11.1 % |
| e4m3 conversion (HFMA2 x 2, HMNMX2, PRMT) | 7.8 / 3.8 % | 0.8 / 6.8 % | 2.7-7.6 / 3.5-7.6 % | 8.2 / 11.5 % |
| the whole X phase (tree, vote, offsets, conversion) | 27.9 / 13.3 % | 10.8 / 10.0 % | 12.6-16.9 / 6.7-10.3 % | 26.1 / 20.7 % |
| bias ldmatrix | 6.4 / 4.6 % | 11.0 / 8.8 % | n/a | 7.1 / 1.3 % |
| epilogue stores (computation kept) | 3.3 / -1.9 % | -5.5 / 0.6 % | n/a | 3.1 / -4.5 % |
| K / V^T TMA loads, half the bias tile | within +-7 % (up to 20 % at hv L2048: zeroed operands also lower the power draw) | | | |

Caveats: a removal that zeroes or constants the data (loads, the exact path) also lowers the power draw (the card is power-capped), so those rows are upper bounds. The row-max tree + vote alone (tree computed, exact body never taken) costs
about as little as nothing; what the 17-30 % "no max machinery" ablation removed is the exact path (offset rise: 4 shuffles, `ex2f`, the 20 FMUL rescale of O) AND the data dependence it creates. The clean test of that
(`TA_F8K=7 TA_F8TAU10=15`: offset one octave lower so the exact path fires about half as often, data kept) gains 0-9 % on synthetic logits and -0.1..+3.7 % on real activations, with a 0.4-0.6 % output change: the exact path is not
what limits the real-activation core. PV is the largest part, the e4m3 conversion and the max machinery together are 7-17 % on real data.


Tried on the F8 core and REMOVED from the code (2026-11-15, `fp8/main`; no gain, default off, the removal changes no output byte: see "Verification"). Core-only, interleaved, 7 rounds, high-variance and flat logits, 4:32 / 4:16 / 8:32 at L512 / 1024 / 2048 and four real-activation layers at L1024 / L2048, against the F8 core with `TA_F8N24`:

| removed idea | what it did | core vs the F8 core | bit-identical |
|---|---|---|---|
| `TA_F8SW=1` (software-pipelined sub-step) | QK^T of the next sub-step under the e4m3 conversion | 4:32 -1.8 / -1.8 / -3.4 % hv, -1.2 .. -1.6 % flat; 8:32 +0.3 .. -4.4 %; 4:16 +0.1 .. +7.8 % | yes |
| `TA_F8SW=2` (16-channel heads only) | same, 16-channel heads | 4:16 +0.5 .. +5.4 %; 32-channel -2 .. +2 % | yes |
| `TA_F8SW=1 TA_F8SWN=2 / 4` | part of the conversion before the next QK^T | +5.7 .. +13.6 % | yes |
| `TA_F8QM=1` | row max of the vote in fp16, one vote | hv -1.4 .. -4.1 %, flat -1.3 .. +3.3 %, real +1.7 .. +9.8 % | yes |
| `TA_F8QM=2` | fp16 quad max in the exact path only | hv +1 .. +3.6 %, flat -1.7 .. +2.2 %, real -3.1 .. +0.6 % | yes |
| `TA_F8K=7 TA_F8TAU10=15` / `TA_F8K=6 TA_F8TAU10=25` | lower probability offset (exact path fires about half as often) | hv -3 .. +2 %, flat -3 .. -9 %, real -0.1 .. +3.7 % | no (0.4-0.8 % rel. L2 on real layers, 1.1-1.4 % on hv) |
| TriMul K1 `TMN_K1_WTMA` | per-warp plane TMA stores (no per-block warpgroup barrier), bf16 K1 | no number was kept (default off, never became faster than the shared-barrier store) | yes |
| TriMul K1 `TMN_K1_PIPE` (F8 compute, "schedule 6" split n128 block) | two n64 accumulators, MMAs of the next half under the epilogue | no number was kept (default off; the F8-compute K1 is itself experimental) | n/a |
| TriMul K3 `K3_MC` (D256) | weight chunks TMA-multicast over 2-CTA clusters | committed "untested on GPU", no number | n/a |

Kept: `TA_F8N24` (default on; heads of 16: PV as m64n24k32, V^T box of 16 rows, ones at rows 16-23): core -3.2 .. -4.9 % on 4 x 16, module 64:4:16 unfused hi -1.7 .. -2.8 %, bit-identical. The ablation bits `TA_ABL` of main's core stay (one mask, also used for main's variants).

### End level (FoldForge, 200 steps, TRUNK seeds, same protocol as the bf16 baseline; "inside" = max same-trunk-seed RMSD to bf16 <= max(1.1 x the largest bf16 seed-pair RMSD, 0.5 A))

| model | route | targets (FoldBench picks) | bf16 trunk seeds | FP8 same-seed RMSD vs bf16 | inside |
|---|---|---|---|---|---|
| Protenix v1 | hi fused / hi + lo fused (calibrated, no overflow) | 9 of 18 (the rest not run with FP8) | 4 | max 0.08-1.01 A (hi), 0.05-2.49 A (hi + lo) against bf16 seed-pair max 0.18-2.44 A | 9 / 9, 9 / 9 |
| Boltz-2 | hi fused / hi + lo fused | 7 | 4 | 0.16-4.13 A against bf16 0.21-4.83 A; **9c90: 12.6 A, ipTM 0.41 vs 0.71-0.79** (bf16 seeds agree to 0.2 A) | 6 / 7 (9c90 out, both planes) |
| OpenDDE | hi / hi + lo, dynamic V^T scale, unfused | 16 | 8 (4 on 8 targets) | max 0.03-4.97 A (hi), 0.04-2.04 A (hi + lo) against bf16 seed-pair max 0.12-6.41 A | 16 / 16, 16 / 16 |
| OpenDDE | hi / hi + lo, static per-site scale (the route `enable_fp8_pv_for` sets) | 16 | 8 (4 on 8 targets) | max 0.03-2.03 A (hi), 0.04-4.86 A (hi + lo) | 15 / 16 each (7g1v: 1.60 A against 1.26 A; its bf16 seeds already jump between two modes 1.25 A apart) |

ipTM: OpenDDE max |d ipTM| to bf16 0.000-0.04 (9c90 0.02-0.04, bf16 range 0.708-0.776); Protenix v1 <= 0.013; Boltz-2 <= 0.073 outside 9c90.
Boltz-2 9c90 is bistable under numerical noise (relative noise on the TriAttn output of 0.5 % flips 2 of 4 draws, 1 % 1 of 4, 2 % 4 of 4; the bf16 seeds agree because that path is deterministic), but a strict reading of the
acceptance rule fails there, so Boltz-2 is not whitelisted. Not applicable: Chai-1 (64-channel heads: no H100 CUDA inference core), ESMFold2 (no TriAttn); OpenFold3 / AF3: checked in the widened section below.

### End level, widened scope (2026-11-15, branch `fp8/e2e`; shipped route through `enable_fp8_pv_for`, S gates as shipped)

Harness: `benchmarks/fp8_e2e/` (`probe.py` call sites, `e2e2.py` runs, `an3.py` / `bz.py` tables, `results.txt` raw tables). Same protocol as above (FoldForge `miniworld` backend, 200 steps, diffusion_seed 0, 1 sample, TRUNK seeds, acceptance = max same-trunk-seed RMSD
of FP8 vs bf16 <= max(1.1 x the largest bf16 trunk-seed-pair RMSD, 0.5 A)). Two differences to the tables above, both on purpose:

- **The bf16 baselines of this section were re-run in the current environment** (FoldForge 52a5877 + this engine). bf16 is bit-deterministic here (9c90 trunk seeds 0-3 rerun in a second job: 4 / 4 bytes equal), but it is NOT the bf16 of the earlier
  rounds (Boltz-2 9c90 seeds 0 and 1: 12.5 A apart from the earlier files; Protenix v1 8chi 0.05 A), so earlier bf16 files must not be mixed with new FP8 files.
- **The earlier end-level runs removed every S gate** (targets of 52-316 residues). With the shipped gates the 384:12:32 (OpenDDE, from L1024) and 256:8:32 (from L768) cores never engage at those sizes, so those runs measured the FP8 core where it is not shipped.
  The runs below use the shipped route, with targets large enough to engage the gates (residue counts 660-1130; the token bucket is the next multiple of 128, the template stacks of every model engage from L256).
- The static scale is calibrated on the first eager call of each site and kept for the whole process (all targets and seeds of a job share it: the stress case of the contract). Targets whose bf16 trunk seeds already differ by > 5 A
  are weak probes (the 1.1 x rule then accepts almost anything); they are listed, but the conclusions rest on the others.

#### Call sites per model (probe: one trunk run of a 316-residue target, bucket 384, every S gate removed so that all sites show; counts are core calls of one `run_trunk`)

| model | sites (d_pair : heads : channels, kind) | calls | engaged by the shipped route |
|---|---|---|---|
| Protenix v1 | 128:4:32 fused; 64:4:32 front-made (templates) | 1040; 160 | whitelist "fused": only the 128 sites; "unfused" adds the templates (gate L256) |
| Protenix v2 | 256:8:32 front-made (trunk; gate L768); 64:2:32 front-made (templates, gate L256) | 1040; 160 | "unfused" only (no fused site in v2) |
| OpenDDE | 384:12:32 cuBLAS front + standalone pass (trunk; gate L1024); 64:2:32 front-made (templates); 12-head sites at ~1.67 x L (8 calls at 1280 for L768) | 1040; 160; 8 | "unfused" |
| Boltz-2 | 128:4:32 fused; 64:4:32 front-made (templates) | 1360; 160 | not whitelisted |
| OpenFold3, AF3 | 128:4:32 fused; 64:4:16 front-made (templates, gate L256) | 1040; 160 | not whitelisted (candidates below) |
| RoseTTAFold3 | 128:4:32 fused only | 1040 | not whitelisted |
| IntelliFold2 | 256:8:32 front-made, 4 sites only (160 calls; no trunk site on this path) | 160 | not whitelisted |
| Chai-1 | 64:4:32 front-made, 160 calls only (trunk not on this path) | 160 | not applicable |
| ESMFold2 | no template path, no TriAttn site | 0 | not applicable |

A model whose template inputs are absent for a target runs bit-identical to bf16 below its trunk gate (Protenix v2 / IntelliFold2 8gn4 / 8ad9 / 8ee9: 0.00 A, nothing engaged).

#### Results per model (n = trunk seeds per target; RMSD in A; "bf16 max" = largest bf16 pair RMSD; "FP8 max" = largest same-seed FP8 vs bf16)

| model, route | targets | bf16 seeds | bf16 max | FP8 max | inside | max abs d ipTM |
|---|---|---|---|---|---|---|
| Protenix v1, fused + unfused (templates, L256 / 384) | 9 small (7g1v 7y37 7yey 8ad9 8chi 8d78 8ee9 8ey7 8fb3) | 4 | 0.20-2.51 | 0.07-1.03 | 9 / 9 | 0.015 |
| Protenix v1, fused + unfused, large | 7zau 8id0 8fxv (L768 / 896 / 1152) | 4 | 4.35-22.69 (7zau an unstable probe) | 0.37-12.83 | 3 / 3 | 0.036 |
| OpenDDE, shipped, L >= 1024 (gate 1024 engaged) | 8d3i 8fxv 8ifx | 4 | 1.30-8.65 | 0.19-6.44 | 3 / 3 | 0.003 |
| OpenDDE, gate 768 for 384:12:32 (see below) | 16 targets of 660-812 residues | 4 (8 on 8scr) | 0.56-29.74 | 0.10-28.93 | 16 / 16 | 0.112 (8dn7, bf16 spread 29.7 A) |
| Protenix v2, "unfused" (256:8:32 from L768, templates from L256) | 11: 7y37 8ad9 8ee9 8gn4 8id0 8vdn 8x3q 8p7u 8ifx 7zau 8fxv | 4 (12 on 7y37) | 0.24-15.13 | 0.00-12.95 | 11 / 11 | 0.179 (8fxv, unstable) |
| OpenFold3, fused + unfused (templates 64:4:16) | 8ad9 8ee9 7y37 8gn4 | 4 | 0.30-2.18 | 0.33-0.55 | 4 / 4 | 0.008 |
| AF3, same | 8ad9 8ee9 7y37 8gn4 | 4 | 0.06-0.69 | 0.04-0.55 | 4 / 4 | 0.011 |
| RoseTTAFold3, fused | 8ad9 8ee9 7y37 8gn4 | 4 | 1.49-2.46 | 0.46-2.24 | 3 / 4 (7y37: 2.24 against 1.49 x 1.1; its ipTM is 0.10, a failed prediction, bad probe) | 0.002 |
| IntelliFold2, fused + unfused (4 template sites, 256:8:32) | 8id0 8j2w 8vdn 8x3q (+ 8ad9 8ee9 nothing engaged) | 4 | 0.22-4.73 | 0.00-0.76 | 6 / 6 | 0.001 |

Per target rows are in `benchmarks/fp8_e2e/results.txt`. Worth knowing: Protenix v2 7y37 flips once in 12 FP8 trunk seeds (4.26 A at seed 1); its bf16 trunk seeds have a second mode too (4.31 A pair maximum over 12 seeds). With 4 bf16 seeds the same
target reads 0.21 A and the FP8 flip is OUT (4.26 A): the verdict depends on the number of bf16 seeds, which is why 7y37 was run with 12.

#### OpenDDE gate 768 for 384:12:32 (S gate of the standalone V^T pass; `FP8_PV_MIN_S[32]` and `FP8_PV_FRONT_MIN_S[(384, 12, 32)]` = 768, evaluation override only, the shipped gate is unchanged)

16 targets in token buckets 768 / 896 (residues 660-812: 7zau 8cjd 8j2w 8auz 8id0 8bf4 8dn7 8vvl 8vdn 8phb 8scr 8p7u 8avz 7u9i 8x3q 9jf5), 4 trunk seeds, FP8 core engaged on the 12-head sites (1040 calls per trunk) that are bf16-identical under the shipped gate:
16 / 16 inside (8scr: with 4 bf16 seeds 1.38 A against 1.05 x 1.1, OUT; with 8 bf16 / 8 FP8 seeds 2.31 A against 3.04 x 1.1, inside). The nine targets whose bf16 trunk seeds agree to < 4 A
(8id0 0.10 A against 1.14, 8avz 0.30 / 0.56, 8x3q 0.38 / 0.93, 8scr (8 seeds), 7u9i 1.00 / 1.76, 8j2w 0.70 / 2.70, 8p7u 0.80 / 3.05, 8vdn 0.11 / 2.71, 8vvl 2.92 / 3.75) are all inside; no saturation flag, no redo, peak below 0.43 of the e4m3 range.
Decision: the numbers allow 768 (OpenDDE gate 768: 16 / 16 inside, no saturation), but the gain is -2 .. -3 % of the module at L768 and the shipped code has one gate per channel count (`FP8_PV_MIN_S` is keyed by channels, so a 384:12:32-only gate needs a per-config key in `_F8Core`):
not changed here (optional, low priority).

#### Boltz-2 9c90 (data for the whitelist decision; Boltz-2 is NOT whitelisted)

24 trunk seeds, bf16 and FP8 (fused 128:4:32 hi plane, as `enable_fp8_pv_for` would set it); bf16 clusters at 3 A: 5 modes with 8 / 6 / 7 / 2 / 1 seeds, **bf16 itself spans 15.3 A between trunk seeds, ipTM 0.40-0.85** (12 seeds: 15.31 A, 0.407-0.813); the earlier "4 seeds agree
to 0.2 A" does not reproduce in the current environment (seeds 0 and 1 are in the low-ipTM mode now). Same-seed FP8 vs bf16: 9 of 24 trunk seeds (37 %) end in a different bf16 mode (flips, 3.8-14.1 A), 15 of 24 stay (0.08-0.92 A);
by the acceptance rule FP8 is inside (13.68 A max against 1.1 x 15.31 A). Population view: the fraction of trunk seeds in the low-ipTM mode (ipTM < 0.6) is 9 / 24 for bf16 and 8 / 24 for FP8 (modes 0-2 hold 8 / 6 / 7 bf16 and 8 / 8 / 7 FP8 seeds, mode 3 two against one, mode 4 one against zero).
With the templates too (fused + unfused, 12 seeds) 4 / 12 flip. Both routes leave the mode population unchanged, but a given trunk seed does not keep its mode (as with 0.5-2 % noise on the TriAttn output, see limits). 9c90 is the only Boltz-2 target of the 7 that fails the strict reading;
whether a target whose own bf16 trunk seeds fall into five modes is a valid probe is the open decision.

#### Scale / guard contract in these runs

288 FP8 cases (Protenix v1 48, Protenix v2 52, OpenDDE 80 (68 + 12), Boltz-2 36, OpenFold3 16, AF3 16, RoseTTAFold3 16, IntelliFold2 24): `take_overflow()` returned False after every forward, 0 redos (`rescale_all` never called). Per-call peak of the V^T planes
as a fraction of the e4m3 range (448; a calibration call is exactly 1.0, excluded; the calibration leaves a 4 x headroom, i.e. 0.25 at the calibration input), max over all calls / worst p99:

| model | sites | log2(scale) range | max peak | max p99 |
|---|---|---|---|---|
| Protenix v1 | 104 fused + 4 templates | -4 .. -1 | 0.643 (L256) | 0.500 |
| Protenix v2 | 104 (256:8:32) + 4 | -10 .. -2 | **0.857** (256:8:32, L768) | 0.571 |
| OpenDDE | 112-115 (384:12:32) + 4 | -9 .. -1 (and -99, below) | 0.429 | 0.393 |
| Boltz-2 (data) | 136 + 4 | -4 .. -2 | 0.429 | 0.321 |
| OpenFold3 / AF3 | 136 + 4 | -4 .. -1 | 0.393 / 0.429 | 0.321 / 0.357 |
| RoseTTAFold3 | 136 | -4 .. -2 | 0.500 | 0.412 |
| IntelliFold2 | 4 | -4 .. -2 | **0.929** (L896, 256:8:32) | 0.929 |

Findings for the contract: (1) the 4 x headroom held in every run, but two sites reach 0.86-0.93 of the range at a later call of a process-wide scale taken on another target (no flag: the planes are 8-14 % below the cap); a target with a 10 % larger V peak would
have saturated and needed the redo. (2) In the OpenDDE runs at L >= 768 some sites calibrate at **log2(scale) = -99** (35 of 1380 heads of the 12-head sites; probably the 3 extra 12-head sites that appear only at ~1.67 x L, whose V is then zero at the calibration call (not isolated); `vscale_from_amax` clamps amax at 1e-30). No flag, so V stays zero there, but
a head calibrated at exactly zero saturates on any later non-zero V, and `rescale_all(4)` x 3 (the redo limit) cannot recover from 2^-99: a floor on the scale (or re-calibrating a head with amax == 0) is the guard to add; not changed here.

## Known limits

- Boltz-2 9c90 is bistable under any numerical noise of 0.1-2 % on the TriAttn output; V on e4m3 flips it (hi + lo too: 2 of 4 trunk seeds). In the current environment its own bf16 trunk seeds fall into 5 modes (24 seeds, 15.3 A, ipTM 0.40-0.85) and FP8 changes the mode of 9 of 24 same trunk seeds while the mode population stays equal (see the widened end-level section). Hence whitelist only, and Boltz-2 is not on it (open decision).
- Scales are per tensor (TriMul) or per head (TriAttn), powers of two. The calibration caches are keyed by weight pointer; `weights_changed()` does not invalidate it (inference only today: after an in-place weight update the cache is stale).
- A saturation on an out-of-distribution input is caught only after the forward (lazy guard); a caller that does not read `overflow_peak()` runs unguarded.
- Prebuild: the TriAttn FP8 variants (F8 core hi / hi + lo, the V^T-plane fronts VTW 1 / 2, bf16 and fp16-exact: 6 artifacts) are in the prebuild plan (`kernels/triangle_attention/cuda/__init__.py`), so they ship in the wheel. The TriMul FP8 variants (the FP8-plane K1 tile(s) and the K3 with `K3_CSCALE`, any-L row form and for D256 also the token-flat form; D256 K1 per length of the requirements row) are in the plan too (`kernels/trimul_inproj/cuda/__init__.py`); `_h100_fp8` builds nothing (cuBLASLt e4m3 is a library call). Verified: a prebuilt bundle (`_prebuild --only trimul_inproj cast`) + an EMPTY local JIT cache + an FP8 forward (eager calibration, 3 calls, then a graph capture and replay) at L256 / 300 / 512 / 1024, D128 / D256, both directions: 0 compiled, 5 (D128) / 6 (D256) artifacts resolved from the packaged root, none from the local cache.
- cuBLASLt e4m3 heuristics and `cvt.satfinite` depend on the CUDA 12 toolchain.
- The TriMul and TriAttn backward are not ported (e5m2 no-go; e4m3 backward no speed gain as built).
- End to end gain is small: TriMul FP8 1.5-4.5 % (compile + graph warm forward; measured before the 2026-11-15 `fp8/tm2` fixes and gates, not re-measured).
- TriAttn unfused sites: the static V scale is calibrated on the first eager call of a site (as the fused front's); after `take_overflow()` the caller must `rescale_all()` and redo. A site not reached through `_infer` has no site key and runs the dynamic pass.
- TriAttn FP8 below the S gate (standalone-pass sites: 1024 tokens, 512 for 16-channel heads; front-made sites: see above, from L256 for d_pair 64) runs the fp16 core: short targets see no change at all.

## Verification of the merged branch (`fp8/main`, 2026-11-15)

- Default (non-FP8) route, TriAttn: 25 module shapes (d_pair 64:4:32 / 64:2:32 / 64:4:16 / 128:4:32 / 256:8:32 / 384:12:32 at L256 / 512 / 1024 / 1536 rotating start / end / flip / masked, plus 128:4:32 L2048), output bytes equal to main ec11768fc: 25 / 25 identical, before and after the macro cleanup. TriMul default (D64 / 128 / 256 / 384, both directions, L256 / 300 / 512 / 1024 / 1536): 40 / 40 identical to main after the cleanup.
- FP8 route against `fp8/vt` (c7bf0ee76; before `TA_F8N24`): hi and hi + lo, the same 25 shapes with every S gate removed, output bytes: 25 / 25 and 25 / 25 identical (`TA_F8N24` is bit-identical, as reported by the core work). After the cleanup against the merged state before it: 25 / 25 and 25 / 25 identical.
- TriMul FP8 planes (D128 / D256, both directions, L256 / 300 / 512 / 1024 / 1536) and the experimental FP8-K1 route: output bytes equal to `fp8/3.0.0` (e58b9488c) after the cleanup, 20 / 20 and 20 / 20.

## Verification of `fp8/tm2` (TriMul FP8 glue fixes, gates, prebuild plan; 2026-11-15)

- Output bytes: TriMul FP8 planes (D128 / D256, both directions, L256 / 300 / 512 / 1024 / 1536) 20 / 20 identical to `fp8/main`; TriMul default (D64 / 128 / 256 / 384, both directions, same L) 40 / 40 identical (the header edit is under `TMN_K1_F8`; the default K1 builds re-key through the header text, their SASS is unchanged).
- CPU tests `tests/cpu/test_prebuild_plan.py`, `test_requirements.py`, `test_judge.py`: 1188 passed. GPU tests `tests/gpu/test_trimul_fp8_planes.py`, `test_trimul_fp8_k1.py`: 30 passed (one job, 1 GPU).
- Prebuild: see "Known limits" (0 compiled with a prebuilt bundle and an empty JIT cache).
