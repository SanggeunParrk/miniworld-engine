# A6000 SWA inference attribution — 2026-09-11

The default MiniWorld SWA inference disadvantage comes from separate Q/K copies, RMSNorm
and RoPE launches. Compiled PyTorch fuses their work into a single preprocessing kernel.
The current MiniWorld output gate does not explain the difference; FlashAttention is shared.

One physical A6000, L384 / 3072 atoms, A5, width128, four heads of width32, BF16,
mask probability .125, depth1, TF32 off, actual torch.compile and manual CUDA Graph replay.
Unchanged native module benchmark/timer, three samples in each process. No cache misses or
full-grid tuning allowed. Production code and A6000 tile caches were unchanged. Hybrid rows
are temporary diagnostic replacements, not updated production defaults or a new numerical
qualification. These inference results do not establish the A48 training ranking.

| Full SWA inference configuration | Median ms |
|---|---:|
| PyTorch | 0.366944 |
| Default MiniWorld | 0.409216 |
| MiniWorld, RMSNorm replaced with torch | 0.392928 |
| MiniWorld, RoPE replaced with torch | 0.403296 |
| MiniWorld, gate replaced with torch | 0.408928 |
| MiniWorld, RMSNorm and RoPE replaced with torch together | 0.366048 |

A separate profiler run over 30 graph replays attributes the MiniWorld preprocessing to
2 clone kernels (22.04 us total), 2 RMSNorm kernels (17.45 us), and 2 RoPE kernels (32.35 us):
about 71.8 us total. The compiled PyTorch preprocessing kernel takes 26.13 us; the combined
RMSNorm/RoPE replacement takes 25.99 us. These are instrumented kernel durations, not values
to add to or subtract from the unprofiled latency table. The individual replacement gains
are not additive because using torch for both enables cross-operation fusion.

The Q/K views come from interleaved QKV storage. RMSNorm flattens them with `reshape(-1,n)`
in `kernels/rmsnorm/triton/main.py`, which requires the copies observed in the graph. This
copy is part of the current integration, not intrinsic to RMSNorm's mathematical operation.
A stride-aware fused normalization/rotation path could avoid those intermediate buffers.

## RoPE cache verification

The current production sm86 plan requires two RoPE keys; both are usable, with zero missing
or invalid keys. The A6000 RoPE cache contains 22 historical/current entries in total, which
is not the same count as current required keys. The live L384/A5 MiniWorld invocation reads
`bfloat16+float32|shape_key=137439087108` and retains its five cached configs. It never enters
cache-miss fallback. The other required packed key is `shape_key=68719610372`.

`_RoPE3D.backward` uses the same `rope_3d_kernel` and saved shape key, with negated sine;
there is no separate RoPE backward autotuner missing from the build.

[Samples, profile groups, actual cache lookups and configuration](a6000-swa-inference-attribution.json).
Raw traces are under `/home/psk6950/practice/miniworld-engine/.scratch/a6000-2026-09/mw-swa-attribution/`.
