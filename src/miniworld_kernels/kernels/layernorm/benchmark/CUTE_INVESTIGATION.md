# LayerNorm: would CuTeDSL / H100-specific work help? (H100 80GB HBM3, bf16)

Investigation of two questions:
1. **Option ②** — is there forward headroom (non-pow2 d, register pressure)?
2. **Option ①** — would a CuTeDSL backward (H100 persistent grid / cluster reduction) beat our triton?

Method: extend `bench.py` with `--suite fwd_tune` (forward-only + achieved HBM
bandwidth) and `--suite cute_bwd` (fwd+bwd vs quack's CuTeDSL norm). Peak HBM3 =
3.35 TB/s. Sources: `bench_layernorm_fwdtune.out`, `bench_layernorm_cutebwd.out`.

---

## Option ② — Forward: at the bandwidth wall, no headroom (NEGATIVE)

The shipped triton forward already sustains **82–90% of HBM peak** across the
sweep. A low-register variant (`triton/lowreg.py`, bf16-resident tile to dodge the
`next_pow2(N)` padding that spills at d=384/768) ties it to the 3rd decimal — the
pad wastes *registers, not bandwidth*, and we are not occupancy-limited enough for
that to matter. quack's CuTeDSL LayerNorm forward is **3–15% slower** than triton.

| d | M=1048576 triton | lowreg | cute (quack) | triton % peak |
|---|---|---|---|---|
| 128 | **0.1856** | 0.1854 | 0.1886 | 86% |
| 256 | **0.3615** | 0.3614 | 0.3648 | 89% |
| 384 | **0.5351** | 0.5353 | 0.5467 | 90% |
| 512 | 0.7123 | **0.7121** | 0.7184 | 90% |
| 768 | 1.0638 | **1.0632** | 1.0937 | 90% |

**Verdict: do not rewrite the forward.** It is bandwidth-bound and solved; neither
a register-pressure rewrite nor CuTeDSL beats it. pytorch sits at 14–46% peak,
which is the whole story behind the 2–6× "vs PyTorch" speedups.

---

## Option ① — Backward: CuTeDSL wins big, and the win grows with d (POSITIVE)

quack ships no LayerNorm *backward* (only RMSNorm; its bwd kernel takes rstd, not
mean), so we bench **quack RMSNorm fwd+bwd as a proxy** for "how fast is a modern
CuTeDSL norm backward." Its backward uses a **persistent grid of `sm_count`
blocks** grid-striding over M (≈132 partial dw rows + a `rms_final_reduce` pass)
vs our triton partial path's `cdiv(M, block_m)` ≈ 16k partial rows and a per-row
inner loop.

Since cute-fwd ≈ triton-fwd, the entire fwd+bwd gap is **backward**. Isolating
backward-only (fwd+bwd − fwd) at M=1048576:

| d | triton bwd (ms) | cute bwd (ms) | cute speedup |
|---|---|---|---|
| 128 | 0.353 | 0.299 | 1.18× |
| 256 | 0.918 | 0.582 | **1.58×** |
| 384 | 1.474 | 0.893 | **1.65×** |
| 512 | 1.587 | 1.419 | 1.12× |
| 768 | 3.821 | 1.657 | **2.31×** |

Full fwd+bwd speedup (cute vs triton) reaches **1.79× at d=768/M=1M** and
**1.68× at d=768/M=147k**. See `bench_layernorm_cutebwd.md` + PNGs.

**Caveat (honest):** the proxy is RMSNorm — no mean-subtraction and no `db`
reduction. A real LN backward adds a second `[sm_count, N]` reduction (db) + the
`c2 = mean(wdy)` term, ≈5–15% more bwd work. The d≥256 wins survive that easily;
the d=128/512 margins are thinner.

### Why triton loses here
Our triton backward is the weak link, for reasons that are **algorithmic, not
"triton vs cute"**:
- the partial kernel (`triton/partial.py`) runs a scalar `for ri in range(BLOCK_M)`
  row loop instead of a vectorized 2D tile;
- `BLOCK_N = next_pow2(N)` → the d=768 register spill that also caps the forward;
- it emits ~16k partial rows (huge) vs quack's ~132 (one per resident block).

---

## Resolution — triton persistent backward (SHIPPED)

Ported quack's backward *algorithm* into triton (`triton/persistent.py`): a
persistent `NUM_SM * waves` grid grid-striding over row tiles, fp32 dw/db
accumulators carried in registers, vectorized `[BLOCK_M, BLOCK_N]` tiles, one
partial row per block. Correctness matches the old path exactly (fwd/dx cos=1.0,
dw/db cos≈0.99999). fwd+bwd at M=1048576:

| d | triton atomic | **persistent** | quack cute | persistent vs cute |
|---|---|---|---|---|
| 256 | 1.281 | 1.296 | 0.943 | cute wins (kept old path at d≤256) |
| 384 | 2.019 | **1.843** | 1.432 | cute 1.29× ahead |
| 512 | 2.313 | **2.122** | 2.139 | **persistent ties/beats cute** |
| 768 | 4.882 | **2.768** | 2.717 | **within 2% of cute** |

vs the old partial path persistent wins 1.06× (d=384/512) to 1.18–1.22× (d=768).
Wired into `_dispatch_bwd`: **d≥384 → persistent**, d=256 → partial/atomic,
d<256 → atomic. This lifted the shipped `layernorm_kernel` fwd+bwd from 1.32–1.39×
to **1.57–1.64× over the legacy atomic triton at d=768**. At d=256 persistent
regresses vs the old partial, so that case keeps the old dispatch.

Remaining gap to cute is now only at d≤384 (cute's smem reduction + cluster
readiness); closing it needs an actual CuTeDSL LN backward, with bounded upside
over persistent.

## Recommendation

- **Forward:** leave it. Bandwidth wall; cute/low-reg both fail to beat it.
- **Backward:** real headroom (1.2–2.3×, growing with d). Two ways to capture it,
  cheapest first:
  1. **Triton rewrite first** — persistent `sm_count`-block grid-stride + a
     vectorized 2D tile + small partial buffer. Reliable, no new toolchain; should
     recover much of the gap. (Note: a prior grid-stride attempt for the *fused*
     `layernorm_linear` bwd was a negative result — context differs, standalone is
     worth re-testing.)
  2. **Port a CuTeDSL LayerNorm backward** (quack RMSNormBackward + mean/db) for
     the full win and H100 cluster-readiness — larger effort, bounded upside above
     the triton rewrite.
- **Bigger picture:** standalone LN is ~solved; the largest H100 lever remains
  *fusion* (LN folded into the consumer GEMM/gate), already pursued in
  `layernorm_linear` / `transition` / `trimul`.
