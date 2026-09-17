# Autotune search-space audit — 2026-09-16

## Current policy: restore unproven schedule reductions

All eight CSV grids edited during this audit have been restored to their original
candidate sets (and original file bytes). No new tile/warp/stage reduction based on
winners from A6000/A5000/A100 remains. A winner on those GPUs does not establish
optimality on other GPUs or on unmeasured workloads.

The earlier representative compiler-policy set `(warps, stages) = (4,2), (4,3), (8,3)`
was a heuristic exploration budget, not a correctness or dominance proof. B2B's added
restrictions on BLOCK_N, covering extents, warps and stages were also withdrawn.
The previous 92% reduction and 396-config B2B space are no longer the current policy.

| Kernel | Withdrawn heuristic space | Restored current space |
|---|---:|---:|
| `transition_fwd_b2b_triton` | 396 | 27,944 |
| `cond_transition_bwd_gemm_swiglu_triton` | 620 | 2,880 |
| `adaln_gemm_gate_triton` | 589 | 2,880 |
| `cond_transition_fwd_b2b_saveact_triton` | 454 | 2,160 |
| `adaln_fwd_gate_triton` | 391 | 2,160 |
| `rmsnorm_adamod_bwd_triton` | 357 | 1,980 |
| `rmsnorm_adamod_fwd_triton` | 334 | 1,584 |
| `layernorm_linear_fwd_triton` | 299 | 1,440 |
| **Total** | **3,440** | **43,028** |

## Corrections retained

- B2B driver tuples come from Transition/SwiGLUFFN module declarations, keeping
  hidden width and expansion paired, and respecting the K <= 128 / K > 128 dispatch.
- The driver actually constructs the requested widths and expansion; previously it
  silently repeated K=128 under unrelated width labels. MSA probes use the existing
  module builder's eight-row fixture and flattened-row key.
- Small B2B tuples are pair (K=64, ND=128), pair (128,256), pair (128,512),
  MSA (64,256), MSA (128,512), and atom (128,256): 38 fallback driver cases.
- Each known equivalent width group is deduplicated independently; unknown widths
  remain. Conditional/gate dimension pairs sharing their first dimension are retained
  instead of losing one through dict conversion, including (768,384)/(768,768).
- The sweep page uses the real CSV parser, handles materialized lists without
  multiplying their marginal axes, and labels fallback cases/upper bounds honestly.
- Runtime cache readers can reuse retained winners under a proved narrowing without
  changing cache stamps or per-workload coverage. Candidate growth still misses.
  CuTe kwargs-only candidate inputs are normalized for membership checks.

## Why the original page said 2,352,000

It multiplied 84 fallback driver cases by 28,000 raw Cartesian configurations.
The parser admits 27,944. The existing `_prefer_covering_b2b` then leaves 560
candidates for each current K=64/128 probe. With corrected dimensions the current
page therefore shows **38 cases × 560 = 21,280**, not 3,960. This driver-cost metric
is neither a full count of mode/flag/helper/compiler launches nor a build ETA.

Git history: `79cca4eb` restored 128/256 covering tiles after a measured regression;
`f306ab55` split BLOCK_N from BLOCK_K_ND, multiplying the raw grid by five. The CSV
kept the full product while runtime pruning rejected much of it. Unrelated width
unions and the fixed-width driver further inflated the old display/workload list.

The pre-existing minimum-covering B2B rule is **not a universal cross-GPU optimality
proof** either. Its tile preference came from earlier performance measurements;
GROUP_M is redundant within its one-output-tile covering branch. This audit leaves
that existing kernel policy unchanged rather than claiming all removed schedules
are mathematically invalid. Device-specific performance pruning needs measurements
on the target device; compile/resource exclusions need actual compiler/device limits.

## A6000 candidate preservation

After restoration, all **274/274 cache-entry winners**, **1,421/1,421 entry candidates**,
**130/130 workload winners** and **646/646 workload candidates** in the eight kernels
remain. Atomic LayerNorm retains **80/80 entry winners and 400/400 entry candidates**.
The [JSON audit](autotune-space-20260916.json) records per-kernel results, old specs,
current hashes and withdrawn heuristic counts. Presence is not a claim that the
current environment/source identity accepts every cache or that every shape is tuned.

## Remaining atomic LayerNorm space

`layernorm_bwd_atomic_triton` remains **264,960 = 184 cases × 1,440 configs**.
Cases are 6 pair lengths × 7 widths × 2 dtypes, 6 token lengths × 7 widths × 2 dtypes,
and 4 atom lengths × 2 widths × 2 dtypes. Configs are 5 BLOCK_K values × 8 BLOCK_M1
values × 6 warp values × 6 stage values; there is no kernel-specific shape prune.
This is a fallback upper bound, not proof every combination is reached, compiles,
is accurate or is worth measuring. No new guessed pruning was applied here.

Its A6000 top-1 records have 54 distinct configurations, all five BLOCK_K values,
all six stage values, and two winners with 32 warps. Simply discarding large values
would lose known winners; keeping only known winners would exclude unexplored GPUs.
The kernel supports covering and two-pass schedules, so BLOCK_K < N is not inherently
invalid. The 141 module rows, including 12 explicit headroom rows, remain intact.

## Validation

Isolated cssb3 Slurm CPU compute job **46552**: **127 passed**, 9 expected cache warning
cases, 29.18 seconds. Regression tests assert exact equality with all eight original
candidate sets, all previously legal stored entry/workload candidates retained,
actual driver dimensions, conditioning pairs, parsing and cache-reader behavior.
Ruff passed for the updated tests. No GPU performance/numerical benchmark was run.

The [sweep page](../autotune-sweep-grid.html) is regenerated. Derived dispatch coverage
remains explicitly **UNVERIFIED** because the trace is stale/incomplete. Fallback
per-op probes do not certify the authoritative module `build all` work list. Existing
user edits, stored cache files, shards and ongoing cache jobs were preserved.

## Atomic LayerNorm shape audit

The current HTML has been regenerated and links this audit from the atomic LayerNorm row.
Its **264,960 = 184 fallback cases × 1,440 configs** remains a fallback search upper bound;
it is not relabeled as a measured production workload count. Tile/warp/stage candidates
were not reduced.

Slurm job **46555**, allocated cssb3 gpu07 A100, used fake tensors with **SM86 dispatch**
to follow the current module code. It traced **323** training invocations (first registered
length of each row, default and forced-atomic paths); **155** invoked this atomic kernel,
with **zero errors**. This is a targeted call-path audit, not a full derive plan or GPU
performance result. It does not prove the absence of a path at every length/on every GPU.

| Issue in the fallback HTML shapes | Evidence / interpretation |
|---|---|
| Pair D=768 | No pair-width-768 atomic call in the traced declared modules. The fallback unions single-stream widths into the pair ladder. There are 12 extrapolated cases (6 lengths × 2 dtypes), costing 17,280 in that metric. This is not a universal unsupported-shape assertion. |
| Atom D=64 | The traced 64-wide non-pair caller is MSA, M=8×128=1,024. The fallback uses atom row-count probes to cover MSA and labels them atom_single; these are not evidence of a real 64-wide atom embedding. |
| Pair/token D=16 | Both are real calls from native checkpoint LayerNorm declarations; do not discard this width as an atom-only artifact. |
| Pair D=267 missing | Atomic call observed at M=128², N=267. |
| Token D=267/451/831/833/2560 missing | All observed in native checkpoint norm calls. |
| Diffusion augmentation absent from displayed token length | N=768 observed at M=48×128=6,144, and atom N=128 at M=48×1,024=49,152. Flattened rows matter for norm tuning. |
| FP32 probe expansion | All atomic calls in this targeted trace had BF16 activation + FP32 auxiliary/affine tensors. The fallback's standalone FP32 probes express kernel support; they are not proof that a declared FP32 model path calls this kernel. |

Observed physical inputs at the representative lengths:

| Caller shape | Flattened M | Observed N |
|---|---:|---|
| atom_single (bfloat16+float32) | 49,152 | 128 |
| msa_token (bfloat16+float32) | 1,024 | 64, 128 |
| token_pair (bfloat16+float32) | 16,384 | 16, 64, 128, 256, 267, 384, 512 |
| token_single (bfloat16+float32) | 128 | 16, 64, 128, 256, 267, 384, 451, 512, 768, 831, 833, 2560 |
| token_single (bfloat16+float32) | 6,144 | 768 |

The leaf `layernorm_native` declarations and observed kernel calls agree on pair widths
16/64/128/256/267/384/512 and token widths
16/64/128/256/267/384/451/512/768/831/833/2560. MSA contributes 64/128; the observed
atom activation width is 128. Internal pair calls from MSA modules must still be
classified by their physical activation, not just by the parent module's stream.

No shapes were deleted based solely on this single-architecture, representative-length
trace. The obsolete generic per-op fallback list is the discrepancy being reported;
`build all` uses module work and cannot be certified or declared missing these shapes
from this fallback page alone. The registered module cases already include the widths
missing from the fallback HTML. A complete current per-architecture derivation is needed
to replace that list authoritatively, including mode, dtype, augmentation and layout.

HTML/space regression job **46556**: **127 tests passed**. The candidate CSVs remain
byte-identical to their pre-audit originals, and no saved GPU cache was altered.

## Model-based coverage and pair D=768

Coverage must start from the selected model/checkpoint/config set, not `op_units()`.
For each model, follow real call sites through templates, trunk/MSA, atom windows,
diffusion conditioning, denoiser and confidence, then collect each dispatched kernel's
own shape, dtype/affine precision, mode, layout and augmentation. Merge identical
workloads only after that and apply GPU support policy. Check usable measurements
against this union. A model constructor census is evidence for dimensions, not a
complete runtime coverage certificate; the stale module registry trace cannot supply
that certificate either.

Current local model defaults and conditioning code were cross-checked with the actual
constructor census in `docs/checkpoint-shapes-20260915.json`:

| Model/config | Trunk pair width | Pair-conditioning LayerNorm width | Pair LN D=768 found? |
|---|---:|---:|---|
| AF3 default | 128 | 267 = 128 + 139 relative features | No |
| Protenix base default v1.0.0 | 128 | 256 = 128 + 128 | No |
| Protenix-v2 | 256 | 512 = 256 + 256 | No |
| OpenDDE default | 384 | 256 = 128 + 128 after trunk projection | No |
| ESMFold2 configured checkpoint | 256 | 512 = 256 + 256 | No |
| MiniWorld team-gm presets | 128 (atom pair 16) | Preset pair width is 128; token DiT single width is 768 | No in checked presets |

OpenDDE is the important potential counterexample: its current config explicitly sets
`c_z_pair_diffusion=128`. Shared `DiffusionConditioning` normalizes trunk width 384,
projects to 128, concatenates with a 128-wide positional encoding, and normalizes 256.
It does not concatenate two 384-wide pairs in that default configuration.

However, the same constructor with `c_z=384, c_z_pair_diffusion=None` defaults the
latter to 384 and constructs **LayerNorm(768)** on concatenated pair tensors. Thus
"no selected checkpoint default currently uses pair LN D=768" is supported; "no model
or configuration can use it" is false. A new/custom configuration must participate
in the coverage contract before its build cases are excluded or included.

The 768-wide norms in the five-model constructor census belong to token diffusion
AdaLN/conditioning/output streams. An internal GEMM expansion of width 768 is a
separate question: a GEMM ND axis must not be added to LayerNorm N unless the call
path actually normalizes that expanded tensor. This audit makes no blanket claim
that no intermediate pair-shaped tensor anywhere can have width 768.

The JSON audit records source hashes for the checked config and conditioning files.
This follow-up used code and constructor evidence; it did not rerun all full models
or change any candidate grid, driver shape, saved cache or model configuration.
