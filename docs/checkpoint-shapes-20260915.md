# Checkpoint shapes and kernel wiring — 2026-09-15

Source: actual model constructors on cssb3/A100, Slurm 43944. No checkpoint tensors were loaded for the census. This records parameter dimensions, not a completed cache build or a runtime dtype census. The companion JSON groups identical module signatures and retains every module path. Configurations: AF3 default; Protenix base default v1.0.0; Protenix-v2; OpenDDE default; installed ESMFold2 checkpoint config.

| Component | AF3 | Protenix v1 | Protenix v2 | OpenDDE | ESMFold2 |
|---|---|---|---|---|---|
| Template TriMul pair/hidden | 64/64 | 64/128 (asymmetric; reference) | 64/64 | 64/64 | no template stack |
| Trunk TriMul pair/hidden | 128/128 | 128/128 | 256/256 | 384/384 | 256/256 |
| Template triangle attention pair / heads × head width | 64 / 4×16 | 64 / 4×32 | 64 / 2×32 | 64 / 2×32 | — |
| Trunk triangle attention | 128 / 4×32 | 128 / 4×32 | 256 / 8×32 | 384 / 12×32 | 256 / 8×32 |
| Token DiT single/condition/pair | 768/384/128 | 768/384/128 | 768/384/256 | 768/384/128 | 768/768/256 |
| Token attention heads × width | 16×48 | 16×48 | 16×48 | 16×48 | 16×48 |
| Single trunk attention | 16×24 | 16×24 | 16×24 | 16×24; refiner 8×48 | — |
| Atom attention | local 32×128; 4×32 | local 32×128; 4×32 | local 32×128; 4×32 | local 32×128; 4×32 | SWA + 3D RoPE; 4×32 |
| Atom FFN | conditional 128/128, ×2 | conditional 128/128, ×2 | conditional 128/128, ×2 | conditional 128/128, ×2 | bare SwiGLU 128→256→128 |
| MSA input/pair/outer | 64/128/32 | 64/128/32 | 128/256/32 | 128/384/32 | 128/256/32 |
| MSA value heads × width | 8×8 | 8×8 | 8×8 | 8×8 | 8×16 |
| LayerNorm widths | 16,64,128,256,267,384,768,831 | 16,64,128,256,384,768,833 | 16,64,128,256,384,512,768,833 | 16,64,128,256,384,768,833 | 128,256,384,451,512,768,2560 |

AF3's unused `c_hidden_mul=128` constructor argument does not describe its template TriMul: the packed 128-output projection contains two branches of 64 channels. Protenix v1 really has 128 channels per branch. Neither channel dimensions nor expansion widths may be clamped to a neighbouring shape.

## Executable build coverage

`src/miniworld_engine/kernels/registry_module.csv` remains the build source of truth. The table expands from 53 to 141 rows. Existing rows and length ladders are preserved. Added rows cover square TriMul 64 and 384, exact template/trunk triangle head layouts, ×2 transitions, MSA transitions, ESMFold2 768/768 conditioning, and exact leaf operations for projected attention, sigmoid gate+output projection, bare SwiGLU, native FP32-affine LayerNorm, FP32-affine LN+BF16 projection, and RMSNorm+modulation.

The new cases live in `autotune/checkpoint_cases.py` and are included by `builder.cases()` and `CASE_NAMES`; they are not documentation-only shapes. Atom pair projection uses actual 32-query/128-key windows, never a dense atom×atom allocation. Native token output LayerNorm declares both non-augmented and diffusion (eval 5/train 48) rows; equal channel dimensions do not imply equal launch shapes. Existing all-BF16 LN+projection probes are retained alongside native FP32-affine probes. The registry change invalidates the build plan fingerprint; old derived/cache artifacts are not evidence that these added rows have been tuned.

Token and atom length ladders are unchanged. Runtime token padding remains multiples of 128; atom padding retains its existing rule. MSA padding buckets remain 2048/4096/8192/16384. The module builder's MSA smoke tensor uses eight rows: the added rows establish **channel** coverage, not exhaustive MSA-depth performance qualification. Length clamping must not be confused with changing the actual model input shape.

## Connected compatible operations

- AF3 and shared AF-family attention: projected square QKV core (previous connection); now sigmoid gate+output projection as one public composite op, with the engine's existing measured fused/split dispatch.
- AF-family atom coordinate projection: FP32-affine LN+projection (128→3) when its activation is BF16. AF3 trunk pair-to-atom projection (128→16) uses the same helper.
- AF3 local atom blocks: FP32-affine LN+pair projection (16→12); existing square attention kernel is not applied to rectangular atom attention.
- AF3 unconditioned diffusion transitions: existing bare SwiGLU kernel after the model's own normalization. No synthetic residual.
- AF-family and native shared MSA: gate+output projection; preserve each implementation's mask, softmax, value-head widths and residual ownership.
- ESMFold2 atom blocks: existing bare SwiGLU and existing RMSNorm+activated-conditioning projection (scale, shift, raw gate). Preserve `nn.RMSNorm(eps=None)`'s dtype-dependent epsilon and exclude MP-normalized projection weights. SWA QK/RoPE and attention/output kernels were already connected.

The newly connected paths select MiniWorld for eligible CUDA BF16 inference tensors; PyTorch and cuEquivariance retain their backend selection. They do not cast FP32 coordinate streams into BF16 merely to enter a kernel. Public composite ops have autograd support; enabling their checkpoint adapters for training is a separate qualification.

## Unsupported combinations remain explicit

- Protenix v1 asymmetric template TriMul 64/128: engine whole TriMul requires equal input/hidden widths. Retain the reference equation; rebuilding a 64/64 cache cannot fix it.
- AF3/Protenix/OpenDDE local atom attention has 32 queries and 128 keys. Engine public projected attention currently assumes square QKV; retain the local reference core, while fusing compatible surrounding operations.
- MSA pair-weighted attention and outer-product contractions retain their existing equations. The engine bias-only attention primitive's row layout is not interchangeable with the MSA depth axis.
- MPLinear and full-MP modulation must apply their effective weight normalization; raw-weight fusion is bypassed.

## Projection dispatch measurement

A100 80GB PCIe, native BF16 parameters except FP32 norm parameters, no autocast,
`torch.compile(fullgraph=True)` and explicit CUDA graph for **both** candidates.
CUDA-event timing averages 100 replays after warmup; pair input uses L=128,
single/coordinate input uses 4096 rows. The reference is the existing MiniWorld
standalone LayerNorm plus native linear, not a pure-PyTorch backend measurement.
The current cache uses heuristic subsets for missing tuned entries, so these are
current-cache dispatch decisions, not evidence of fully tuned kernel limits.

| Projection | Linear compute | Fused ms | Existing ms | Decision |
|---|---|---:|---:|---|
| pair 267→128 | native BF16 | 0.246323 | 0.043653 | retain standalone norm + linear |
| pair 256→128 | FP32 | 0.195082 | 0.104776 | retain standalone norm + linear |
| pair 512→256 | FP32 | 0.759255 | 0.315945 | retain standalone norm + linear |
| single 384→128 | FP32 | 0.112323 | 0.054497 | retain standalone norm + linear |
| coordinate 128→3 | FP32 | 0.015913 | 0.017797 | fuse for eligible BF16 input |

The shared checkpoint helper limits LN+projection fusion to output width ≤16.
Wider callers still use the common helper and fall back to their existing norm
and projection. Their standalone norm widths are represented in the registry;
unselected wide fused projection shapes do not inflate the new build table.

## Validation and scope

- Constructor census: Slurm 43944, all five model variants above.
- Engine build/API/layout regression: 719 passed, Slurm 43953_0.
- Core model/backend GPU tests: 31 passed across Slurm 43951_1 (5) and
  43954_1 (26). Includes nonzero-weight numerical comparisons, autograd where
  applicable, fullgraph compilation and CUDA graph replay; ESM atom uses 4096 atoms.
- Final registry/layout checks: 719 passed; projection numeric/compile and dispatch checks: 6 passed (Slurm 44200).
- Final fake-tensor derivation: all 88 added rows, 907 units, 244 op/dtype/key combinations, zero errors.
- Projection numeric/compile and dispatch checks are in
  `FoldForge/tests/test_checkpoint_projections.py`; raw measurements and final
  registry/launcher checks are recorded in `checkpoint-validation-20260915.json`.
- Launcher derivation tests every added channel/dtype/mode/option at the first
  length of its semantic stream, using fake tensors. This checks wiring and key
  construction; it is not a GPU numerical test for every registry combination.

A pre-existing fake-build regression fixture really cleared the shared generated
Triton cache. The fixture now stubs that side effect and validation uses private
Triton/Inductor cache directories. Persistent autotune JSON and saved build shards
were not removed. Initial interrupted/failed runs are superseded by the isolated
passing runs above.

These changes do not certify all model outputs end to end, enable checkpoint
training adapters, rebuild every GPU's persistent cache, or supply a new end-to-end
speedup. The HTML viewer marks old derived-key counts as UNVERIFIED until the build
plan is regenerated; the executable module table is current.
