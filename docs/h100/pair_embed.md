# La-Proteina pair embeddings and token-pair initialisation on H100

## Scope

Four parameter-owning module classes whose math is "binned one-hots times a matrix", which the kernels compute as a gather of table
rows (`one_hot @ W == W^T[bin]`) without building the one-hots:

| class | file | output | parameters | gradient |
|---|---|---|---|---|
| `PairEmbedAdaLN` | `modules/pair_embed.py` | `pm * (LN(h) gamma[b] + beta[b])`, `h = LN_affine(W [sep | d(xt) | d(sc)]) pm`; bf16 `[B, n, n, 256]` | `weight [256, n_sep + n_xt + n_sc]` (127 + 30 + 30), `ln_weight`, `ln_bias` | weight, LN affine, `gamma`, `beta` |
| `PairEmbedGeom` | `modules/pair_embed.py` | `M W [sep 127 | 4 x 21 distances | 5 x 21 angles | chain product]`, optional affine-free LayerNorm (`ln_eps`); bf16 `[B, n, n, 256]` | `weight [256, 317]` (rounded to bf16 first) | weight |
| `PairEmbedCA` | `modules/pair_embed.py` | `M W [sep 127 | CA distance 30]`, optional LayerNorm; bf16 `[B, n, n, 256]` | `weight [256, 157]` (bf16-rounded) | weight |
| `TokenPairInit` | `modules/token_pair_init.py` | `left_i + right_j + W_rel onehot_rel(i, j) + W_bond onehot(bond_ij)`; fp32 `[B, L, L, 128]` | `w_rel [128, 139]`, `w_bond [128, 2]` | `left`, `right`, both weights |

Backends of all four: `pytorch` (dense reference), `cuda` (H100 sm_90a; A100: the `cuda/sm80` builds of each family), `triton`
(any CUDA card). Coordinates, residue indices and ids are data (no gradient).

Requirements rows: token_pair stream, lengths 128 / 256 / 384, `float32 | master`, `eval | train`, 1 / 5 samples; `PairEmbedGeom` and
`PairEmbedCA` with and without `ln_eps=1e-05`.

Judging: [../measure.md](../measure.md); backends: [../modules.md](../modules.md); kernels: [../kernels.md](../kernels.md).

## Kernels

Each family is one shape-independent torch extension built on first use (`pair_embed_adaln_sm90`, `pair_embed_geom_sm90`,
`token_pair_init_sm90`; `gencodes("90a")`), launched on torch's current stream (CUDA-graph capturable), wrapped as opaque ops.

| requirements name | role | kernels (source) | what it does |
|---|---|---|---|
| `pair_embed_adaln_fwd_sm90` | forward | `pe_fwd` (`cuda/sm90/pair_embed_adaln.cu`) | one warp per (b, i) row, 32 pairs at a time: lane `l` bins pair (i, j0 + l) (separation, CA distances of `xt` / `sc`), packs the bins into a 32-bit word (saved for the backward); per pair the warp gathers 8 channels per lane of the three fp32 table rows, runs both LayerNorms with shuffles, applies `gamma` / `beta`, stores 16 B of bf16 |
| `pair_embed_adaln_bwd_sm90` | backward | `pe_bwd_rows`, `pe_bwd_tbl4` (or `pe_bwd_tbl`), `pe_reduce_rows`, `pe_reduce_tbl` | rows kernel recomputes the pair, writes `dx` (gradient of the table sum, bf16) and per-CTA partials of `dgamma` / `dbeta` / `dln_w` / `dln_b`; table kernel; fixed-order reduces. The batch runs in chunks so `dx` stays under 512 MB |
| `pair_embed_geom_fwd_sm90` / `_fwd_train_sm90` | forward_infer / forward_train | `pg_fwd` (`cuda/sm90/pair_embed_geom.cu`) | encoder pair: persistent CTAs of 16 warps, one warp per (b, i) row, 32 pairs at a time; bins (binary search over the limits in shared memory, distances / dihedrals / bond angles as torch computes them), sums up to ten bf16 table rows (table resident in shared memory) in fp32, optional LayerNorm, 16-byte stores; the train variant also writes a 16-byte code per pair |
| `pair_embed_ca_fwd_sm90` / `_ca_fwd_train_sm90` | forward_infer / forward_train | `pg_fwd_ca` | decoder pair, 32 warps per CTA, items of (row, 32-column chunk) |
| `pair_embed_geom_ln_bwd_sm90` | backward | `pg_ln_bwd<MODE, UL>` | LayerNorm backward with the pair recomputed from the codes and the table (never stored); UL pairs per warp step, loads pipelined two steps deep; zeros for dead pairs |
| `pair_embed_geom_bwd_sm90` | backward | `pg_bwd_mma4<MODE>`, `pg_reduce_tbl` | table gradient `dT = onehot^T dpair` on the tensor cores, then a fixed-order reduce to bf16 |
| `token_pair_init_prep_sm90` | forward_infer | `tpi_prep` (`cuda/sm90/token_pair_init.cu`) | forward-only glue in one launch: `ids [5, B L]` int32 from the five id tensors and `tbl = [W_rel^T ; W_bond^T]` fp32 |
| `token_pair_init_fwd_sm90` | forward | `tpi_fwd` | a CTA (8 warps) writes R rows of z for one batch element (R = 4 up to L 256, else 2); the class-row table arrives by one bulk (TMA) copy overlapping the packing of the R rows' bins; each warp loads `right[j]` once for the R rows |
| `token_pair_init_bwd_sm90` | backward | `tpi_bwd`, `tpi_reduce` | a CTA takes a quarter of the channels (32, one per lane) of 8 rows over every column; its 4 warps split the columns; per position a warp adds `dz` to its private shared-memory bin gradient (all lanes hit the same bin), the row sums and a per-row-block column partial; `tpi_reduce` adds the partials in fixed order |

`pair_embed_geom.cu` keeps two older table-gradient kernels (`pg_bwd_mma`, mma.sync; `pg_bwd_tbl`, shared-memory read-modify-write)
selectable by the `impl` argument; the loader passes `BWD_IMPL = 1` (`pg_bwd_mma4`). `pair_embed_adaln.cu` likewise keeps `pe_bwd_tbl`;
`pe_bwd_tbl4` serves when the table has at most 192 rows (the default 187), `pe_bwd_tbl` above.

Triton (`kernels/<family>/triton/main.py`): `pair_embed_geom_{bins,fwd_gather,dw,bwd_reduce,bwd_ln}_triton`,
`pair_embed_ca_bins_triton`, `pair_embed_adaln_{fwd,bwd,dw,bwd_reduce}_triton`, `token_pair_init_{fwd,bwd_dx,dw,dw_reduce}_triton`:
the same codes / gathers as the CUDA kernels, table gradients as one-hot products with fixed slices and fixed-order reduces. They are
launched by the `"triton"` backends; `requirements_kernel.csv` lists them without the `_triton` suffix.

## Design

**Table gradient on wgmma** (`pg_bwd_mma4`, `pe_bwd_tbl4`): `dT[row] = sum over the pairs using the row of dpair`. Two extra warps
scatter the bf16 one-hot operand (table rows x 64 pairs, 128-byte rows, 16-byte chunks XOR-swizzled by row) into shared memory from
the saved codes / bin words; the 64-pair x 64-channel gradient tile arrives by cp.async; one warpgroup per 64 table rows issues
`wgmma.m64n64k16` with both operands from shared memory (A K-major, B N-major, 128-byte swizzle). CTA (q, t) = channel slice q of the
fixed pair slice t (33 slices), tiles in order; the slice partials are added in slice order. No atomics: bitwise deterministic.

**Codes.** `PairEmbedGeom` / `PairEmbedCA` training forwards write one 16-byte code per pair (word 0: packed bins with a live bit,
word 1: the five angle bins, word 2: the chain product), which the LayerNorm backward and the table gradient replay; inference writes
none. `PairEmbedAdaLN` saves a 32-bit bin word per pair (masked bit, has-sc bit).

**Token-pair bins.** `pair_bins` computes AlphaFold 3's three relative-position classes and the same-entity flag; they are packed
8 bits each with the bond bit into a 32-bit word (`n_rel - 2 <= 255`).

**Forward-only glue.** `TokenPairInit` builds `ids` and the table in one launch (`tpi_prep`) when no weight needs a gradient and the
inputs are contiguous fp32 weights / int32-int64 ids; otherwise torch ops build them (differentiable). `PairEmbedGeom` / `PairEmbedCA`
build the bf16 table with one transpose-and-cast copy when untracked.

## Numerics

- Pair embeddings: binning in fp32 exactly as `torch.bucketize` would (count of limits strictly below the value); table-row sums and
  LayerNorm statistics in fp32; output bf16 (also for fp32 inputs, as every backend rounds at the same place). `PairEmbedGeom` /
  `PairEmbedCA` round the weight to bf16 first (a bf16-autocast Linear); their table gradient is accumulated in fp32 in fixed order and
  rounded to bf16 once. `PairEmbedAdaLN` keeps an fp32 table; gradients come back in each parameter's dtype.
- `TokenPairInit`: exact fp32 arithmetic (no TF32), sum order per element left + right, the class rows, the entity row, the bond row;
  bf16 `left` / `right` / table are cast to fp32 and the gradients returned in each input's dtype.
- Every backward is bitwise deterministic (per-CTA partials summed in fixed order, no float atomics).

## Launch configs

CUDA: fixed geometry (warps per CTA and grid from the sizes and the SM count); the CUDA kernels read no launch-config table.
Triton: `@configured` with defaults (e.g. `BLOCK_J = 16`, 4 warps for the row kernels; `BLOCK_P = 128` for the table-gradient
kernels); every Triton kernel has an entry in its family's `configs/h100_sm90.json`.

## Limits

- `PairEmbedAdaLN`: CUDA tensors; weight, LN affine, `gamma`, `beta`, `xt` fp32 or bf16; `xt [B, n, 3]`, `residue_idx` / `mask [B, n]`,
  `sc` like `xt` or None, `gamma` / `beta [B, 256]`; every `n_*` <= 255 (8-bit bins); non-empty. `"cuda"`: sm_90 (or sm_80) and
  `n_sep + n_xt + n_sc <= 220`.
- `PairEmbedGeom`: weight and `bb` fp32 or bf16, `bb [B, n, 4, 3]` (N, CA, C, CB), five `[B, n]` tensors; `"cuda"`: sm_90 (or sm_80).
- `PairEmbedCA`: weight and `ca` fp32 or bf16, `ca [B, n, 3]`, `mask` / `residue_idx [B, n]`; `"cuda"`: sm_90 (or sm_80).
- `TokenPairInit`: `left` / `right` / weights fp32 or bf16, `left` / `right [B, L, 128]`, five integer `[B, L]` ids, `bond [B, L, L]`,
  `n_rel - 2 <= 255`; `"cuda"`: sm_90 (or sm_80).

## History

Old page: `git show 65f37dc3c^:docs/gpus/h100/pair_ops/pair_ops.md`.
