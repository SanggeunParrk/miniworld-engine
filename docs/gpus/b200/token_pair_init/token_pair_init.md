# Token pair initialisation on B200 (sm100)

Kernel-level status of `kernels/token_pair_init`; the module-level summary is in [b200.md](../b200.md). The op is **not in the
completion tables yet**: it has no row in the shape registry and no bench target, so there is no **성능 확인** to give (the
maintainer's call). **cache build ✓**: nothing on this path autotunes (fixed launch shapes, a cubin built on first use into
`MINIWORLD_ENGINE_JIT_ROOT`).

The op is the pair stream of MiniWorld's input feature embedder (`miniworld.modules.input_embedder`):

    z[b, i, j, :] = left[b, i, :] + right[b, j, :] + W_rel . onehot(rel(i, j)) + W_bond . onehot(bond[b, i, j])

`rel(i, j)` is AlphaFold 3's relative-position encoding: the clipped residue offset (66 classes, "other chain" the last), the clipped
token offset (same chain and residue only), the clipped chain-symmetry offset (6 classes) and a same-entity flag: 139 columns for
`r_max` 32 / `s_max` 2. The model ran it as a 139-wide fp32 one-hot (84 MB at 384 tokens), a Linear over it, the outer sum, and a
second one-hot Linear for the bond. A one-hot times a matrix is a row gather, so the kernels never build it, and they sum in exact
fp32 (the reference's TF32 Linears round the weights).

## Scope and dispatch

fp32, d_pair 128, any B and any L (ragged lengths included), five id tensors [B, L] of any integer dtype, `bond` [B, L, L] (bool,
integer or float: any nonzero is a bond), bins that fit 8 bits (`r_max` 32 / `s_max` 2 use 139 - 2). `token_pair_init.refusal(...)`
returns `None` when the op serves a call and otherwise the reason; it reads metadata only, so it traces under `torch.compile`. A
refused call runs the caller's unfused ops: MiniWorld's `InputEmbedder` asks it first, and `MINIWORLD_FUSED_PAIR_INIT=0` turns the
op off there.

Plain CUDA-core kernels, no tensor cores: the op is bandwidth bound. The cubin is built on first use by the newest nvcc that knows
sm_100a and launched through the driver on torch's current stream, so it is CUDA-graph capturable (tested). The two ops are registered as
`token_pair_init_fwd` / `token_pair_init_bwd` and covered by `tests/compile/test_compile_wrap_coverage.py`.

## Forward and backward

### F1 · forward `token_pair_init_fwd`

`kernels/token_pair_init/cuda/sm100/token_pair_init.cu`. One pass that writes `z`, the only large tensor. Each CTA first turns the
ids of its rows into one packed word per pair in shared memory with all threads (`bin1 | bin2 << 8 | bin3 << 16 | same_entity << 24 |
bond << 25`), then gathers the table rows and adds `left[i] + right[j]`. One row of `z` per CTA up to L 384, two from L 512, where the
72 KB table load is amortised (`fwd_rows`).

### B1 · backward `token_pair_init_bwd`

One pass over `dz`: `dleft` (row sums), `dright` (column sums, global vector atomics) and the class-bin gradients `dW[bin][:]`, which
are accumulated per run in registers (a bin changes rarely along j: the clipped offsets are constant away from the diagonal), added per
CTA in shared memory and then to global once. One row per CTA up to L 256, two above (`bwd_rows`). The atomics mean the last bits of
`dright` / `dW` can differ from run to run; ids that jump around (a shuffled residue index) make every position a run, which is correct
but slow.

## Measurements (2026-10-02)

One B200 (sm_100a, 1000 W cap, shared lab server), torch 2.13.0+cu129, triton 3.7.1, branch `feat/b200-embedder-fa4-pair-init` on engine main
`5d8bb030`; fp32, B = 1, CUDA-graph timing (50 replays after warm-up). "ours" is the op as a caller sees it (the id packing around the
kernels included); the reference is `token_pair_init_reference`, the dense fp32 one-hot Linear the model ran.

| L | forward ours (ms) | forward reference (ms) | × | forward + backward ours (ms) | forward + backward reference (ms) | × |
|---|---|---|---|---|---|---|
| 128 | 0.025 | 0.139 | 5.6 | 0.053 | 0.215 | 4.0 |
| 256 | 0.035 | 0.298 | 8.5 | 0.076 | 0.448 | 5.9 |
| 384 | 0.045 | 0.549 | 12.1 | 0.151 | 0.809 | 5.4 |
| 768 | 0.109 | 1.843 | 16.9 | 0.443 | 2.724 | 6.2 |

In MiniWorld's input feature embedder (three SWA atom blocks, 4096 atoms / 384 tokens, forward + backward, CUDA graph, one B200; the
embedder before this branch: the one-hot atom-to-token einsums, the legacy flash op, the unfused pair stream), each step of the branch:

| step | graph wall (ms) | launches |
|---|---|---|
| before | 1.84 | 307 |
| + FlashAttention-4 forward saves its lse (`swa_flash_saves_lse`) | 1.41 | 204 |
| + atom-to-token mean as a scatter-add (MiniWorld) | 1.38 | 200 |
| + token-pair initialisation (this op) | 1.08 | 190 |

The scatter-add step is in MiniWorld (`input_feature_embedder_esmfold2_style.py`) and the last step needs MiniWorld's `input_embedder.py` to
call this op (it asks `refusal` first and falls back).

## Tests

`tests/integrations/test_token_pair_init_gpu.py` compares the output and every gradient with the dense fp32 reference (ragged lengths,
B > 1, bond dtypes, a refusal for each unsupported call) and records the op into a CUDA graph.
