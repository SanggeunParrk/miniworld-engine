# ProteinMPNN Triton kernels on H100

## Scope

`ProteinMPNN` (`modules/mpnn/`): backbone KNN features, a node / edge message-passing encoder, a teacher-forced decoder and the
output projection. Backends: `pytorch`, `triton` and `cuda`. `"cuda"` is the A100 (sm_80) kernels only (`kernels/mpnn_*/cuda/sm80`,
`kernels/integrations/mpnn_msg_sm80.py`, `mpnn_edge_sm80.py`): on H100 `_supports_cuda` refuses (`mpnn_sm80_unsupported`) and the
Triton kernels below serve the model. There is no `"anthropic"` backend (the release kit binds no kernel for the model's math). `"triton"` (or `"cuda"`) sets every child that has that
backend; projections, transitions, the other norms and the output projection stay PyTorch statements.

Requirements row: `ProteinMPNN`, protein_node stream, lengths 1024-8192, node / edge / hidden width 128, `k_neighbors=48`, encoder
and decoder depth 3, `dropout=0.1`, dtype `master` only (fp32 parameters, bf16 activations under autocast), `eval | train`.

This note is written from the code. Judging: [../measure.md](../measure.md); backends: [../modules.md](../modules.md); kernels:
[../kernels.md](../kernels.md).

## Which child runs which family

| child (`"triton"`) | family | eval | train |
|---|---|---|---|
| `EncoderLayer` (`fused=True`, `ProteinMPNNConfig.fused_encoder`, default) | `mpnn_node_message`, `mpnn_edge_tail` | fused node message and edge tail, nothing saved | save-activation kernels, or recompute kernels with `replay_in_backward` |
| `NodeMessageUpdate` (decoder; encoder when not fused) | `mpnn_message` | `mpnn_message_fwd_gemm_gelu_reduce` (one-kernel no-save fusion) | `fwd_gemm` + `fwd_gelu_reduce` keeping the projection, or the replay variant |
| `EdgeMessageUpdate` (encoder when not fused) | `mpnn_edge_mlp` | `fwd_gemm_b2b` (back-to-back GEMM, saves nothing) | `fwd_gemm` saving the projection, or `b2b` with recompute |
| `EdgeDropout` (encoder when not fused) | `mpnn_edge_dropout` | identity (reference statement) | ATen dropout forward, mask packed to 1 bit per element, Triton backward |
| `EdgeLayerNorm` (encoder when not fused) | `mpnn_edge_layernorm` (backward on the `layernorm` family's kernel) | reference statement | native forward, a bf16 copy of the input saved for the backward |
| `RelativePositionEmbedding` | `mpnn_relative_position` | table lookup | backward: bucket reduction of the table / bias gradient instead of ATen's sorted `embedding_dense_backward` |

## Kernels (family = directory)

| family | kernels | what they do |
|---|---|---|
| `mpnn_node_message` | `fwd_gemm`, `bwd_recompute`, `bwd_dx` | the edge block of the packed encoder projection with the two node-side blocks added and the neighbour gather folded in, GELU, hidden projection, GELU, masked neighbour sum / K; one neighbour group (K <= 128) held in registers |
| `mpnn_edge_tail` | `fwd_gemm_{gather,b2b,layernorm}_saveact`, `fwd_gemm{,_layernorm}_recompute`, `bwd_{layernorm_saveact,dx_saveact,dx_gather_saveact,recompute,dx_recompute}` | the encoder edge update `layer_norm(edge + dropout(W3 gelu(W2 gelu(query + edge W1e^T + neighbor[index]))))` with every link in registers; the recompute variants replay the chain in backward from the input, so none of the three GEMM activations is stored |
| `mpnn_message` | `fwd_gemm`, `fwd_gelu_reduce`, `fwd_gemm_gelu_reduce` (eval), `bwd_dx`, `bwd_reduce_dbias`, `bwd_zero_dbias` | `sum_k mask_k gelu(W_h gelu(pre_k) + b_h) / K` for 48 neighbours, and its backward |
| `mpnn_edge_mlp` | `fwd_gemm_b2b`, `fwd_gemm` | `W_o gelu(W_h gelu(pre) + b_h) + b_o` per edge |
| `mpnn_edge_dropout` | `fwd_packed`, `bwd_packed` | 1-bit dropout mask packing and the masked backward |
| `mpnn_relative_position` | `bwd_reduce` | bucket reduction (a whole table privatised per program; at most 512 rows x 128 channels) |

All names carry the `mpnn_<family>_` prefix (e.g. `mpnn_edge_tail_fwd_gemm_gather_saveact`). `requirements_kernel.csv` lists only
the Triton kernels the requirements row reaches (fused encoder, saved activations): `mpnn_node_message_fwd_gemm`, the six
`mpnn_edge_tail_*_saveact`, the six `mpnn_message_*` and `mpnn_relative_position_bwd_reduce`; the recompute variants, the node-message
backward kernels, `mpnn_edge_mlp_*` and `mpnn_edge_dropout_*` are in the code but in no row. The CSV's `*_sm80` rows are the A100
kernels.

## Numerics

- bf16 activations, fp32 accumulation; projection parameters bf16 (fp32 masters cast under bf16 autocast).
- The fused edge tail returns its LayerNorm output in bf16 (autocast's `layer_norm` would return fp32; the next consumer is an
  autocast `F.linear` that casts it back) and keeps the residual in bf16.
- The fused edge tail draws dropout from Triton's Philox stream keyed by a seed tensor (the backward reproduces the mask), not from
  `aten::native_dropout`'s stream; the module's `dropout` / `norm` contribute only their probability and parameters.
- The node message divides by the graph width K (`neighbor_scale`), not by the count of valid edges, and scales the output bias by
  the valid-edge fraction (ProteinMPNN's rule).

## Launch configs

H100 tables (`configs/h100_sm90.json`, top 4 per dtype / row bucket, rank 1 by default) exist for the CSV's Triton kernels:
`mpnn_message` (its five forward / backward kernels), `mpnn_edge_tail` (the six `*_saveact`), `mpnn_node_message_fwd_gemm`,
`mpnn_relative_position_bwd_reduce`. Every other `@configured` MPNN kernel has no H100 entry and launches its default.

## Limits

- `ProteinMPNN._supports_triton`: CUDA tensors; node / edge / hidden width 128; `k_neighbors == 48`.
- Activations (`mpnn_triton_unsupported`): non-empty bf16 CUDA `[..., 128]` within 32-bit offsets; projection parameters bf16 (or
  fp32 under bf16 autocast), contiguous, CUDA.
- `NodeMessageUpdate`: preactivation `[..., 48, 128]`, edge mask of the leading shape without gradient, hidden projection `[128, 128]`
  with bias.
- `EdgeMessageUpdate`: both projections `[128, 128]` with bias.
- Fused `EncoderLayer`: bf16 edge states `[B, L, K, 128]` (K <= 128, contiguous), node states `[B, L, 128]` bf16 (fp32 only under
  bf16 autocast), int64 neighbour indices of the edge shape, a float edge mask.
- `EdgeDropout`: training with gradient, `0 < p < 1`, not in place, not under deterministic algorithms, contiguous fp32 / bf16.
- `RelativePositionEmbedding`: table <= 512 rows and <= 128 channels, fp32 or bf16, bias of the same dtype.

## History

No dedicated design page existed before this note; the old family list was deleted (see git history).
