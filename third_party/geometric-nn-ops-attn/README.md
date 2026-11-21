# geometric-nn-ops (AMD / ROCm) - triangle attention baseline of E5

Vendor-neutral Triton kernels for triangle ops (forward + backward), AMD authored, Apache-2.0, https://github.com/ROCm/geometric-nn-ops (pinned in UPSTREAM.json).
The ROCm documentation targets MI300/MI350, but the kernels are plain Triton and run on NVIDIA (H100, A100, RTX 5090 verified here, see E5 records).
The triangle-attention kernel is the OpenFold3 Triton Evoformer-attention kernel in a cuEquivariance-style signature (`triangle_attention(q, k, v, bias, mask, scale)`).

| op | covered by E5 | notes |
|---|---|---|
| `triangle_attention` | yes (`gnops`): TriangleAttention classes, AttentionPairBias | Triton fwd + bwd; bf16 / fp16 / fp32; bool key mask; pair bias `[B,1,H,N,N]`; scale inside |
| `attention_pair_bias` | no | pure torch (no kernel yet) |
| `triangle_multiplicative_update` | not here | belongs to the triangle-multiplication candidates (branch third-party-trimul) |

Directory is named `geometric-nn-ops-attn` so it does not collide with a trimul entry of the other branch. On a card the adapter reads `$PAPER_EXT_ATTN/geometric-nn-ops`
(`DEST=$PAPER_EXT_ATTN/geometric-nn-ops ./fetch.sh`). `upstream/` is gitignored.
