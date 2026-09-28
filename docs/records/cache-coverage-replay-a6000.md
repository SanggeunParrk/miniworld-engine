# What the module matrix asks for and the shipped cache does not have

A record of one measurement, not a current reference. Re-measure with
`miniworld-engine dev audit --replay` on a card rather than citing these numbers.

Taken 2026-08-25 on an NVIDIA RTX A6000 (sm86), against the cache shipped at `d2638d4`, by
driving the module build matrix with capture off and collecting every lookup the cache did not
serve. The static check on the same cache reports `coverage: 91 OK, missing_pairs: 0` -- the two
answer different questions, which is the point of keeping this:

  * static: is every DECLARED `(op, dtype, shape bucket)` present? Declared work is what
    `op_units` enumerates, and the shipped cache was built from exactly that.
  * replay: does a run of the module matrix find what it asks for? The cache key also carries
    each kernel's constexprs (`H`, `ND`, `H2`, `SAVE_PREACT`, `ADD_RESIDUAL`, ...), which no
    declared work list enumerates.

**363 lookups missed, across 42 of 91 ops.** Every one falls back to a heuristic subset of the
grid inside the call, which is the cost the cache exists to remove.

This is evidence for the pending cache rebuild, and a work list for it: these are the keys the
rebuild has to reach, and reaching them means the sweep's unit list has to include the module
matrix, not only `op_units`.

## By op

| op | missed lookups |
|----|---------------:|
| `trimul_gemm_gate_mmajor_triton` | 39 |
| `layernorm_fwd_saveact_triton` | 30 |
| `layernorm_fwd_saveact_strided_triton` | 18 |
| `gated_projection_gate_dropres_triton` | 18 |
| `layernorm_fwd_rowscale_triton` | 16 |
| `layernorm_bwd_atomic_strided_triton` | 15 |
| `adaln_epilogue_saveact_triton` | 15 |
| `adaln_bwd_pre_dx_triton` | 15 |
| `transition_bwd_swiglu_recompute_triton` | 12 |
| `layernorm_fwd_strided_triton` | 12 |
| `layernorm_bwd_atomic_triton` | 12 |
| `augmented_attention_fwd_triton` | 12 |
| `adaln_epilogue_triton` | 12 |
| `cond_transition_swiglu_triton` | 10 |
| `trimul_bwd_gate_packed_triton` | 9 |
| `triangle_attention_fwd_triton` | 9 |
| `augmented_attention_bwd_split_triton` | 9 |
| `augmented_attention_bwd_pre_triton` | 9 |
| `transition_expand_swiglu_triton` | 8 |
| `gated_projection_gate_gemm_triton` | 6 |
| `gated_projection_bwd_gate_dropres_triton` | 6 |
| `gated_projection_bwd_dx_triton` | 6 |
| `cond_transition_squeeze_gate_triton` | 6 |
| `cond_transition_fwd_b2b_triton` | 6 |
| `cond_transition_expand_swiglu_triton` | 6 |
| `cond_transition_bwd_swiglu_flat_triton` | 5 |
| `trimul_outproj_gemm_gate_triton` | 3 |
| `trimul_gemm_gate_triton` | 3 |
| `triangle_attention_bwd_pre_triton` | 3 |
| `triangle_attention_bwd_dq_triton` | 3 |
| `triangle_attention_bwd_dkdv_triton` | 3 |
| `layernorm_linear_fwd_triton` | 3 |
| `layernorm_linear_fwd_fp32_triton` | 3 |
| `layernorm_bwd_split_triton` | 3 |
| `adaln_fwd_triton` | 3 |
| `adaln_bwd_dx_dlnw_triton` | 3 |
| `layernorm_stats_triton` | 2 |
| `gated_projection_gate_triton` | 2 |
| `gated_projection_gate_flat_triton` | 2 |
| `gated_projection_bwd_gate_flat_triton` | 2 |
| `cond_transition_squeeze_gate_saveact_triton` | 2 |
| `cond_transition_expand_swiglu_saveact_triton` | 2 |

## Every miss

```
adaln_bwd_dx_dlnw_triton                     float32|K2=256,NC=128,shape_key=1024
adaln_bwd_dx_dlnw_triton                     float32|K2=256,NC=128,shape_key=256
adaln_bwd_dx_dlnw_triton                     float32|K2=256,NC=128,shape_key=512
adaln_bwd_pre_dx_triton                      bfloat16+float32|N=384,shape_key=1024
adaln_bwd_pre_dx_triton                      bfloat16+float32|N=384,shape_key=256
adaln_bwd_pre_dx_triton                      bfloat16+float32|N=384,shape_key=512
adaln_bwd_pre_dx_triton                      bfloat16+float32|N=768,shape_key=1024
adaln_bwd_pre_dx_triton                      bfloat16+float32|N=768,shape_key=256
adaln_bwd_pre_dx_triton                      bfloat16+float32|N=768,shape_key=512
adaln_bwd_pre_dx_triton                      float32|N=128,shape_key=1024
adaln_bwd_pre_dx_triton                      float32|N=128,shape_key=256
adaln_bwd_pre_dx_triton                      float32|N=128,shape_key=512
adaln_bwd_pre_dx_triton                      float32|N=384,shape_key=1024
adaln_bwd_pre_dx_triton                      float32|N=384,shape_key=256
adaln_bwd_pre_dx_triton                      float32|N=384,shape_key=512
adaln_bwd_pre_dx_triton                      float32|N=768,shape_key=1024
adaln_bwd_pre_dx_triton                      float32|N=768,shape_key=256
adaln_bwd_pre_dx_triton                      float32|N=768,shape_key=512
adaln_epilogue_saveact_triton                bfloat16+float32|HAS_SB=1,N=384,shape_key=1024
adaln_epilogue_saveact_triton                bfloat16+float32|HAS_SB=1,N=384,shape_key=256
adaln_epilogue_saveact_triton                bfloat16+float32|HAS_SB=1,N=384,shape_key=512
adaln_epilogue_saveact_triton                bfloat16+float32|HAS_SB=1,N=768,shape_key=1024
adaln_epilogue_saveact_triton                bfloat16+float32|HAS_SB=1,N=768,shape_key=256
adaln_epilogue_saveact_triton                bfloat16+float32|HAS_SB=1,N=768,shape_key=512
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=128,shape_key=1024
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=128,shape_key=256
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=128,shape_key=512
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=384,shape_key=1024
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=384,shape_key=256
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=384,shape_key=512
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=768,shape_key=1024
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=768,shape_key=256
adaln_epilogue_saveact_triton                float32|HAS_SB=1,N=768,shape_key=512
adaln_epilogue_triton                        bfloat16|N=384,shape_key=1024
adaln_epilogue_triton                        bfloat16|N=384,shape_key=256
adaln_epilogue_triton                        bfloat16|N=384,shape_key=512
adaln_epilogue_triton                        bfloat16|N=768,shape_key=1024
adaln_epilogue_triton                        bfloat16|N=768,shape_key=256
adaln_epilogue_triton                        bfloat16|N=768,shape_key=512
adaln_epilogue_triton                        float32|N=384,shape_key=1024
adaln_epilogue_triton                        float32|N=384,shape_key=256
adaln_epilogue_triton                        float32|N=384,shape_key=512
adaln_epilogue_triton                        float32|N=768,shape_key=1024
adaln_epilogue_triton                        float32|N=768,shape_key=256
adaln_epilogue_triton                        float32|N=768,shape_key=512
adaln_fwd_triton                             float32|NC=128,NX=128,shape_key=1024
adaln_fwd_triton                             float32|NC=128,NX=128,shape_key=256
adaln_fwd_triton                             float32|NC=128,NX=128,shape_key=512
augmented_attention_bwd_pre_triton           bfloat16+float32|H=16,HEAD_DIM=24,shape_key=1024
augmented_attention_bwd_pre_triton           bfloat16+float32|H=16,HEAD_DIM=24,shape_key=256
augmented_attention_bwd_pre_triton           bfloat16+float32|H=16,HEAD_DIM=24,shape_key=512
augmented_attention_bwd_pre_triton           float32|H=16,HEAD_DIM=24,shape_key=1024
augmented_attention_bwd_pre_triton           float32|H=16,HEAD_DIM=24,shape_key=256
augmented_attention_bwd_pre_triton           float32|H=16,HEAD_DIM=24,shape_key=512
augmented_attention_bwd_pre_triton           float32|H=16,HEAD_DIM=48,shape_key=1024
augmented_attention_bwd_pre_triton           float32|H=16,HEAD_DIM=48,shape_key=256
augmented_attention_bwd_pre_triton           float32|H=16,HEAD_DIM=48,shape_key=512
augmented_attention_bwd_split_triton         bfloat16+float32|H=16,HEAD_DIM=24,shape_key=1024
augmented_attention_bwd_split_triton         bfloat16+float32|H=16,HEAD_DIM=24,shape_key=256
augmented_attention_bwd_split_triton         bfloat16+float32|H=16,HEAD_DIM=24,shape_key=512
augmented_attention_bwd_split_triton         float32|H=16,HEAD_DIM=24,shape_key=1024
augmented_attention_bwd_split_triton         float32|H=16,HEAD_DIM=24,shape_key=256
augmented_attention_bwd_split_triton         float32|H=16,HEAD_DIM=24,shape_key=512
augmented_attention_bwd_split_triton         float32|H=16,HEAD_DIM=48,shape_key=1024
augmented_attention_bwd_split_triton         float32|H=16,HEAD_DIM=48,shape_key=256
augmented_attention_bwd_split_triton         float32|H=16,HEAD_DIM=48,shape_key=512
augmented_attention_fwd_triton               bfloat16+float32|H=16,HEAD_DIM=24,shape_key=1024
augmented_attention_fwd_triton               bfloat16+float32|H=16,HEAD_DIM=24,shape_key=256
augmented_attention_fwd_triton               bfloat16+float32|H=16,HEAD_DIM=24,shape_key=512
augmented_attention_fwd_triton               bfloat16+float32|H=16,HEAD_DIM=48,shape_key=1024
augmented_attention_fwd_triton               bfloat16+float32|H=16,HEAD_DIM=48,shape_key=256
augmented_attention_fwd_triton               bfloat16+float32|H=16,HEAD_DIM=48,shape_key=512
augmented_attention_fwd_triton               float32|H=16,HEAD_DIM=24,shape_key=1024
augmented_attention_fwd_triton               float32|H=16,HEAD_DIM=24,shape_key=256
augmented_attention_fwd_triton               float32|H=16,HEAD_DIM=24,shape_key=512
augmented_attention_fwd_triton               float32|H=16,HEAD_DIM=48,shape_key=1024
augmented_attention_fwd_triton               float32|H=16,HEAD_DIM=48,shape_key=256
augmented_attention_fwd_triton               float32|H=16,HEAD_DIM=48,shape_key=512
cond_transition_bwd_swiglu_flat_triton       bfloat16|ND=256,shape_key=256
cond_transition_bwd_swiglu_flat_triton       bfloat16|ND=256,shape_key=512
cond_transition_bwd_swiglu_flat_triton       bfloat16|ND=768,shape_key=1024
cond_transition_bwd_swiglu_flat_triton       bfloat16|ND=768,shape_key=256
cond_transition_bwd_swiglu_flat_triton       bfloat16|ND=768,shape_key=512
cond_transition_expand_swiglu_saveact_triton float32|K=384,ND=768,shape_key=256
cond_transition_expand_swiglu_saveact_triton float32|K=384,ND=768,shape_key=512
cond_transition_expand_swiglu_triton         bfloat16|K=384,ND=768,shape_key=1024
cond_transition_expand_swiglu_triton         bfloat16|K=384,ND=768,shape_key=256
cond_transition_expand_swiglu_triton         bfloat16|K=384,ND=768,shape_key=512
cond_transition_expand_swiglu_triton         float32|K=384,ND=768,shape_key=1024
cond_transition_expand_swiglu_triton         float32|K=384,ND=768,shape_key=256
cond_transition_expand_swiglu_triton         float32|K=384,ND=768,shape_key=512
cond_transition_fwd_b2b_triton               bfloat16|DC=128,K=128,ND=256,shape_key=1024
cond_transition_fwd_b2b_triton               bfloat16|DC=128,K=128,ND=256,shape_key=256
cond_transition_fwd_b2b_triton               bfloat16|DC=128,K=128,ND=256,shape_key=512
cond_transition_fwd_b2b_triton               float32|DC=128,K=128,ND=256,shape_key=1024
cond_transition_fwd_b2b_triton               float32|DC=128,K=128,ND=256,shape_key=256
cond_transition_fwd_b2b_triton               float32|DC=128,K=128,ND=256,shape_key=512
cond_transition_squeeze_gate_saveact_triton  float32|D=384,DC=384,ND=768,shape_key=256
cond_transition_squeeze_gate_saveact_triton  float32|D=384,DC=384,ND=768,shape_key=512
cond_transition_squeeze_gate_triton          bfloat16|DC=384,ND=768,shape_key=1024
cond_transition_squeeze_gate_triton          bfloat16|DC=384,ND=768,shape_key=256
cond_transition_squeeze_gate_triton          bfloat16|DC=384,ND=768,shape_key=512
cond_transition_squeeze_gate_triton          float32|DC=384,ND=768,shape_key=1024
cond_transition_squeeze_gate_triton          float32|DC=384,ND=768,shape_key=256
cond_transition_squeeze_gate_triton          float32|DC=384,ND=768,shape_key=512
cond_transition_swiglu_triton                bfloat16|ND=256,shape_key=1024
cond_transition_swiglu_triton                bfloat16|ND=256,shape_key=256
cond_transition_swiglu_triton                bfloat16|ND=256,shape_key=512
cond_transition_swiglu_triton                bfloat16|ND=768,shape_key=1024
cond_transition_swiglu_triton                bfloat16|ND=768,shape_key=256
cond_transition_swiglu_triton                bfloat16|ND=768,shape_key=512
cond_transition_swiglu_triton                float32|ND=256,shape_key=1024
cond_transition_swiglu_triton                float32|ND=256,shape_key=512
cond_transition_swiglu_triton                float32|ND=768,shape_key=1024
cond_transition_swiglu_triton                float32|ND=768,shape_key=512
gated_projection_bwd_dx_triton               bfloat16|DH=256,N=128,shape_key=256
gated_projection_bwd_dx_triton               bfloat16|DH=256,N=128,shape_key=384
gated_projection_bwd_dx_triton               bfloat16|DH=256,N=128,shape_key=512
gated_projection_bwd_dx_triton               bfloat16|DH=512,N=256,shape_key=256
gated_projection_bwd_dx_triton               bfloat16|DH=512,N=256,shape_key=384
gated_projection_bwd_dx_triton               bfloat16|DH=512,N=256,shape_key=512
gated_projection_bwd_gate_dropres_triton     bfloat16|FROM_PREACT=0,N=256,USE_DROPOUT=0,shape_key=256
gated_projection_bwd_gate_dropres_triton     bfloat16|FROM_PREACT=0,N=256,USE_DROPOUT=0,shape_key=384
gated_projection_bwd_gate_dropres_triton     bfloat16|FROM_PREACT=0,N=256,USE_DROPOUT=0,shape_key=512
gated_projection_bwd_gate_dropres_triton     bfloat16|FROM_PREACT=0,N=512,USE_DROPOUT=0,shape_key=256
gated_projection_bwd_gate_dropres_triton     bfloat16|FROM_PREACT=0,N=512,USE_DROPOUT=0,shape_key=384
gated_projection_bwd_gate_dropres_triton     bfloat16|FROM_PREACT=0,N=512,USE_DROPOUT=0,shape_key=512
gated_projection_bwd_gate_flat_triton        float32|shape_key=1024
gated_projection_bwd_gate_flat_triton        float32|shape_key=512
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=128,SAVE_GATE=0,USE_DROPOUT=0,shape_key=256
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=128,SAVE_GATE=0,USE_DROPOUT=0,shape_key=384
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=128,SAVE_GATE=0,USE_DROPOUT=0,shape_key=512
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=128,SAVE_GATE=1,USE_DROPOUT=0,shape_key=256
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=128,SAVE_GATE=1,USE_DROPOUT=0,shape_key=384
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=128,SAVE_GATE=1,USE_DROPOUT=0,shape_key=512
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=256,SAVE_GATE=0,USE_DROPOUT=0,shape_key=256
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=256,SAVE_GATE=0,USE_DROPOUT=0,shape_key=384
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=256,SAVE_GATE=0,USE_DROPOUT=0,shape_key=512
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=256,SAVE_GATE=1,USE_DROPOUT=0,shape_key=256
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=256,SAVE_GATE=1,USE_DROPOUT=0,shape_key=384
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=256,SAVE_GATE=1,USE_DROPOUT=0,shape_key=512
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=512,SAVE_GATE=0,USE_DROPOUT=0,shape_key=256
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=512,SAVE_GATE=0,USE_DROPOUT=0,shape_key=384
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=512,SAVE_GATE=0,USE_DROPOUT=0,shape_key=512
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=512,SAVE_GATE=1,USE_DROPOUT=0,shape_key=256
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=512,SAVE_GATE=1,USE_DROPOUT=0,shape_key=384
gated_projection_gate_dropres_triton         bfloat16|ADD_RESIDUAL=1,N=512,SAVE_GATE=1,USE_DROPOUT=0,shape_key=512
gated_projection_gate_flat_triton            float32|shape_key=1024
gated_projection_gate_flat_triton            float32|shape_key=512
gated_projection_gate_gemm_triton            bfloat16|DH=256,N=128,shape_key=256
gated_projection_gate_gemm_triton            bfloat16|DH=256,N=128,shape_key=384
gated_projection_gate_gemm_triton            bfloat16|DH=256,N=128,shape_key=512
gated_projection_gate_gemm_triton            bfloat16|DH=512,N=256,shape_key=256
gated_projection_gate_gemm_triton            bfloat16|DH=512,N=256,shape_key=384
gated_projection_gate_gemm_triton            bfloat16|DH=512,N=256,shape_key=512
gated_projection_gate_triton                 bfloat16|R=256,shape_key=147456
gated_projection_gate_triton                 bfloat16|R=256,shape_key=65536
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=1024,shape_key=256
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=256,shape_key=256
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=384,shape_key=1024
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=384,shape_key=256
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=384,shape_key=512
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=512,shape_key=256
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=768,shape_key=1024
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=768,shape_key=256
layernorm_bwd_atomic_strided_triton          bfloat16+float32|N=768,shape_key=512
layernorm_bwd_atomic_strided_triton          float32|N=384,shape_key=1024
layernorm_bwd_atomic_strided_triton          float32|N=384,shape_key=256
layernorm_bwd_atomic_strided_triton          float32|N=384,shape_key=512
layernorm_bwd_atomic_strided_triton          float32|N=768,shape_key=1024
layernorm_bwd_atomic_strided_triton          float32|N=768,shape_key=256
layernorm_bwd_atomic_strided_triton          float32|N=768,shape_key=512
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=1048576
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=147456
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=262144
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=65536
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=1048576
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=147456
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=262144
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=65536
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=1048576
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=147456
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=262144
layernorm_bwd_atomic_triton                  bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=65536
layernorm_bwd_split_triton                   bfloat16+float32|N=1024,shape_key=256
layernorm_bwd_split_triton                   bfloat16+float32|N=256,shape_key=256
layernorm_bwd_split_triton                   bfloat16+float32|N=512,shape_key=256
layernorm_fwd_rowscale_triton                bfloat16+float32|D=128,shape_key=1048576
layernorm_fwd_rowscale_triton                bfloat16+float32|D=128,shape_key=147456
layernorm_fwd_rowscale_triton                bfloat16+float32|D=128,shape_key=262144
layernorm_fwd_rowscale_triton                bfloat16+float32|D=128,shape_key=65536
layernorm_fwd_rowscale_triton                bfloat16+float32|D=256,shape_key=1048576
layernorm_fwd_rowscale_triton                bfloat16+float32|D=256,shape_key=147456
layernorm_fwd_rowscale_triton                bfloat16+float32|D=256,shape_key=262144
layernorm_fwd_rowscale_triton                bfloat16+float32|D=256,shape_key=65536
layernorm_fwd_rowscale_triton                bfloat16+float32|D=384,shape_key=1048576
layernorm_fwd_rowscale_triton                bfloat16+float32|D=384,shape_key=147456
layernorm_fwd_rowscale_triton                bfloat16+float32|D=384,shape_key=262144
layernorm_fwd_rowscale_triton                bfloat16+float32|D=384,shape_key=65536
layernorm_fwd_rowscale_triton                bfloat16+float32|D=512,shape_key=1048576
layernorm_fwd_rowscale_triton                bfloat16+float32|D=512,shape_key=147456
layernorm_fwd_rowscale_triton                bfloat16+float32|D=512,shape_key=262144
layernorm_fwd_rowscale_triton                bfloat16+float32|D=512,shape_key=65536
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=1024,shape_key=256
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=256,shape_key=256
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=384,shape_key=1024
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=384,shape_key=256
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=384,shape_key=512
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=512,shape_key=256
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=768,shape_key=1024
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=768,shape_key=256
layernorm_fwd_saveact_strided_triton         bfloat16+float32|N=768,shape_key=512
layernorm_fwd_saveact_strided_triton         float32|N=128,shape_key=1024
layernorm_fwd_saveact_strided_triton         float32|N=128,shape_key=256
layernorm_fwd_saveact_strided_triton         float32|N=128,shape_key=512
layernorm_fwd_saveact_strided_triton         float32|N=384,shape_key=1024
layernorm_fwd_saveact_strided_triton         float32|N=384,shape_key=256
layernorm_fwd_saveact_strided_triton         float32|N=384,shape_key=512
layernorm_fwd_saveact_strided_triton         float32|N=768,shape_key=1024
layernorm_fwd_saveact_strided_triton         float32|N=768,shape_key=256
layernorm_fwd_saveact_strided_triton         float32|N=768,shape_key=512
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=1048576
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=147456
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=262144
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=256,shape_key=65536
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=1048576
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=147456
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=262144
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=384,shape_key=65536
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=1048576
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=147456
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=262144
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=512,shape_key=65536
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=64,shape_key=2048
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=0,N=64,shape_key=4096
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=128,shape_key=1048576
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=128,shape_key=147456
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=128,shape_key=262144
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=128,shape_key=65536
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=256,shape_key=1048576
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=256,shape_key=147456
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=256,shape_key=262144
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=256,shape_key=65536
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=384,shape_key=1048576
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=384,shape_key=147456
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=384,shape_key=262144
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=384,shape_key=65536
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=512,shape_key=1048576
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=512,shape_key=147456
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=512,shape_key=262144
layernorm_fwd_saveact_triton                 bfloat16+float32|HAS_ROWSCALE=1,N=512,shape_key=65536
layernorm_fwd_strided_triton                 bfloat16|HAS_W=1,N=384,shape_key=1024
layernorm_fwd_strided_triton                 bfloat16|HAS_W=1,N=384,shape_key=256
layernorm_fwd_strided_triton                 bfloat16|HAS_W=1,N=384,shape_key=512
layernorm_fwd_strided_triton                 bfloat16|HAS_W=1,N=768,shape_key=1024
layernorm_fwd_strided_triton                 bfloat16|HAS_W=1,N=768,shape_key=256
layernorm_fwd_strided_triton                 bfloat16|HAS_W=1,N=768,shape_key=512
layernorm_fwd_strided_triton                 float32|HAS_W=1,N=384,shape_key=1024
layernorm_fwd_strided_triton                 float32|HAS_W=1,N=384,shape_key=256
layernorm_fwd_strided_triton                 float32|HAS_W=1,N=384,shape_key=512
layernorm_fwd_strided_triton                 float32|HAS_W=1,N=768,shape_key=1024
layernorm_fwd_strided_triton                 float32|HAS_W=1,N=768,shape_key=256
layernorm_fwd_strided_triton                 float32|HAS_W=1,N=768,shape_key=512
layernorm_linear_fwd_fp32_triton             bfloat16+float32|N=256,NH=8,shape_key=147456
layernorm_linear_fwd_fp32_triton             bfloat16+float32|N=256,NH=8,shape_key=262144
layernorm_linear_fwd_fp32_triton             bfloat16+float32|N=256,NH=8,shape_key=65536
layernorm_linear_fwd_triton                  bfloat16+float32|HAS_BIAS=0,K=128,N=520,shape_key=1048576
layernorm_linear_fwd_triton                  bfloat16+float32|HAS_BIAS=0,K=128,N=520,shape_key=147456
layernorm_linear_fwd_triton                  bfloat16+float32|HAS_BIAS=0,K=128,N=520,shape_key=262144
layernorm_stats_triton                       bfloat16+float32|K=512,shape_key=147456
layernorm_stats_triton                       bfloat16+float32|K=512,shape_key=65536
transition_bwd_swiglu_recompute_triton       bfloat16|K=128,ND=512,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=1048576
transition_bwd_swiglu_recompute_triton       bfloat16|K=128,ND=512,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=147456
transition_bwd_swiglu_recompute_triton       bfloat16|K=128,ND=512,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=262144
transition_bwd_swiglu_recompute_triton       bfloat16|K=128,ND=512,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=65536
transition_bwd_swiglu_recompute_triton       bfloat16|K=256,ND=1024,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=1048576
transition_bwd_swiglu_recompute_triton       bfloat16|K=256,ND=1024,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=147456
transition_bwd_swiglu_recompute_triton       bfloat16|K=256,ND=1024,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=262144
transition_bwd_swiglu_recompute_triton       bfloat16|K=256,ND=1024,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=65536
transition_bwd_swiglu_recompute_triton       bfloat16|K=384,ND=1536,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=1048576
transition_bwd_swiglu_recompute_triton       bfloat16|K=384,ND=1536,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=147456
transition_bwd_swiglu_recompute_triton       bfloat16|K=384,ND=1536,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=262144
transition_bwd_swiglu_recompute_triton       bfloat16|K=384,ND=1536,NORMALIZE=0,STACK_DAB=1,STORE_H=1,shape_key=65536
transition_expand_swiglu_triton              bfloat16|N=256,n=4,shape_key=1048576
transition_expand_swiglu_triton              bfloat16|N=256,n=4,shape_key=147456
transition_expand_swiglu_triton              bfloat16|N=256,n=4,shape_key=262144
transition_expand_swiglu_triton              bfloat16|N=256,n=4,shape_key=65536
transition_expand_swiglu_triton              bfloat16|N=384,n=4,shape_key=1048576
transition_expand_swiglu_triton              bfloat16|N=384,n=4,shape_key=147456
transition_expand_swiglu_triton              bfloat16|N=384,n=4,shape_key=262144
transition_expand_swiglu_triton              bfloat16|N=384,n=4,shape_key=65536
triangle_attention_bwd_dkdv_triton           bfloat16+float32|HEAD_DIM=16,shape_key=256
triangle_attention_bwd_dkdv_triton           bfloat16+float32|HEAD_DIM=16,shape_key=384
triangle_attention_bwd_dkdv_triton           bfloat16+float32|HEAD_DIM=16,shape_key=512
triangle_attention_bwd_dq_triton             bfloat16+float32|HEAD_DIM=16,shape_key=256
triangle_attention_bwd_dq_triton             bfloat16+float32|HEAD_DIM=16,shape_key=384
triangle_attention_bwd_dq_triton             bfloat16+float32|HEAD_DIM=16,shape_key=512
triangle_attention_bwd_pre_triton            bfloat16+float32|HEAD_DIM=16,shape_key=256
triangle_attention_bwd_pre_triton            bfloat16+float32|HEAD_DIM=16,shape_key=384
triangle_attention_bwd_pre_triton            bfloat16+float32|HEAD_DIM=16,shape_key=512
triangle_attention_fwd_triton                bfloat16+float32|H=16,HEAD_DIM=16,shape_key=256
triangle_attention_fwd_triton                bfloat16+float32|H=16,HEAD_DIM=16,shape_key=384
triangle_attention_fwd_triton                bfloat16+float32|H=16,HEAD_DIM=16,shape_key=512
triangle_attention_fwd_triton                bfloat16+float32|H=4,HEAD_DIM=64,shape_key=256
triangle_attention_fwd_triton                bfloat16+float32|H=4,HEAD_DIM=64,shape_key=384
triangle_attention_fwd_triton                bfloat16+float32|H=4,HEAD_DIM=64,shape_key=512
triangle_attention_fwd_triton                bfloat16+float32|H=8,HEAD_DIM=16,shape_key=256
triangle_attention_fwd_triton                bfloat16+float32|H=8,HEAD_DIM=16,shape_key=384
triangle_attention_fwd_triton                bfloat16+float32|H=8,HEAD_DIM=16,shape_key=512
trimul_bwd_gate_packed_triton                bfloat16|D=1024,shape_key=256
trimul_bwd_gate_packed_triton                bfloat16|D=1024,shape_key=384
trimul_bwd_gate_packed_triton                bfloat16|D=1024,shape_key=512
trimul_bwd_gate_packed_triton                bfloat16|D=256,shape_key=256
trimul_bwd_gate_packed_triton                bfloat16|D=256,shape_key=384
trimul_bwd_gate_packed_triton                bfloat16|D=256,shape_key=512
trimul_bwd_gate_packed_triton                bfloat16|D=512,shape_key=256
trimul_bwd_gate_packed_triton                bfloat16|D=512,shape_key=384
trimul_bwd_gate_packed_triton                bfloat16|D=512,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=1024,K=512,SAVE_PREACT=0,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=1024,K=512,SAVE_PREACT=0,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=1024,K=512,SAVE_PREACT=0,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=1024,K=512,SAVE_PREACT=1,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=1024,K=512,SAVE_PREACT=1,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=1024,K=512,SAVE_PREACT=1,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=128,K=128,SAVE_PREACT=0,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=128,K=128,SAVE_PREACT=0,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=128,K=128,SAVE_PREACT=0,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=128,K=128,SAVE_PREACT=1,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=128,K=128,SAVE_PREACT=1,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=128,K=128,SAVE_PREACT=1,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=128,SAVE_PREACT=0,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=128,SAVE_PREACT=0,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=128,SAVE_PREACT=0,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=256,SAVE_PREACT=0,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=256,SAVE_PREACT=0,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=256,SAVE_PREACT=0,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=256,SAVE_PREACT=1,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=256,SAVE_PREACT=1,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=256,K=256,SAVE_PREACT=1,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=384,K=384,SAVE_PREACT=0,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=384,K=384,SAVE_PREACT=0,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=384,K=384,SAVE_PREACT=0,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=384,K=384,SAVE_PREACT=1,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=384,K=384,SAVE_PREACT=1,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=384,K=384,SAVE_PREACT=1,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=256,SAVE_PREACT=0,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=256,SAVE_PREACT=0,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=256,SAVE_PREACT=0,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=256,SAVE_PREACT=1,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=256,SAVE_PREACT=1,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=256,SAVE_PREACT=1,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=512,SAVE_PREACT=0,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=512,SAVE_PREACT=0,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=512,SAVE_PREACT=0,shape_key=512
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=512,SAVE_PREACT=1,shape_key=256
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=512,SAVE_PREACT=1,shape_key=384
trimul_gemm_gate_mmajor_triton               bfloat16|H2=512,K=512,SAVE_PREACT=1,shape_key=512
trimul_gemm_gate_triton                      bfloat16|N=256,shape_key=256
trimul_gemm_gate_triton                      bfloat16|N=256,shape_key=384
trimul_gemm_gate_triton                      bfloat16|N=256,shape_key=512
trimul_outproj_gemm_gate_triton              bfloat16|N=256,shape_key=256
trimul_outproj_gemm_gate_triton              bfloat16|N=256,shape_key=384
trimul_outproj_gemm_gate_triton              bfloat16|N=256,shape_key=512
```
