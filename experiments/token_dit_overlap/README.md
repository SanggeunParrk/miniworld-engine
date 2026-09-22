# Token DiT: what is left beyond v6, measured

`probe.py` re-assembles the v6 step of `../token_dit_fused` (frozen at cf70909b, imported, not edited) and times
variants of it. H100, 24 blocks, S = 5, bf16, CUDA graph, per block:

| variant | L768 | L384 | what it tells |
|---|---:|---:|---|
| v6 step as packaged | 170.9 us | 86.8 us | |
| re-assembled here | 174.3 us | 88.4 us | the baseline below |
| both `resgate_adaln_rows` passes skipped | 148.8 us | 75.1 us | the most that moving AdaLN into a GEMM can save: 25.5 / 13.3 us, 15 % (numbers wrong) |
| attention skipped | 124.0 us | 73.7 us | the core is 50 / 15 us of the block (numbers wrong) |
| samples 3 + 2 on two streams | 177.1 us | 93.9 us | slower |
| samples 2 + 2 + 1 on three streams | 180.4 us | 99.8 us | slower |
| samples 1 x 5 on five streams | 202.3 us | 105.0 us | slower |

**Stream-level overlap loses.** The samples are fully independent through the token DiT, so splitting them over
streams lets one group's attention (softmax, ALU-bound) run beside another group's GEMMs (tensor-bound). It does run
concurrently, but every GEMM gets smaller (M = 1920 or less instead of 3840) and loses more efficiency than the overlap
wins back. This also lowers confidence in the "persistent per-block kernel" bound (~100-110 us): the overlap that
bound assumes is not free at these sizes.

**AdaLN-into-GEMM is worth at most 15 %.** Skipping both residual + AdaLN passes is the ceiling. A real version keeps
the residual update (fp32 read-modify-write of x in the Wo / squeeze epilogue) and the AdaLN transform (in the next
GEMM's A-operand load), so it saves the xa and y round trips -- 23.6 MB per half-block at L768, ~16 us a block -- not
the whole 25.5 us. It needs custom CUDA GEMMs that match cuBLAS / quack: Triton lost 3.6x on exactly this (v1).
