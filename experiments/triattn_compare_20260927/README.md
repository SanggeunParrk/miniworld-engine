# Matched TriangleAttention backend comparison

Scope: full training module, B1/D128/H4/head32, L384/768, starting/ending,
BF16 activation/linear weights, FP32 LayerNorm, p_drop=.25, 10% token mask,
identical initialized state/input/dy. All arms use torch.compile(fullgraph=True)
and manual CUDA Graph capture. The current engine, native-fusions-disabled
Triton attention path, PyTorch einsum/softmax module reference, and cuEquivariance
attention module adapter are measured on the same GPU in rotating/reversed
order, 90 samples per arm. Kernel-only times are not compared to module times.

Triton disables the five native backward/front fusions and sets
MINIWORLD_TRIATTN_TRAINING_FWD=0 before capture. Existing normalization/gate
helpers remain part of that composed module. The label does not claim that
every leaf operation is Triton. Default cache/tuning policy is retained
(autotune_miss_cap=24); this is not a freshly exhaustive Triton tuning result.

Numerical comparison uses identical nonzero weights and dropout=0, checking
output and all gradients versus current engine at relative-L2 <.03. Training
measurements then use live dropout RNG with output-change and finite checks.
This validates comparability, not a new strict kernel promotion qualification.
The reported PyTorch reference is the repository's explicit einsum/softmax
implementation, not an assertion of the best possible SDPA/FlashAttention model.

Each captured arm has an actual CUDA activity trace. Engine and comparator
execution must be distinguished using those kernels, not module backend labels.
Capture/compile failures are explicit errors; an error is never a timing win.
Source/script SHA256s and all samples are retained in the result JSON.
