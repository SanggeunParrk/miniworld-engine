# MPNN native BF16 precision contract

MPNN now supports the same native low-precision convention used by the MiniWorld
module benchmarks: ordinary model parameters are BF16, while LayerNorm affine
parameters remain FP32. No autocast context or FP32 master model is required.

```python
model = ProteinMPNN(config).cuda().bfloat16().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Keep backbone coordinates FP32 and sequence/neighbor indices integer.
# Mask inputs may remain FP32; multiplication boundaries preserve feature dtype.
def loss_fn():
    logits = model(*inputs, checkpoint_layers=False)
    return torch.nn.functional.cross_entropy(logits.float(), sequence)

compiled_loss = torch.compile(
    loss_fn, fullgraph=True, options={"triton.cudagraphs": False}
)
optimizer.zero_grad(set_to_none=True)
loss = compiled_loss()
loss.backward()
optimizer.step()
```

For D128, three encoder and three decoder layers, 1,656,389 ordinary parameters
are BF16 and 4,096 LayerNorm parameters are FP32. Gradients follow parameter
dtypes. Standard AdamW creates its moments in the corresponding parameter dtype;
there is no separate FP32 master parameter copy in this setup.

The precision exceptions are explicit: LayerNorm affine and statistics use FP32,
geometry/distance/RBF construction uses FP32 coordinates, and cross-entropy is
computed on FP32 logits. GEMM/reduction kernels can accumulate internally in FP32;
that does not imply FP32 parameter storage. BF16 activations return from LayerNorm
and enter ordinary projections without autocast. FP64 reference tests remain FP64.

## Fixed boundaries

- MPNN LayerNorm preserves FP32 affine through `.to(torch.bfloat16)` and repeated
  `.bfloat16()` calls, without first rounding an existing gamma through BF16.
- Edge-tail dispatch admits native BF16 projection parameters with FP32 norm
  affine for both memory and compute backends.
- FP32 geometric features and custom message reductions are cast at native BF16
  projection boundaries. The FP32-parameter autocast path retains its own casting.
- Residue/decoding masks no longer promote BF16 residual streams or dense decoder
  inputs back to FP32 before a BF16 linear operation.

## Benchmark interpretation

The historical B8/L8192 values in
[the A5000/A6000 comparison](mpnn-a5000-a6000-comparison.md) used **BF16 autocast
with FP32 parameters**. They are not native BF16 timings or memory measurements.
The historical record remains intact; do not relabel those values or derive a
native BF16 PyTorch speedup from them.

Any new full-model comparison must apply this precision contract to both model
implementations: compile ON, training CUDA Graphs OFF, dropout 0.25, no checkpoint
API calls, same inputs and loss, and the same optimizer/timing scope. Inference
comparisons use compiled forward plus CUDA Graph replay for both implementations.

The existing kernel-family runner now defaults to this convention too:

```bash
python -m benchmarks.runners.mpnn_compare --precision bf16 --out native.json
# Explicit historical control:
python -m benchmarks.runners.mpnn_compare --precision bf16-mixed --out autocast.json
```

Its rows record actual input/parameter dtypes and the autocast mode. The BF16
relative-position comparison uses the existing BF16-family tolerance (5%); the
historical FP32-parameter relative-position control retains its 1e-4 threshold.
This runner measures individual operations without AdamW, not whole-model speedup.

## Native-weight memory-kernel fix

On A6000, native BF16 edge-tail replay required 104,448 bytes of shared memory
against a 101,376-byte limit, including with the ordinary 24-candidate fallback.
The three full 128x128 BF16 weights remained live together across the row-tile loop.
Replay and dX now load each native BF16 weight at its use inside that loop, with
loop-invariant code motion disabled for that branch. The FP32-weight/autocast
branch keeps its existing weight preloads. The arithmetic, outputs, autograd ABI,
and public tuning grid are unchanged. Source identity invalidates affected tuned
entries; this work does not build an exhaustive native BF16 cache grid.

A minimal-tile A6000 check (job 1682244) passed fullgraph full-model output/gradient
comparison with the pure PyTorch native BF16 path, plus the native BF16 compressed
LayerNorm check. CPU regression after the final changes: **61 passed, 32 skipped**
(GPU-marked cases deselected). Full-model accuracy tests use B2/L64, K48, D128,
3+3 layers, masked residues, dropout zero for numerical comparison, no autocast,
and FP32 loss. A separate A6000 smoke run also completed forward/backward with
dropout 0.25 in eager and fullgraph modes with every parameter gradient finite.
These are correctness checks, not B8/L8192 performance or convergence results.

Final A6000 regression, job **1682247**: **86 passed** using the ordinary
24-candidate fallback, without narrowing either backward kernel's grid. This
includes all four compiled native BF16 full-model combinations (memory/compute
edge-tail and nodes, retained/regenerated RBF), FP64 and historical autocast model
parity, edge-tail forward/backward/dropout tests, feature-memory tests, native
benchmark tensor-dtype checks, observed compiled training with graphs disabled,
and observed compiled inference with manual graph replay. Raw log:
`.scratch/mpnn-native-bf16/tests-1682247.log`.
Ruff and ty passed for the changed implementation/runner; `git diff --check` passed.
