# Page templates for docs/gpus/

<!--
Layout per GPU (H100 first; B200 / A100 move to it next):

  docs/gpus/<gpu>/
    <gpu>.md                  environment + module-level completion tables
    dispatch.md               module dispatch contracts on this GPU
    <module>/
      <module>.md             kernel-level tables, kernel-flow figures, measurements
      figures/
        <module>_<variant>.json          figure spec
        <module>_<variant>_<name>.svg    generated (python -m miniworld_engine.viz.kernel_flow <spec>)
        <module>_<variant>_<name>.png    rsvg-convert -z 2 -b white <svg> -o <png>; pages embed the PNG

Table rules (both pages):
- Columns = shapes the model shape registry declares (plus every width with a CUDA path).
  bf16-only modules (TriMul, TriAttn): (Length, Dimension); otherwise (Length, Dimension, dtype).
- Rows = implementation / 성능 확인 / cache build.
  implementation: CUDA on H100, or 미구현 where no CUDA path exists (kernel tables name the backend).
  성능 확인: maintainer only, ✓ / △ / ✗ (△ = the fastest measured, not yet complete).
  cache build: ✓ / ✗ (✓ also when nothing needs a cache).
- Kernel tables only for CUDA / Triton kernels; PyTorch and cuBLAS steps appear in the figures only.
- Measurements (module page): one table per variant and mode; columns PyTorch compiled /
  cuEquivariance / Anthropic / ours / × (ours vs the fastest other); rows = shapes and conditions.
-->

## `<gpu>/<gpu>.md`

```md
# <GPU> (<sm>)

Cluster / partition / QoS, nodes, driver and CUDA limits, CPU and GPU allocation commands
(with `--mem`), GPU-specific pitfalls.

## Completion status

<date>, <version>, torch <x> / triton <y>. <one line on 미구현 / dtype / cache build>.

### <Module> (<variant>) — [<module>/<module>.md](<module>/<module>.md)

#### Inference

| (Length, Dimension) | (<L>, <D>) | … |
|---|---|---|
| implementation | CUDA / 미구현 | |
| 성능 확인 | ✗ | |
| cache build | ✓ / ✗ | |

#### Training

(same)
```

## `<gpu>/<module>/<module>.md`

```md
# <Module> on <GPU> (<sm>)

Scope, dtype, figure legend, where dispatch lives.

## <Variant>

### Inference

#### <path> · <shapes it covers>

![<module> <variant> inference, <path>](figures/<module>_<variant>_inference_<path>.png)

##### <id> · <kernel name>

| (Length, Dimension) | (<L>, <D>) | … |
|---|---|---|
| implementation | CUDA / Triton | |
| 성능 확인 | ✗ | |
| cache build | ✓ / ✗ | |

### Training

(same; one figure per path, forward and backward stacked, each with its own HBM box)

## Measurements (<date>)

### <Variant> · Inference

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
```
