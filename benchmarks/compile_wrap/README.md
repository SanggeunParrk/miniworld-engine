# `compile_wrap` measurements

The four scripts that produced every number quoted in the `compile_wrap="custom_op"` change --
in the commit messages, in `kernels/_compile.py`, in `settings.py`, and in
`docs/benchmarking-cautions.md`.

They live here rather than in a scratch directory on purpose. This repo already learned that
lesson once for the autotune sweeps: a claim whose measurement script is not in the tree cannot
be re-run, re-checked, or corrected by anyone but the person who ran it. Every table below is
reproducible from a checkout.

| script | question |
|---|---|
| `graph_structure.py` | how many graphs / breaks / captured nodes each module traces to, per mode, plus a bit-for-bit gradient parity check between the modes |
| `block_regimes.py` | Pairformer ×4 across the four regimes (eager, +manual CUDA graph, `torch.compile`, both) -- the compile-vs-cudagraph question |
| `module_regimes.py` | the same A/B per module, which is where the gain turns out to track how much unfused work surrounds a kernel |
| `reduce_overhead.py` | whether inductor's cudagraph-trees engage, read off inductor's own counters rather than off the clock |

Each takes `--wrap {disable,custom_op}` and runs one mode per process, because
`settings.compile_wrap` is consumed when the kernel modules import -- a single interpreter can
only ever hold one of the two.

```
PYTHONPATH=src python benchmarks/compile_wrap/block_regimes.py --wrap custom_op
```

## Two traps these scripts are shaped around

**Measure COLD.** `graph_structure.py` explains the graph *before* its parity call, not after.
The other order is how the zero-break result was first reported wrongly: the parity call warms a
dispatch cache whose miss path graph-breaks, so a pairformer block traced to 1 graph warm and 6
cold. Real training's first compiled step is a cold trace.

**A skip is silent.** `mode="reduce-overhead"` that declines to capture still runs and still
returns correct answers -- only slower. `reduce_overhead.py` therefore reports
`torch._dynamo.utils.counters["inductor"]`, so "captured" is read from inductor rather than
inferred from a timing that could differ for any reason.
