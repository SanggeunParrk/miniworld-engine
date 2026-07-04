# Benchmarking cautions (graph-break kernels: cute / triton / cuBLAS)

Hard-won lessons from benchmarking the trimul kernels. These ops are
`@torch.compiler.disable` islands made of many small cute/triton/cuBLAS launches,
so they are **host/launch-bound at small problem sizes**. That makes them very
easy to mis-measure. Read this before trusting any trimul number.

See also `docs/benchmarks.md` for the harness/CSV/plot convention.

---

## 1. The measurement *regime* is the #1 variable — pick it deliberately

The same kernel gives wildly different numbers in three regimes:

| regime | what it measures | when it lies |
|---|---|---|
| **eager** | raw op + Python host dispatch per launch | over-counts host at small L |
| **compile** (`torch.compile`, default) | eager op **+ Dynamo wrapper overhead** around the `@disable` island | makes small-L look like a *tie* when it isn't; never removes launch overhead |
| **cudagraph** (manual / `make_graphed_callables`) | GPU work only, host/launch removed | the real deployment regime for these kernels |

Concrete: single-dir trimul fwd+bwd @ L=384 — `compile` ≈ tie vs dt-v1, but
`cudagraph` = **1.3–1.4×**. Inference @ L=128 — `compile` 0.62 ms vs `cudagraph`
0.10 ms (**6.3× of the wall is pure host overhead**). If you only ever run
`compile=true`, you will conclude these kernels "tie at small L" — which is false.

**Rule:** for graph-break kernels, always report **cudagraph** alongside compile.
`compile`-only is the canonical hard-rule baseline but *misrepresents* launch-bound
ops at small L.

## 2. `@torch.compiler.disable` ⇒ `torch.compile` can never CUDA-graph it

- The op is an opaque eager island. `compile` (any mode) runs it eager and only
  adds Dynamo guard/dispatch overhead — it is **slower than eager**, not faster
  (measured: fwd+bwd L=384 eager 1.33 ms → compiled 1.99 ms).
- `mode="reduce-overhead"` (cudagraph-trees) does **NOT** capture a graph-break /
  disabled region — measured: reduce-overhead gave **zero** speedup on our op.
- Only **manual `torch.cuda.graph` capture** or **`make_graphed_callables`** captures
  the eager kernels and removes host overhead.

Why `@disable` at all: letting Inductor trace in re-codegens our cuBLAS GEMMs into
slower matmuls and graph-breaks at every cute/triton Function (~2× slower; the
`custom_op` variant is ~1.9× slower from save-as-output). `@disable` is the lesser
evil — and the cost is that cudagraph must be applied *manually*.

## 3. Diagnosing a surprising number — a decision tree

- **Ours AND the baseline both moved** between two measurements → it is the
  **regime / harness / library**, NOT a kernel regression. (We chased a single-dir
  0.89→1.5 ms "regression" and a bidir 1.88→1.99 — both were measurement, not code:
  the code was byte-identical.)
- **Same regime, different number** → there is a hidden variable. Find it by
  toggling ONE thing in the SAME bench (see §4). (1.886 vs 1.989 was *the mask*,
  not compile.)
- **A small op looks "free"** → check the regime. Host-bound regimes (eager/compile
  at small L) hide small GPU ops in launch-idle time. The op's true cost only shows
  where it is exposed (cudagraph / large L). The mask multiply looked free under
  compile (~2%, dismissed as noise) but cost **~6%** under cudagraph.

## 4. Isolate ONE variable in the SAME bench — never cross-compare runs

Cross-run comparison conflates: mask on/off, `do_bench` (L2-flush) vs event-timing,
fabric wrapping, warm vs cold autotune cache, even node/thermal. To attribute a
gap, hold everything fixed and toggle the one variable.

Example that settled the mask question: same script, same event-timing, `mask=None`
vs `mask=all-ones` (all-valid → pure multiply cost, no correctness change) →
isolated the mask at +0.10 ms (~6%), and the fold's reduction to ~1.5%.

## 5. Measurement hygiene

- **Compute node only** (`srun`/`sbatch`) — never the login node.
- **Fresh `QUACK_CACHE_DIR`** per run; **enough warmup** so quack/triton autotune is
  fully settled *before* timing. For cudagraph, warmup must trigger every autotune +
  allocation **before capture** (capture forbids new allocations outside the pool).
- **`do_bench` flushes L2 each rep** (cold cache); manual event-timing runs replays
  back-to-back (warm). They give different absolute numbers — don't mix them in one
  comparison.
- **`torch.profiler` self-CPU times are inflated** by instrumentation. Use them only
  for relative attribution; get true host cost from the clean **eager wall − cudagraph
  wall** delta, not from profiler self-time.
- **Old recorded numbers may not reproduce** (warm-cache/luck). Re-measure; don't
  trust a single historical value.
- **SLURM `/tmp` is node-local** — write job output / scripts to a shared path
  (`$HOME`), not `/tmp`, or you lose them.

## 6. CUDA-graph integration gotchas (for real training)

- **`make_graphed_callables`** = single static shape (pad-to-max regime). Auto-manages
  static buffers + input copy. **Cannot wrap the same module at multiple shapes** —
  re-wrapping leaks the first shape (`size 128 must match 256`).
- **Manual capture** = the answer for **bucketed** training (one graph per bucket,
  shared params + pool). You must manage static buffers yourself: allocate once per
  bucket, `static_in.copy_(real_pair)` each step, then `g.replay()`.
- A bench that **reuses a fixed input** omits the per-step `copy_`, so manual looks
  ~5% faster than `make_graphed_callables` (which includes the copy). Real training
  pays the copy either way → they are equal. Choose by shape, not by this artifact.
- **`fabric.backward` vs `loss.backward()`**: fabric-wrapped (bf16-mixed) modules
  forbid raw `.backward()` under capture. Capture the harness's existing step
  (which calls `fabric.backward`), not a hand-rolled backward.
- **`optimizer.zero_grad(set_to_none=False)`** between steps — `set_to_none=True`
  frees the static `.grad` buffers the captured backward writes into.
- Module-scoped capture is safe re RNG **only if** dropout/RNG lives *outside* the
  captured module (it does for AF trimul — dropout is block-level). RNG inside a
  captured region freezes the mask across replays.

## 7. Compare the SAME computation (apples-to-apples)

The miniworld trimul training path originally **ignored the mask** (`del mask`) while
the dt-v1 baseline **applied** it — so with `mask_prob>0` they were not computing the
same thing and the comparison was invalid. Make every implementation honor the same
inputs (mask, dtype, layout) before comparing.

## 8. Misc traps

- **YAML bool trap**: hydra/OmegaConf parse `off`/`on`/`yes`/`no` as booleans. Use
  string sentinels like `disabled` for enum-ish config fields (we hit `cudagraph=off`
  → `False` → validation error).
- **`artifacts/**` is gitignored** (only `.gitkeep` tracked). SVG/CSV are local
  outputs — commit the *code* that regenerates them, not the binaries.
- **`fold` over `multiply`**: a per-row mask applied as a separate `x_n*mask` is ~6%
  (an extra (M,D) HBM round-trip); folded into the LN epilogue (`row_scale`) it is
  ~1%. Don't pay for a separate elementwise when an adjacent kernel already touches
  the tensor.
