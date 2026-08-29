# When it goes wrong

Each section is a message you will actually see, what produces it, and the command that ends it.
Every one of these cost time before it was diagnosed; the point of writing them down is that the
second person does not pay again.

`tests/layout/test_cli_documented_commands.py` and the checks noted below tie these to the source,
so a renamed message cannot leave a section here describing something that no longer happens.

---

## A build hangs with no output

> *(nothing — the job sits, `squeue` says RUNNING, the log has not moved in ten minutes)*

Two causes, and they look identical from outside.

**A leftover JIT build lock.** `torch.utils.cpp_extension.load` serialises builds with a
`FileBaton`, and `FileBaton.wait()` polls forever. A build that died — OOM, a kill, a node failure
— leaves its lock behind, and every later call waits on a lock nobody holds. This cost thirteen
hours once.

`load_extension` now bounds it: a lock older than 30 minutes is reclaimed, a fresh one is waited on
for 10 minutes and then raises naming the file and the `rm`. A failed build also takes its own lock
with it. If you are on an older checkout, or want to clear the ground first:

```bash
find ~/.cache/torch_extensions -name lock -delete
```

**Your own pipe.** `sbatch ... | head -40` or `| tail` blocks until the producer exits, so the log
stays empty while the job runs perfectly. Two full afternoons went to this. Do not filter a job's
stdout; write it whole and grep the file afterwards.

Check which one you have before assuming: `squeue`, then look for a `ninja` or `nvcc` process on
the node. No compiler process and a lock file present is the first case.

---

## `dynamic_func() missing 2 required positional arguments: 'BLOCK_M1' and 'BLOCK_K'`

Every triton kernel dies at its first launch. The op has no config list, so triton falls back to a
substitute config that does not carry the tile axes.

Either no config set was found, or `MINIWORLD_CONFIG_DIR` names one that does not exist. The second
case used to be silent — a stale value looked exactly like a good one — and now raises with the
value and both ways out. Unset the variable to use the packaged `grid`:

```bash
unset MINIWORLD_CONFIG_DIR
```

---

## `unknown config set 'configs/grid'`

The config sets moved into the package. Short names (`grid`, `blk16`, `accuracy`, …) and real paths
both work, and `configs/<name>` is still accepted for the command lines that predate the move. A
name outside the list is a typo, and it fails rather than falling back to the default. The error
prints the list.

---

## `cuBLASDx headers not found`

Three CUDA transition extensions include cuBLASDx. Point one of `MINIWORLD_MATHDX_HOME`,
`MATHDX_HOME` or `NVIDIA_MATHDX_HOME` at a mathdx root — the directory containing
`include/cublasdx.hpp` — or `pip install nvidia-mathdx`. The message lists every location it tried.

Nothing else in the library needs it; the other kernels build without mathdx.

---

## `no tuned autotune cache entry for this shape`

A warning, not an error. The op runs, searching a heuristic subset of its grid instead of using a
tuned config, so it is slower and may pick a worse config. Expected on a card with no shipped
cache, or at a shape outside the tuned buckets.

```bash
miniworld-engine build all --gpus <n> --shards ~/.cache/miniworld-build
```

That is hours of work. If you only need the shapes you actually run, build the ops you use rather
than `all`.

---

## `build all` reports far fewer units than expected

The unit count is `(op, dtype, shape-bucket)` over the registry. If it says 527 where it should say
859, the checkout predates the fp32 dtype fix and half of every fp32 kernel's work is missing —
while coverage still reports `missing 0`, because it counted against what the build enumerated
rather than against the registry. `git pull`.

---

## A kernel is `skipped (wrong card smXX, or not declared at bf16)`

Two reasons, one line, and neither is a failure — the kernel was never launched.

**Wrong card**: its `arch` is above yours. Nine kernels are declared sm90/sm100; see
`docs/supported.md` for which, and note that `tuned_for` is informational while `arch` is the
enforced gate.

**Not declared at this precision**: `registry.csv`'s `dtypes` column says which precisions the
kernel runs at, and a run at another one passes it by. Export `MINIWORLD_DRIVER_DTYPE=fp32` (or
`bf16`) to run the other half; between the two, every declared kernel is reached.

Both are decided before the driver runs. Driving a kernel a run is not going to judge costs a
compile, and a compile that fails gets recorded as the kernel's failure — which is how an fp32 run
once reported `trimul_outproj_layernorm_gemm_gate_triton`, a bf16-only row, as FAILED.

---

## Two runs of the same kernel give different bytes

Expected for one kernel and one only: `augmented_attention_bwd_atomic_triton` accumulates with
unordered atomics. `tests/numerics/test_determinism_gpu.py` names it in `NOT_BITWISE` with the
reason, asserts it still differs, and asserts the difference stays inside its declared band.
Anything else differing run to run within one process is a bug — report it with the two outputs.

Across a cache rebuild, a different GPU or a different config set, last-bit differences are
expected everywhere: a different config means a different reduction order.

---

## `miniworld-engine: error: unrecognized arguments: --bench-budget`

Removed in 1.0.0, deliberately. It compared one drained-stream launch against `do_bench`'s queued
median — quantities that differ by 10x here — so the first config in grid order won and the cache
was wrong while the build looked 37% faster. Drop the flag. See CHANGELOG 1.0.0 for the
measurements; caches built with it need rebuilding.

---

## `import miniworld_kernels` fails

The distribution and import name became `miniworld-engine` / `miniworld_engine` in 1.0.0. There is
no alias. Pin `miniworld-engine>=1.0.0` and rewrite the import; a submodule consumer also updates
the path and URL.
