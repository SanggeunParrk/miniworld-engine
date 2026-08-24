# Contributing to miniworld-engine

The bar this repo holds itself to is written down in
[`docs/library-standards.md`](docs/library-standards.md), and the open work against it is in
[`plan.md`](plan.md). Read the first before proposing a change to the second.

## The three gates

One command runs all of them, in CI's order:

```bash
pixi run ci        # ruff check → ty check → pytest -m 'not gpu'
```

Individually:

```bash
pixi run ruff-check     # ruff check src tests benchmarks tools
pixi run types          # ty check src tests benchmarks tools
pixi run test           # the whole CPU suite, ~60 s, no GPU
pixi run test-gpu       # the GPU suite (numerical + compile-regime); needs an allocated card
```

`.github/workflows/ci.yml` runs the same three over the same paths. All three **gate** — there is no
`|| true` anywhere, and a green step that checks nothing is treated as a bug (it has happened here:
the `ty` step once ran against an install with no torch, so every attribute was `Unknown`).

The CPU suite must stay CPU-only and fast. A test that needs a device goes behind
`@pytest.mark.gpu`.

## What a change includes

**A test that fails without it.** Not a test that exercises the new code — a test that reproduces
the defect. If the change is a refactor with no behaviour delta, the test is the invariant the
refactor establishes (see `tests/test_kernel_layout.py`, `tests/test_bench_target_vocabulary.py`:
each exists because something drifted silently).

**A CHANGELOG entry for anything a consumer can observe** — the public surface, the CLI, a config
key, a default, an error message they might match on. Entries say what was wrong, not just what
changed.

**A reason with every suppression.** `# noqa: RULE -- why` and `# ty: ignore[rule] -- why`. A bare
suppression fails review, and a `# noqa` for a rule the config does not select fails `ruff`
(RUF100). If a rule is wrong for this codebase, turn it off in `pyproject.toml` with the
measurement that justifies it — that is what the existing exclusions do.

**A commit message that states the defect.** The convention here is a subject line naming the
change and a body naming what was broken and how it was found. `git log` is the repo's reasoning
record; treat it as one.

## Where things go

| you are adding | it goes | the shape is enforced by |
|---|---|---|
| a fused kernel | `src/miniworld_engine/kernels/<family>/` | `tests/test_kernel_layout.py` |
| a model-level op | `src/miniworld_engine/modules/<op>/module.py` | `tests/test_module_layout.py` |
| a kernel's launcher / checker | `kernels/drivers/<family>.py`, `kernels/checks/<family>.py` | `tests/test_registry_complete.py` |
| a benchmark target | `benchmarks/{kernels,modules}/<target>/` + a `configs/bench.yaml` | `tests/test_bench_config_per_target.py` |
| a public name | `kernels/__init__.py` or `ops/__init__.py`, **and** `_CONTRACT` in `tests/test_public_api.py` | `tests/test_public_api.py` |

Every new kernel needs a row in `src/miniworld_engine/kernels/registry.csv`: that file is the
declared inventory, and coverage is measured against it rather than against whatever ran. A kernel
with no driver is reported `untested` — a visible hole, never a pass.

## Naming

One name per thing, across code, CLI, docs, config and directories. A kernel bench target is named
after its family in `registry.csv`; a module bench target after the module it constructs; the two
levels are separate namespaces, which is why `triangle_attention` is legal as both.
`tests/test_bench_target_vocabulary.py` ties the four namespaces together and will reject a name
that exists in only three of them.

No abbreviations that the engine does not itself use as a canonical name. `adaln` is fine (it is a
package name); `tri_attn` is not.

## Vendored code

`kernels/**/{triton,cute,cuda}/` and the `baseline_dtv1*` modules are faithful ports. They are
excluded from lint and from the type checker — the exclusions and the reason live together in
`pyproject.toml`. Do not restyle them; a diff against upstream is worth more than local
consistency there.

## Benchmarks

A number without provenance is not a result. The long-form CSV is the artifact: it records device,
torch and CUDA versions, mode, `compiled`, `cudagraph`, `compile_wrap`, precision, dtypes and the
execution path, per row. Do not quote a figure that no committed table can produce.

Pick the regime deliberately. `cudagraph=manual` is the default only so the committed tables stay
reproducible — measured on an A6000, the compile-only regime *beats* eager+graph by up to 4.46x on
modules with unfused work around their kernels. The default is not a recommendation.

## Running out of context on the cache

`miniworld-engine build all` is hours of GPU time and rewrites this card's entries in
`src/miniworld_engine/autotune/data/`. That directory is a build artifact that lives inside the
package; the pre-commit hook (`.githooks/pre-commit`) refuses to mix cache files with code in one
commit, because that has happened four times. Commit a cache on purpose, once, when a build is
finished:

```bash
git config core.hooksPath .githooks     # once per clone
```

## Deprecation

Removing a public name is a two-step process, and both steps are enforced.

**Step 1 — deprecate.** Add the name to `kernels._DEPRECATED` (or `ops`') with a message that says
why and **what to use instead**; a bare "deprecated" just makes the consumer grep this repo. The
name keeps working. Add a `### Deprecated` entry to the CHANGELOG for that release.

**Step 2 — remove, no earlier than two releases later.** Drop it from `__all__` / `_LAZY_EXPORTS`
and from `_CONTRACT` in `tests/test_public_api.py`, in the same commit, with a CHANGELOG entry
under `### Removed`.

What holds this up:

- `tests/test_public_api.py` freezes the surface, so a removal that skips the CHANGELOG fails.
- it also asserts every `_DEPRECATED` name is still in `__all__` (deprecated is not removed), that
  each message names a replacement, and that using the name really does warn.
- "using" has two shapes and both count: most of the surface resolves through `__getattr__`, so
  for those *resolution is the use* and the warning fires on attribute access; three names are
  plain module-level functions, where `__getattr__` never runs and access alone is not use
  (`hasattr`, `dir()` and a re-export would all warn for nothing), so there the call is the use.
- a deprecated lazy name is deliberately **not** cached into `globals()`. Caching is what makes
  `__getattr__` run once per process, and a warning that fires only on the first access in a
  long-lived process is one most callers never see.

Currently deprecated: `kernels.cuda_transition` — it has never had an implementation (it deferred
to a `transition/cuda` symbol git has no record of) and calling it raises `NotImplementedError`.
It is in the frozen surface, so it could not simply be deleted; now it says so.
