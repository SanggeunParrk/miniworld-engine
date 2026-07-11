# Third Party

External dependencies for this repo are consumed as **pip packages**, not git
submodules, so a parent project (e.g. team-gm) that vendors miniworld-kernels as
a submodule doesn't inherit a recursive-clone of large upstreams.

- **CUTLASS / CuTeDSL**: the `.cute` kernel backends use the `nvidia-cutlass-dsl`
  wheel (plus `quack-kernels`), pinned in the `[cute]` optional-dependency group
  in `pyproject.toml`. The former `ct_cutlass_workbench/cutlass` git submodule
  (an uninitialized, unreferenced gitlink — migration debt) has been removed.

Add new externals as pinned pip dependencies (an extra in `pyproject.toml`)
rather than submodules whenever a wheel exists.
