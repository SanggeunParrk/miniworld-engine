# Third-party notices

MiniWorld Engine previously developed its own inference kernels. Anthropic's
published biomolecular modeling implementation achieved substantially better
results on the workloads we studied. Version 2 builds on that work with explicit
attribution; our contribution is its integration, measured adaptations, and
training forward/backward kernels. We do not claim the upstream inference
implementation as independent MiniWorld work, or universal superiority over it.

Upstream: https://github.com/anthropics/uplifting-biomolecular-modeling
Pinned revision: `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.
The exact original repository URL and source hashes are also recorded in
`experiments/trimul_b7b12/vendor/anthropic_v5/UPSTREAM.json`.

Anthropic native v5 sources and derived TMA/WGMMA device code retain Apache-2.0
terms. See [the license](licenses/Anthropic-Apache-2.0.txt) and the preserved
`NOTICE`, `LICENSE` and `UPSTREAM.json` files in the vendored experiment. This
includes Transition's `cuda/anthropic_v5/` headers and the research capsule's
upstream snapshot. Original notices in source files remain in place. Other
MiniWorld code remains under the root MIT license. The isolated engine snapshot
in the research capsule retains that same MIT license for engine-owned code.

The installed TriMul runtime also includes selected native v5/overlay sources in
`src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/` and adapted native
Python driver/launch helpers in its parent directory. `PROVENANCE.json` maps
selected source bodies to the preserved experiment archives and their hashes.
The corresponding Apache-2.0 license applies to these inherited portions.
