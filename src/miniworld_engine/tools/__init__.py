"""One-off probes that answer a question about this library's own kernels.

Not a public API and not a stable one: each module here was written to settle one question --
does the hand-entered `kind` match what the source does, how many buckets does the build matrix
really produce, which launch parameter is bound where -- and several are cited from `docs/kernels/`
as the method behind a number.

They live inside the package rather than in a top-level `tools/` because they are about the
kernels next to them, and because the tests that consume them were reaching in with
a `sys.path.insert` into a top-level `tools/`. A plain import is better than a path hack.

The `.sbatch` launchers beside them are for this cluster; they are not Python and do not ship.
"""
