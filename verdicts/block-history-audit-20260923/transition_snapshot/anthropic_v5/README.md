# Anthropic trimul native v5 device headers (vendored)

Three device headers from Anthropic's `trimul` native v5 kernel package, upstream revision
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, Apache-2.0. They are the sm_90a primitive layer
(`wgmma`, TMA, mbarrier, `ldmatrix`/`stmatrix`, swizzle helpers) that
`transition_fused_{fwd,bwd}_sm90a_kernel.cu` are written against.

They live inside the package, not under `experiments/`, so the wired kernel builds from an
installed engine and not only from a source checkout. The copy under
`experiments/trimul_b7b12/vendor/anthropic_v5/csrc/` is the same three files and is what
`experiments/transition_fused/verify_package.py` hashes.

    sha256  953ebc8f0c6668976b7282fd6ecdd360e98448ef67a32ebe0ba894060fe849e6  tmn_kernels.cuh
    sha256  f774e39f8346998f96f9c839d142fe29d2733c2b4ef5e811e05b7165ea6fb2aa  tmn_ptx.cuh
    sha256  3504d522868ad81d2d072a3a744fe042da5b3761736729aca1a5be3fbfe57f8b  common/tmn_math.cuh

Unmodified. Do not edit them here; re-vendor instead.
