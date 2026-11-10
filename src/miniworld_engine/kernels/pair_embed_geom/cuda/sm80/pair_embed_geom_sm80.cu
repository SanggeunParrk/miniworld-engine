// pair_embed_geom_sm80.cu -- the A100 (sm_80) build of sm90/pair_embed_geom.cu: the same kernels and host entry points, the source compiled
// unchanged (the H100 build's file stays byte-identical); pe_sm80.cuh supplies the sm_80 forms of its five sm_90-only helpers (the table
// gradient's warpgroup product as mma.sync on the same swizzled shared-memory operands).
#include "pe_sm80.cuh"
#include "../sm90/pair_embed_geom.cu"
