"""Keep a dedicated dplane tile so its next TMA load overlaps the epilogue."""

from functools import lru_cache
from pathlib import Path
import hashlib
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


@lru_cache(None)
def headers(base):
    names = ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh")
    payload = Path(__file__).read_bytes() + b"".join(
        (base / n).read_bytes() for n in names
    )
    dest = T.cache_dir() / ("dpl_pipeline_" + hashlib.sha256(payload).hexdigest()[:16])
    for name in names:
        s = (base / name).read_text()
        if name == "tmn_kernels.cuh":

            def replace(old, new):
                nonlocal s
                assert s.count(old) == 1, (old, s.count(old))
                s = s.replace(old, new)

            replace(
                "static constexpr int SMEM_STAGE = NCWG * 8192;",
                "static constexpr int SMEM_STAGE = NCWG * 12288;",
            )
            replace(
                "  extern __shared__ __align__(1024) uint8_t smem[];\n  uint8_t* sA = smem;",
                '  static_assert(G::SCHED == 1, "dplane pipeline requires schedule 1");\n'
                "  extern __shared__ __align__(1024) uint8_t smem[];\n  uint8_t* sA = smem;",
            )
            replace(
                "  const size_t plane = (size_t)NPI * (size_t)p.Np;",
                "  const uint32_t dpl_u = smem_u32(sStage) + G::NCWG * 8192 + cw * 4096;\n"
                "  const size_t plane = (size_t)NPI * (size_t)p.Np;",
            )
            replace(
                "    auto prefetch_dpl = [&](int b) {\n"
                "// Included inside Hopper K1's epilogue after all preactivation MMAs retire.\n"
                "// Recompute sigmoid/projection into a token-major SW128 staging tile.\n"
                "// Drain the previous async shared read before reusing this warpgroup's tile.\n"
                "if (st_elect) tma_store_wait_read<0>();\n"
                "fence_proxy_async();\n"
                "named_bar_sync(bar_id, 128);\n"
                "// The same 8 KB stage first receives [32 channels][64 tokens] dplanes.\n"
                "// Every lane reads it before the whole warpgroup transposes into dcat.\n",
                "    auto prefetch_dpl = [&](int b) {\n"
                "// Dedicated 4 KB per warpgroup, disjoint from the dcat output stage.\n",
            )
            replace(
                "tma_load_2d(sStage + cw * 8192, &p.tm_dpl, barD_full + cw,",
                "tma_load_2d(sStage + G::NCWG * 8192 + cw * 4096, &p.tm_dpl, barD_full + cw,",
            )
            replace(
                "ldsm_x4_t(da[0], stage_u + sts_off0);\nldsm_x4_t(da[1], stage_u + sts_off1);",
                "ldsm_x4_t(da[0], dpl_u + sts_off0);\n"
                "ldsm_x4_t(da[1], dpl_u + sts_off1);\n"
                "// All generic-proxy reads complete before the next async-proxy refill.\n"
                "fence_proxy_async();\n"
                "named_bar_sync(bar_id, 128);\n"
                "if (b + 1 < NBLK) prefetch_dpl(b + 1);",
            )
            replace(
                "// WAR: the channel-major input and token-major output share the stage.\n"
                "named_bar_sync(bar_id, 128);",
                "// Reuse dcat only after the preceding store consumed its shared source.\n"
                "if (st_elect) tma_store_wait_read<0>();\n"
                "fence_proxy_async();\n"
                "named_bar_sync(bar_id, 128);",
            )
            replace(
                "      (void)acc1; (void)issue_block;",
                "      (void)acc1; (void)issue_block;\n      prefetch_dpl(0);",
            )
            replace(
                "      for (int b = 0; b < NBLK; ++b, w_iter += SPB) {\n        prefetch_dpl(b);",
                "      for (int b = 0; b < NBLK; ++b, w_iter += SPB) {",
            )
        T.publish_header(dest / name, s)
    return dest
