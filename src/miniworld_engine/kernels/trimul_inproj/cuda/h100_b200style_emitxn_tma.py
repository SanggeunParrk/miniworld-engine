"""Emit D256/D384 backward normalized inputs through the existing K1 epilogue stage."""

from functools import lru_cache
from pathlib import Path
import hashlib
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


@lru_cache(None)
def headers(base):
    names = ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh")
    payload = Path(__file__).read_bytes() + b"".join(
        T.read_header(base / n).encode() for n in names
    )
    dest = T.cache_dir() / ("emitxn_tma_" + hashlib.sha256(payload).hexdigest()[:16])
    for name in names:
        s = T.read_header(base / name)
        if name == "tmn_kernels.cuh":
            anchor = "  CUtensorMap tm_dcat, tm_dpl;"
            assert s.count(anchor) == 1
            s = s.replace(anchor, "  CUtensorMap tm_dcat, tm_dpl, tm_xn;")
            start = s.index("    if (EMITX) {")
            end = s.index("    if (SAVE) {", start)
            s = (
                s[:start]
                + """    if (EMITX) {
      // The preceding persistent tile may still have a dcat TMA store in flight.
      if (st_elect) tma_store_wait_read<0>();
      fence_proxy_async();
      named_bar_sync(bar_id, 128);
      const int tr = 16 * wiw + 8 * (idx & 1) + rr;
      const int cb = 8 * (idx >> 1);
      static_for<KS / 4>([&](auto k_) {
        constexpr int k = decltype(k_)::value;
        static_for<4>([&](auto q_) {
          constexpr int q = decltype(q_)::value;
          stsm_x4(stage_u + swz128(tr, 2 * (cb + q * 16)),
                  fa[k * 4 + q][0], fa[k * 4 + q][1],
                  fa[k * 4 + q][2], fa[k * 4 + q][3]);
        });
        fence_proxy_async();
        named_bar_sync(bar_id, 128);
        if (st_elect) {
          const int col = k * 64, tok = iw * p.N + jw;
          asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\\n"
                       :: "l"(&p.tm_xn), "r"(stage_u), "r"(col), "r"(tok) : "memory");
          tma_store_commit();
          tma_store_wait_read<0>();
        }
        // All lanes wait until the TMA proxy has consumed the reusable stage.
        fence_proxy_async();
        named_bar_sync(bar_id, 128);
      });
    }
"""
                + s[end:]
            )
        T.publish_header(dest / name, s)
    return dest
