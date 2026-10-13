"""Overlap D384 K1 gradient loads with MMA; retain B200 fusion and staging."""

from functools import lru_cache
from pathlib import Path
import hashlib
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


@lru_cache(None)
def headers(base):
    names = ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh")
    payload = Path(__file__).read_bytes() + b"".join(
        (base / name).read_bytes() for name in names
    )
    payload += (T.SOURCES / "b200style/k1wb_epilogue.inc").read_bytes()
    dest = T.cache_dir() / (
        "k1_dpl_prefetch_" + hashlib.sha256(payload).hexdigest()[:16]
    )
    for name in ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh"):
        source = (base / name).read_text()
        if name == "tmn_kernels.cuh":
            lead = (
                (T.SOURCES / "b200style/k1wb_epilogue.inc")
                .read_text()
                .split("mbar_wait(barD_full")[0]
            )
            assert source.count(lead) == 1
            source = source.replace(lead, "")
            anchor = "    auto epilogue = [&](float (&ac)[32], int b, uint32_t wi, bool release) {"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor,
                "    auto prefetch_dpl = [&](int b) {\n" + lead + "\n    };\n" + anchor,
            )
            anchor = "      for (int b = 0; b < NBLK; ++b, w_iter += SPB) {\n        static_for<SPB>"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor,
                "      for (int b = 0; b < NBLK; ++b, w_iter += SPB) {\n        prefetch_dpl(b);\n        static_for<SPB>",
            )
            anchor = "    issue_block(acc0, w_iter);\n#pragma unroll 1"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor,
                "    issue_block(acc0, w_iter);\n    prefetch_dpl(0);\n#pragma unroll 1",
            )
            anchor = "      epilogue(acc0, b, w_iter, true);"
            assert source.count(anchor) == 1
            source = source.replace(anchor, anchor + "\n      prefetch_dpl(b + 1);")
            anchor = "      epilogue(acc1, b + 1, w_iter + SPB, true);"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor, anchor + "\n      if (b + 2 < NBLK) prefetch_dpl(b + 2);"
            )
        T.publish_header(dest / name, source)
    return dest
