"""Live weight packing without a nested torch.compile runtime invocation."""
import triton
import triton.language as tl


@triton.jit
def _pack(DST, WL, WLG, WR, WRG, R: tl.constexpr, D: tl.constexpr,
          LS0: tl.constexpr, LS1: tl.constexpr,
          LGS0: tl.constexpr, LGS1: tl.constexpr,
          RS0: tl.constexpr, RS1: tl.constexpr,
          RGS0: tl.constexpr, RGS1: tl.constexpr,
          BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    source_row = (row // 64) * 32 + row % 32
    right = source_row >= R
    gate = row % 64 < 32
    r = source_row % R
    ptr = tl.where(right,
                   tl.where(gate, WRG + r * RGS0 + col * RGS1,
                            WR + r * RS0 + col * RS1),
                   tl.where(gate, WLG + r * LGS0 + col * LGS1,
                            WL + r * LS0 + col * LS1))
    tl.store(DST + i, tl.load(ptr, i < 4 * R * D, other=0), i < 4 * R * D)


def pack_into(dst, wl, wlg, wr, wrg):
    r, d = wl.shape
    if r % 32 or dst.shape != (4 * r, d) or not dst.is_contiguous():
        raise ValueError("weight pack requires 32-row groups and contiguous output")
    if any(w.shape != wl.shape or w.dtype != dst.dtype or w.device != dst.device
           for w in (wl, wlg, wr, wrg)):
        raise ValueError("weight pack inputs must share shape, dtype and device")
    _pack[(triton.cdiv(4 * r * d, 1024),)](
        dst, wl, wlg, wr, wrg, r, d,
        *wl.stride(), *wlg.stride(), *wr.stride(), *wrg.stride(),
        BLOCK=1024, num_warps=4)
