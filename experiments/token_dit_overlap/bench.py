"""One timing routine for this package, the engine's: triton's do_bench, which zeroes an L2-sized buffer before every
timed iteration. Replaying a CUDA graph instead (what the earlier scripts here did) leaves the operands hot in L2, which
flatters whichever variant re-reads the most -- exactly the comparison a fused kernel is trying to win.

`settings.bench_clear_mb` sizes that buffer in the engine; do_bench's default provider (256 MB) evicts an H100's 50 MB L2
just as completely, so this uses the default unless the engine has one installed.
"""
import triton.testing


def us(fn, warmup=25, rep=100):
    """Median microseconds per call, L2 evicted between iterations."""
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")) * 1e3


def us_all(fn, warmup=25, rep=100):
    """(median, min, max) microseconds."""
    q = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.0, 1.0])
    return tuple(float(x) * 1e3 for x in q)
