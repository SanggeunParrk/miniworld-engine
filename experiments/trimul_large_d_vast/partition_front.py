"""Short-shape experiment: distribute complete output-channel groups over CTAs.

Each CTA keeps the original input normalization, MMA accumulation and gate math.
Only y==0 owns the common normalized activation save. AB/pre saves are disjoint.
"""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from wide_saved_front_overlap import SavedOverlapFront

def build(plan, parts):
    # D384 normalizes inside the front. Its start-credit barrier depends on the
    # unpartitioned channel index; this rejected experiment is unsafe there.
    assert plan.p.D == 512 and plan.p.n == 384
    assert parts in (1, 2, 4, 8) and (plan.p.D // 8) % parts == 0
    original_compile = T.compile_text
    def compile_partition(source, flags):
        # Both the original header and shared-input variant are included.
        source = source.replace('NBLK = G::NBLK,', f'NBLK = G::NBLK / {parts},')
        source = source.replace('64 * b);', f'64 * (b + blockIdx.y * (G::NBLK / {parts})));')
        source = source.replace('if (EMITX) {', 'if (EMITX && blockIdx.y == 0) {')
        marker = 'auto epilogue = [&](float (&ac)[32], int b, uint32_t wi, bool release) {'
        assert marker in source
        source = source.replace(marker, marker + f' b += blockIdx.y * (G::NBLK / {parts});')
        return original_compile(source, flags)
    T.compile_text = compile_partition
    try:
        op = SavedOverlapFront(plan.baseline_front, plan.pre)
    finally:
        T.compile_text = original_compile
    class Partition:
        def __init__(self): self.__dict__.update(op.__dict__)
        def __call__(self):
            if not self.normalize: F.normalize_into(self.xn, self.x, self.gi, self.bi)
            self.k.launch((self.grid, parts, 1), (self.threads, 1, 1), [self.params], self.smem)
    return Partition()
