import re, sys
root = sys.argv[1]
K = root + "/src/foldjet_engine/kernels/"
# 1. loader
p = K + "triangle_attention/cuda/b200_bidir.py"; s = open(p).read()
s += '''

@functools.lru_cache(maxsize=None)
def lnb():
    """The B200 LayerNorm backward of the bias-only paths (``b200_sources/lnbwd_sm100.cu``, C = 128): the shared glue's ``ln_bwd`` with its
    row loads issued together and a row step ahead, a persistent grid (fewer d gamma / d beta partial rows). dx bitwise equal."""
    ensure_cuda_home()
    src = Path(__file__).parent / "b200_sources"
    return load_extension(
        name="triattn_b200_lnbwd_v1",
        sources=[str(src / "lnbwd_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *gencodes("100a")],
        verbose=False,
    )
'''
open(p, "w").write(s)
# 2. integration
p = K + "integrations/bidir_triattn_h100.py"; s = open(p).read()
anchor = "def _mm_f32(a, b):"
helper = '''#: (N1, cuda:b200_sm100 71 / Bidirectional 5) the bias-only paths' LayerNorm backward (C = 128) on the B200 kernel of
#: ``b200_sources/lnbwd_sm100.cu``: x / dyln / dy of a row in flight together and a row step ahead, 2 CTAs an SM (fewer partial rows for the
#: column sum). dx bitwise equal, d gamma / d beta fp32 sums regrouped. MINIWORLD_BIDIR_B200_LNBWD=0 turns it off.
B200_LNBWD = os.environ.get("MINIWORLD_BIDIR_B200_LNBWD", "1") != "0"


def _ln_bwd(g, dyln, x, dy, lnw, mean, rstd, L):
    """-> (dx, partials [n, 2, C]) of the LayerNorm backward + the residual's dy (the shared glue's ``ln_bwd`` in the starting layout)."""
    if B200_LNBWD and _b200() and x.shape[1] == 128:
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        return b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2)
    return g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)


'''
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)
old = "    dx, part = g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)  # + the residual's dy"
assert s.count(old) == 2
s = s.replace(old, "    dx, part = _ln_bwd(g, dyln, x, dy, lnw, mean, rstd, L)  # + the residual's dy")
open(p, "w").write(s)
# 3. versions
p = root + "/src/foldjet_engine/modules/triangle.py"; s = open(p).read()
a = '        "cuda:b200_sm100": 69,  # 69 ('
assert s.count(a) == 1
s = s.replace(a, '        "cuda:b200_sm100": 71,  # 71 (2026-11-18, N1 triattn, training): the bias-only paths\' LayerNorm backward (C = 128) on a B200 kernel (loads issued together and a row step ahead, 2 CTAs an SM): the single-direction row 188 step -2.2 .. -4 %, dx bitwise equal;\n        # 69 (', 1)
b = '        "cuda:b200_sm100": 4,  # 4 ('
assert s.count(b) == 1
s = s.replace(b, '        "cuda:b200_sm100": 5,  # 5 (2026-11-18, N1 triattn, training): the LayerNorm backward (C = 128) on a B200 kernel (shared with the single-direction bias-only path of TriangleAttention), dx bitwise equal;\n        # 4 (', 1)
open(p, "w").write(s)
print("ok")
