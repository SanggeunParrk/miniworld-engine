import sys, re
root = sys.argv[1]; C = root + "/src/foldjet_engine/kernels/triangle_attention/cuda/"
p = C + "b200_sources/sm100.cuh"; s = open(p).read()
a = "// bf16 / fp32 reduce-add of a row tile into the caller's tensor"
helper = '''// (N1) tile visiting order of the persistent row-tile kernels. The ending node's x / y tiles are 128 rows (b, n, s0 .. s0 + 127) at the caller's
// [b, s, n]: one 256-byte piece a row, 128 rows L * 256 B apart. Consecutive CTAs take consecutive logical tiles = consecutive s blocks of one n: their
// pieces are far apart. With TA_TPERM the sequence number runs over n fastest (the same s block of 148 consecutive n at once: 148 x 256 B contiguous a row).
#ifndef TA_TPERM
#define TA_TPERM 0
#endif
__device__ __forceinline__ int tperm(int sq, int L, int g) {
  if (!TA_TPERM || !(g & 1)) return sq;
  const int nb = L >> 7, nt1 = L * nb, b = sq / nt1, r = sq - b * nt1;
  return b * nt1 + (r % L) * nb + r / L;
}
'''
assert a in s
s = s.replace(a, helper + a, 1); open(p, "w").write(s)
p = C + "b200_sources/triattn_mod_sm100.cu"; s = open(p).read()
i0 = s.index("    int R, int L, int ntiles, float eps,")   # front kernel params
i1 = s.index("// K = 128; double-buffered TMEM")             # end of front region
body = s[i0:i1]
old = "for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {"
n = body.count(old); print("front loops", n)
body = body.replace(old, "for (int sq_ = blockIdx.x; sq_ < ntiles; sq_ += gridDim.x, ++lt) {\n      const int t = tperm(sq_, L, TRANS);")
s = s[:i0] + body + s[i1:]
open(p, "w").write(s)
p = C + "b200_sources/tailg_sm100.cuh"; s = open(p).read()
old = "  auto tile = [&](int i) { return (int)blockIdx.x + i * (int)gridDim.x; };"
assert old in s
s = s.replace(old, "  auto tile = [&](int i) { return tperm((int)blockIdx.x + i * (int)gridDim.x, L, TRANS); };")
open(p, "w").write(s)
p = C + "b200_triattn.py"; s = open(p).read()
old = '''def _mod():
    ensure_cuda_home()
    return load_extension(
        name="triattn_b200_mod_v22_tt10", sources=[str(_SRC / "triattn_mod_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False
    )'''
new = '''#: (N1) the ending node's row-tile kernels visit tiles n-fastest (`tperm`, sm100.cuh). FOLDJET_ENGINE_TRIATTN_B200_TPERM=0 turns it off.
TPERM = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_TPERM", "1") != "0"


def _mod():
    ensure_cuda_home()
    return load_extension(
        name="triattn_b200_mod_v22_tt10" + ("_tp" if TPERM else ""),
        sources=[str(_SRC / "triattn_mod_sm100.cu")],
        extra_cuda_cflags=[*_flags(), *(("-DTA_TPERM=1",) if TPERM else ())],
        verbose=False,
    )'''
assert old in s
s = s.replace(old, new)
old = '''    return load_extension(
        name="triattn_b200_wide_v45_tt7", sources=[str(_SRC / "triattn_wide_sm100.cu")], extra_cuda_cflags=_flags(), verbose=False
    )'''
new = '''    return load_extension(
        name="triattn_b200_wide_v45_tt7" + ("_tp" if TPERM else ""),
        sources=[str(_SRC / "triattn_wide_sm100.cu")],
        extra_cuda_cflags=[*_flags(), *(("-DTA_TPERM=1",) if TPERM else ())],
        verbose=False,
    )'''
assert old in s
s = s.replace(old, new)
open(p, "w").write(s); print("ok")
