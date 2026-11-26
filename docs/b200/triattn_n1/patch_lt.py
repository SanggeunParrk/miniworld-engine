import sys
root = sys.argv[1]; K = root + "/src/foldjet_engine/kernels/"
src = open(sys.argv[2]).read()
open(K + "triangle_attention/cuda/b200_sources/ltmm_sm100.cu", "w").write(src)
p = K + "triangle_attention/cuda/b200_bidir.py"; s = open(p).read()
s += '''

@functools.lru_cache(maxsize=None)
def ltmm():
    """Tuned cublasLt bf16 -> fp32 GEMMs (``b200_sources/ltmm_sm100.cu``): the first call of a shape outside a graph capture times the heuristic's 32
    candidates (deterministic and accurate ones only) and keeps the fastest; used for the bias-only backward's tall-K contractions."""
    ensure_cuda_home()
    src = Path(__file__).parent / "b200_sources"
    return load_extension(
        name="triattn_b200_ltmm_v1",
        sources=[str(src / "ltmm_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *gencodes("100a")],
        verbose=False,
    )
'''
open(p, "w").write(s)
p = K + "triangle_attention/cuda/__init__.py"; s = open(p).read()
assert "yield b200_bidir.lnb" in s
s = s.replace("        yield b200_bidir.lnb   # (N1) the bias-only LayerNorm backward at C = 128\n", "        yield b200_bidir.lnb   # (N1) the bias-only LayerNorm backward at C = 128\n        yield b200_bidir.ltmm  # (N1) tuned cublasLt tall-K GEMMs of the bias-only backward\n")
open(p, "w").write(s)
p = K + "integrations/bidir_triattn_h100.py"; s = open(p).read()
old = "def _mm_f32(a, b):\n    return torch.mm(a, b, out_dtype=torch.float32)\n"
new = '''#: (N1, cuda:b200_sm100 71 / Bidirectional 5) the tall-K fp32-output contractions of the bias-only backward (dWo, dW, dP: K = 16384 .. L L) on tuned cublasLt
#: algorithms (``b200_sources/ltmm_sm100.cu``, chosen per shape by a deterministic / accurate timing sweep at the first eager call). FOLDJET_ENGINE_BIDIR_B200_LT=0 turns it off.
B200_LT = os.environ.get("FOLDJET_ENGINE_BIDIR_B200_LT", "1") != "0"


def _lt():
    if not (B200_LT and _b200()):
        return None
    from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

    return b200_bidir.ltmm()


def _mm_f32(a, b):
    e = _lt()
    if e is not None and a.shape[1] >= 16384 and (a.stride(-1) == 1 or a.stride(-2) == 1) and (b.stride(-1) == 1 or b.stride(-2) == 1):
        return e.lt_mm(a, b, a)
    return torch.mm(a, b, out_dtype=torch.float32)


def _bmm_f32(a, b):
    e = _lt()
    if e is not None and a.shape[2] >= 16384 and (a.stride(-1) == 1 or a.stride(-2) == 1) and (b.stride(-1) == 1 or b.stride(-2) == 1):
        return e.lt_mm(a, b, a)
    return torch.bmm(a, b, out_dtype=torch.float32)
'''
assert old in s
s = s.replace(old, new)
old2 = "            dp = torch.bmm(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L), out_dtype=torch.float32)  # [H, j, k]\n"
assert s.count(old2) == 1
s = s.replace(old2, "            dp = _bmm_f32(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L))  # [H, j, k]\n")
open(p, "w").write(s); print("ok")
