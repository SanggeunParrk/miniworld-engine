"""OPM epilogue: the mask count from the bit mask vs a precomputed fp32 norm (same O, Wo, residual)."""
import os, pathlib, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import timeit
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "epibits"; d.mkdir(parents=True, exist_ok=True)
ext = load("epibits", [os.environ.get("EPI_SRC", str(src / "opm_sm100.cu"))], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
bf = torch.bfloat16; N, S, CH, CZ = 384, 1024, 32, 128
mask = torch.rand(S, N, device="cuda") > 0.1
bits = torch.zeros(N, S // 32, dtype=torch.int64, device="cuda")
for w in range(32): bits |= (mask.t().reshape(N, S // 32, 32)[..., w].long() << w)
bits = (bits - ((bits >> 31) & 1) * (1 << 32)).to(torch.int32)
mf = mask.float(); norm = (mf.t() @ mf).clamp(min=1).contiguous()
O = torch.randn(N * CH, N * CH, device="cuda", dtype=bf); wo = (torch.randn(CZ, CH * CH, device="cuda") * 0.03).to(bf)
bias = (torch.randn(CZ, device="cuda") * 0.1).to(bf).float(); res = torch.randn(1, N, N, CZ, device="cuda", dtype=bf)
zb = ext.opm_epilogue(O, bits, wo, bias, N, N, res); zn = ext.opm_epilogue(O, norm, wo, bias, N, N, res)
print("bits vs norm max diff", (zb.float() - zn.float()).abs().max().item())
for name, a in (("bits", bits), ("norm", norm)):
    print(f"{name}: {timeit(lambda: ext.opm_epilogue(O, a, wo, bias, N, N, res)) * 1e3:.1f} us")
A2 = torch.randn(N * CH, S, device="cuda", dtype=bf); BT = torch.randn(N * CH, S, device="cuda", dtype=bf)
Ob = torch.empty_like(O)
mm = lambda: torch.matmul(A2, BT.t(), out=Ob)
t_mm = timeit(mm); t_both = timeit(lambda: (mm(), ext.opm_epilogue(Ob, bits, wo, bias, N, N, res)))
t_cp = timeit(lambda: (mm(), ext.opm_epilogue(O, bits, wo, bias, N, N, res)))
print(f"gemm {t_mm*1e3:.1f} us; gemm + epilogue(its O) {t_both*1e3:.1f} (+{(t_both-t_mm)*1e3:.1f}); gemm + epilogue(other O) +{(t_cp-t_mm)*1e3:.1f}")
