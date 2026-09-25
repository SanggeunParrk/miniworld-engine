import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "prtl"; d.mkdir(parents=True, exist_ok=True)
ext = load("prtl", [str(here / "opm_tl.cu")], extra_include_paths=[str(inc)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
bf = torch.bfloat16; N, S, CH, CM = 384, 1024, 32, 64
m = torch.randn(S, N, CM, device="cuda", dtype=bf); mask = torch.rand(S, N, device="cuda") > 0.1
lnw = 1 + 0.1 * torch.randn(CM, device="cuda"); lnb = 0.1 * torch.randn(CM, device="cuda")
wa = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf); wb = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf)
for _ in range(5): ext.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, False, False)
torch.cuda.synchronize(); os.environ["PR_TL"] = "1"
ext.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, False, False); torch.cuda.synchronize()
