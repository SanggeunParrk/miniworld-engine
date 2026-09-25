import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "pftl"; d.mkdir(parents=True, exist_ok=True)
ext = load("pftl", [str(here / "pwa_tl.cu")], extra_include_paths=[str(inc)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
bf = torch.bfloat16; N, S, H, C, D = 384, 1024, 8, 32, 64
m = torch.randn(S, N, D, device="cuda", dtype=bf); y = torch.randn(S, N, D, device="cuda", dtype=bf)
w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf); v = torch.randn(H, N, S * C, device="cuda", dtype=bf)
wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wo = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf)
for _ in range(3): ext.pwa_fwd(w, v, y, wg, wo, m, False, None, 1.0)
torch.cuda.synchronize()
os.environ["PF_TL"] = "1"
ext.pwa_fwd(w, v, y, wg, wo, m, False, None, 1.0); torch.cuda.synchronize()
