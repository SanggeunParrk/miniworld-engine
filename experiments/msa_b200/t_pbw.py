"""pair_bwd (sm100) standalone: sustained time, energy, power."""
import os, pathlib, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "tctr"; d.mkdir(parents=True, exist_ok=True)
ext = load("tctr", [os.environ.get("PBW_SRC", str(src / "pwa_sm100.cu"))], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
bf = torch.bfloat16; N, H = 384, 8
z = torch.randn(N, N, 128, device="cuda", dtype=bf); w16 = torch.softmax(torch.randn(H, N, N, device="cuda"), -1).to(bf)
dw = torch.randn(H, N, N, device="cuda") * 1e-3; lnw = 1 + 0.1 * torch.randn(128, device="cuda"); lnb = 0.1 * torch.randn(128, device="cuda")
wb = (torch.randn(H, 128, device="cuda") * 0.1).to(bf)
r = sustained(lambda: ext.pair_bwd(z, w16, dw, lnw, lnb, 1e-5, wb), secs=2.0)
print(f"pair_bwd: {r['ms']*1e3:.1f} us  {r['J']*1e3:.2f} mJ  {r['W']:.0f} W")
