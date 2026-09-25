"""pwa_ctr (the forward contraction alone) vs torch.bmm; sustained time and energy."""
import os, pathlib, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "tctr"; d.mkdir(parents=True, exist_ok=True)
ext = load("tctr", [str(src / "pwa_sm100.cu")], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
bf = torch.bfloat16; N, S, H, C = 384, 1024, 8, 32
w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf); v = torch.randn(H, N, S * C, device="cuda", dtype=bf)
o = ext.pwa_ctr(w, v)
ref = torch.bmm(w.float(), v.float()).view(H, N, S, C).permute(2, 1, 0, 3).reshape(S, N, H * C)
print("rel err", ((o.float() - ref).norm() / ref.norm()).item())
for name, fn in (("pwa_ctr", lambda: ext.pwa_ctr(w, v)), ("torch.bmm", lambda: torch.bmm(w, v))):
    r = sustained(fn, secs=2.0); print(f"{name}: {r['ms']*1e3:.1f} us  {r['J']*1e3:.1f} mJ  {r['W']:.0f} W")
