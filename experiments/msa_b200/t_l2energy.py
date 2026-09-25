"""Energy per byte: L2 -> SMEM TMA streaming (L2-resident buffer) vs an HBM read+write stream, against idle power."""
import os, pathlib, sys, time, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import energy_j, sustained
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "t_l2bw"; build.mkdir(parents=True, exist_ok=True)
ext = load("t_l2bw", [str(here / "t_l2bw.cu")], extra_include_paths=[str(inc)], build_directory=str(build), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"])
torch.cuda.synchronize(); e0 = energy_j(); time.sleep(3.0); idle = (energy_j() - e0) / 3.0
print(f"idle {idle:.0f} W")
buf = torch.randn(32 * 1024 * 1024 // 128, 64, device="cuda").to(torch.bfloat16)
for _ in range(3): ext.run(buf, 4000, 12, 148)
torch.cuda.synchronize(); n = 0; e0 = energy_j(); t0 = time.time()
while time.time() - t0 < 3.0:
    ext.run(buf, 4000, 12, 148); n += 1
torch.cuda.synchronize(); dt = time.time() - t0; de = energy_j() - e0
byts = n * 148 * 4000 * 16384 * 3                          # each run() launches the kernel 3 times
W = de / dt; bw = byts / dt
print(f"L2->SMEM: {bw/1e12:.2f} TB/s at {W:.0f} W -> {(W - idle) / bw * 1e12:.1f} pJ/B dynamic ({W / bw * 1e12:.1f} total)")
x = torch.randn(512 * 1024 * 1024, device="cuda", dtype=torch.bfloat16); y = torch.empty_like(x)
r = sustained(lambda: torch.mul(x, 1.0, out=y), secs=3.0); bw = 2 * x.numel() * 2 / (r["ms"] * 1e-3)
print(f"HBM stream: {bw/1e12:.2f} TB/s at {r['W']:.0f} W -> {(r['W'] - idle) / bw * 1e12:.1f} pJ/B dynamic ({r['W'] / bw * 1e12:.1f} total)")
