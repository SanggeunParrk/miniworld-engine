"""per-kernel CUDA time of the sm100 triattn backward (torch.profiler), L from TA_L"""
import os, sys, pathlib, torch
from torch.utils.cpp_extension import load
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
tag = os.environ.get("TA_TAG", "cur")
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / f"triattn_{tag}"; d.mkdir(parents=True, exist_ok=True)
ext = load(f"triattn_{tag}", [os.environ.get("TA_SRC", str(src / "triattn_sm100.cu"))], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
H, D, L = 4, 32, int(os.environ.get("TA_L", "384"))
sc = D ** -0.5
g = torch.Generator(device="cuda").manual_seed(0)
qn, kn, vn, don = (torch.randn(1, L, L, H, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(4))
bias = torch.randn(1, H, L, L, device="cuda", generator=g).to(torch.bfloat16)
out, lse, _ = ext.triattn_fwd(qn, kn, vn, bias, sc, True)
delta = (out.float() * don.float()).sum(-1).permute(0, 1, 3, 2).contiguous()
for _ in range(5): ext.triattn_bwd(qn, kn, vn, bias, don, lse, delta, sc)
torch.cuda.synchronize()
if os.environ.get("TA_NCU"):
    torch.cuda.cudart().cudaProfilerStart(); ext.triattn_bwd(qn, kn, vn, bias, don, lse, delta, sc); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop(); sys.exit(0)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(20): ext.triattn_bwd(qn, kn, vn, bias, don, lse, delta, sc)
    torch.cuda.synchronize()
for e in p.key_averages():
    if e.device_time_total > 0: print(f"{e.key[:60]:60s} {e.device_time_total/e.count:9.1f} us  x{e.count}")
