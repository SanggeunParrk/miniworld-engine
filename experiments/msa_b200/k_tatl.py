import os, sys, pathlib, torch
from torch.utils.cpp_extension import load
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "tatl"; d.mkdir(parents=True, exist_ok=True)
ext = load("tatl", [str(pathlib.Path(__file__).parent / "ta_tl.cu")], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
L, H, D = 384, 4, 32
q, k, v = (torch.randn(1, L, H, L, D, device="cuda").to(torch.bfloat16) for _ in range(3))
bias = torch.randn(1, 1, H, L, L, device="cuda").to(torch.bfloat16)
for _ in range(5): ext.triattn_fwd(q, k, v, bias, D ** -0.5)
torch.cuda.synchronize(); os.environ["TA_TL"] = "1"; ext.triattn_fwd(q, k, v, bias, D ** -0.5); torch.cuda.synchronize()
