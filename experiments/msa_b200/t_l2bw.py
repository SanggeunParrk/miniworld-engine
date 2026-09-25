import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "t_l2bw"; build.mkdir(parents=True, exist_ok=True)
ext = load("t_l2bw", [str(here / "t_l2bw.cu")], extra_include_paths=[str(inc)], build_directory=str(build), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"])
for mb in (4, 32, 512):
    buf = torch.randn(mb * 1024 * 1024 // 128, 64, device="cuda").to(torch.bfloat16)
    for nst in (4, 8, 12):
        print(f"buffer {mb} MB, {nst} stages x 16 KiB, 148 CTAs: {ext.run(buf, 4000, nst, 148):.2f} TB/s; 296 CTAs: {ext.run(buf, 2000, nst, 296):.2f} TB/s", flush=True)
