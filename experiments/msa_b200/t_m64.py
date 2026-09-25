import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "t_m64"; build.mkdir(parents=True, exist_ok=True)
ext = load("t_m64", [str(here / "t_m64.cu")], extra_include_paths=[str(inc)], build_directory=str(build), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"])
o = ext.run(16)
col0 = o[:, 0].cpu()
print("lane -> D row+1 (column 0), 0 = untouched:")
for w in range(4): print(f"  lanes {w*32:3d}-{w*32+31:3d}:", col0[w*32:(w+1)*32].int().tolist())
print("columns 0..7 of lane 0 and lane 16:", o[0].tolist(), o[16].tolist())
