import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "t_boxbw"; build.mkdir(parents=True, exist_ok=True)
ext = load("t_boxbw", [str(here / "t_boxbw.cu")], extra_include_paths=[str(inc)], build_directory=str(build), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"])
v = torch.randn(8 * 384, 1024 * 32, device="cuda").to(torch.bfloat16)     # the PWA v tensor [H*N][S*C] (201 MB)
for mode, name in ((0, "box 64x64 rows, 128B rows, SW128"), (1, "PWA v box (32 c, 64 j, 3 s), 64B rows"), (2, "box 64 cols x 192 rows, 128B rows")):
    print(f"{name:40s}: {ext.run(v, mode):.2f} TB/s", flush=True)
