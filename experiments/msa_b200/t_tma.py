import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "t_tma"; build.mkdir(parents=True, exist_ok=True)
ext = load("t_tma", [str(here / "t_tma.cu")], extra_include_paths=[str(inc)], build_directory=str(build),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"])
N = 64
# O[(i,c),(j,e)] = code(i,c,j,e) as int16 packed in bf16 slots
i, c, j, e = torch.meshgrid(*[torch.arange(n) for n in (N, 32, N, 32)], indexing="ij")
code = (i * 1000000 + c * 10000 + j * 100 + e)
O16 = (((c * 32 + e) + (j % 8) * 1024)).to(torch.int16).reshape(N * 32, N * 32).cuda()   # enough to identify (c, e, j%8)
for sw in (0, 1):
    raw = ext.run(O16.view(torch.bfloat16), N, N, sw).view(torch.int16).cpu().view(2, 128, 32)
    exp = torch.zeros(2, 128, 32, dtype=torch.int16)
    for cc in range(2):
      for p in range(128):
        il, jl = divmod(p, 32)
        for ee in range(32):
            col = ((ee // 8) ^ ((p >> 1) % 4)) * 8 + ee % 8 if sw else ee
            exp[cc, p, col] = cc * 32 + ee + (jl % 8) * 1024
    print("swizzle", sw, "match", bool((raw == exp).all()), "row0", raw[0, 0, :12].tolist(), "row2", raw[0, 2, :12].tolist())
