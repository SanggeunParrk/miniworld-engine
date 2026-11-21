import torch, os
from miniworld_engine.kernels._nvcc import load_extension, ensure_cuda_home, gencodes, host_flags
ensure_cuda_home()
e = load_extension(name="s90_dsm_v1", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsm.cu")], extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a")], verbose=False)
for cs in (2, 4, 8, 16):
    print(f"cluster {cs:2d}: " + " | ".join(f"{kb} KiB a CTA: {min(e.run(cs, cs * 6, 200, kb) for _ in range(3)):.2f} us" for kb in (4, 16, 64)), flush=True)
