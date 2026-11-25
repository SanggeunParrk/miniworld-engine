import torch, os
from foldjet_engine.kernels._nvcc import load_extension, ensure_cuda_home, gencodes, host_flags
ensure_cuda_home()
e = load_extension(name="s90_cb_v1", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "cb.cu")], extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a")], verbose=False)
for cs in (1, 2, 4, 8, 16):
    for ctas in (cs, 16 * 6 if cs <= 16 else cs):
        ctas = (ctas // cs) * cs
        print(f"cluster {cs:2d}, grid {ctas:3d} CTAs: release/acquire {min(e.run(cs, ctas, 2000, 0) for _ in range(3)):.3f} us | relaxed arrive + wait {min(e.run(cs, ctas, 2000, 1) for _ in range(3)):.3f} us | default {min(e.run(cs, ctas, 2000, 2) for _ in range(3)):.3f} us a barrier", flush=True)
