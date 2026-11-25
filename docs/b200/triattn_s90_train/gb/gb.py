import torch, os
from foldjet_engine.kernels._nvcc import load_extension, ensure_cuda_home, gencodes, host_flags
ensure_cuda_home()
e = load_extension(name="s90_gb_v1", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "gb.cu")], extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a")], verbose=False)
for nb in (32, 74, 128, 148):
    print(f"CTAs {nb}: barrier cost (us): " + " ".join(f"work{w}: {min(e.run(nb, 2000, w)[0] for _ in range(3)):.2f}" for w in (0, 100)), flush=True)
