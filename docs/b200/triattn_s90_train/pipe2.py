import os
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension, ensure_cuda_home
ensure_cuda_home()
e = load_extension(name="s90_pipe_v1", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipe.cu")], extra_cuda_cflags=K._flags(), verbose=False)
for v, n in ((4, "MUFU only"), (31, "real mix"), (27, "mix without MUFU")):
    print(n, " ".join(f"nw{nw}: {min(e.run(v, nw) for _ in range(5)):.2f}" for nw in (4, 8, 12, 16, 24)), "clk/warp-pair/SMSP", flush=True)
