import sys, torch, os
from pathlib import Path
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
from foldjet_engine.kernels._nvcc import load_extension, ensure_cuda_home
ensure_cuda_home()
e = load_extension(name="n1_tmast", sources=[sys.argv[1]], extra_cuda_cflags=[*K._flags(), "-I/NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/n1-triattn/src/foldjet_engine/kernels/triangle_attention/cuda/b200_sources"], verbose=False)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
names = ["2-D contiguous 64x128 (starting)", "4-D transposed 64x1x128 (ending today)", "same, n-fastest order", "transposed 128x1x128 none (256 B runs)", "box 128x8x16 (2 KiB runs)"]
for L in [int(x) for x in sys.argv[2:]]:
    x = torch.randn(L, L, 128, device="cuda", dtype=torch.bfloat16); nbytes = L * L * 256
    for V in range(5):
        ts = []
        for _ in range(12):
            flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); e.tma_st(x, L, V, 148); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
        ts.sort(); t = ts[len(ts) // 2]
        print(f"L{L} V{V} {names[V]:50s} {t:7.1f} us  {nbytes / t / 1e6:6.2f} TB/s", flush=True)
