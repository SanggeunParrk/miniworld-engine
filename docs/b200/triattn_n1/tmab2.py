import sys, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension, ensure_cuda_home
ensure_cuda_home()
e = load_extension(name="n1_tmab2", sources=[sys.argv[1]], extra_cuda_cflags=[*K._flags(), "-I" + sys.argv[2]], verbose=False)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for L in [int(x) for x in sys.argv[3:]]:
    x = torch.randn(L, L, 128, device="cuda", dtype=torch.bfloat16); o = torch.randn(L * L, 128, device="cuda", dtype=torch.bfloat16)
    for V, nm in ((0, "today: x 64x1x128 boxes + o 2 boxes of 128 rows"), (1, "8n x 16s tile: x 2 KiB-run box + o 16 boxes of 16 rows")):
        ts = []
        for _ in range(14):
            flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); e.tma_bw2(x, o, L, V, 148); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
        ts.sort(); print(f"L{L} V{V} {nm:56s} {ts[len(ts)//2]:7.1f} us ({L*L*512/ts[len(ts)//2]/1e6:.2f} TB/s)", flush=True)
