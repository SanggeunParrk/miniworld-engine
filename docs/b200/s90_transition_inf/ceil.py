# ceil.py: sustained cuBLAS tensor throughput on this B200 after a warm-up, for runs of different length (the power cap lowers the SM clock):
# bf16 / fp16 / tf32 GEMMs of K = D shapes of the Transition expand ([M, 256] x [256, 2048], M = 2^20 .. 2^22) and a square 8192 GEMM; TFLOP/s per
# kernel from graph replays of n kernels back to back (n picked so one replay is ~0.25 / 1 / 4 / 16 ms), after 1 s of the same work.
import time, torch
torch.backends.cuda.matmul.allow_tf32 = True
def run(dt, M, K, N, target_ms):
    a = torch.randn(M, K, device="cuda", dtype=dt); b = torch.randn(K, N, device="cuda", dtype=dt); c = torch.empty(M, N, device="cuda", dtype=dt)
    for _ in range(3): torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(True) for _ in range(2)]
    ev[0].record()
    for _ in range(5): torch.matmul(a, b, out=c)
    ev[1].record(); torch.cuda.synchronize(); t1 = ev[0].elapsed_time(ev[1]) / 5
    n = max(1, int(target_ms / t1))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): torch.matmul(a, b, out=c)
    t_end = time.time() + 1.0
    while time.time() < t_end: g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        ev[0].record(); g.replay(); ev[1].record(); ev[1].synchronize(); ts.append(ev[0].elapsed_time(ev[1]))
    ts.sort(); t = ts[len(ts) // 2] / n
    return 2 * M * K * N / t * 1e-9, ts[len(ts) // 2]
for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("tf32", torch.float32)):
    for (M, K, N) in ((1 << 20, 256, 2048), (1 << 22, 256, 2048), (8192, 8192, 8192)):
        for tgt in (0.25, 1.0, 4.0, 16.0):
            tf, tr = run(dt, M, K, N, tgt)
            print(f"{name} M{M} K{K} N{N} replay {tr:7.2f} ms: {tf:8.1f} TFLOP/s", flush=True)
