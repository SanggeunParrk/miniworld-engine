import torch, drv
for rows in (4096, 65536):                      # 1 MB (L2) and 16 MB
    t = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    m = drv.TensorMap(t, [128, rows], 256, [64, 64])
    for n in ("l2s2", "l2s4", "l2s6"):
        k = drv.Kernel("build/l2_bench.cubin", n, 6 * 32768 + 1024)
        o = torch.zeros(148, dtype=torch.int64, device="cuda")
        it = 400
        for ctas in (148, 74):
            k((ctas, 1, 1), (32, 1, 1), m, o, it, rows); torch.cuda.synchronize()
            k((ctas, 1, 1), (32, 1, 1), m, o, it, rows); torch.cuda.synchronize()
            clk = o[:ctas].float().median().item()
            print(f"src {rows*256>>20} MB {n} ctas {ctas}: {32768*it/clk:.1f} B/clk/SM, aggregate {32768*it*ctas/clk*1.965e9/1e12:.2f} TB/s, per-tile {clk/it:.0f} clk")
