import torch, drv
for thr in (256, 512, 1024):
    for n in ("op_ex2", "op_sig", "op_sigp", "op_fma"):
        k = drv.Kernel("build/sfu_bench.cubin", n, 0)
        o = torch.zeros(148 * thr + 148, device="cuda")
        it = 2000
        k((148, 1, 1), (thr, 1, 1), o, it); torch.cuda.synchronize()
        clk = o[148 * thr:].median().item()
        print(f"threads {thr} {n}: {thr * 16 * it / clk:.1f} elements/clk/SM")
