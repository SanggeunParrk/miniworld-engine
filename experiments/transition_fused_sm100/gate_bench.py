import torch, drv
for n, floor in (("gate_ldst", "ld/st + trivial math"), ("gate_dx", "MUFU floor 1024 clk/chunk"), ("gate_pipe", "pipelined, MUFU floor 1024"), ("swiglu_fwd", "MUFU floor 1024 clk/chunk")):
    k = drv.Kernel("build/gate_bench.cubin", n, 0)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    it = 2000
    k((148, 1, 1), (256, 1, 1), o, it); torch.cuda.synchronize()
    print(f"{n}: {o.float().median().item() / it:.0f} clk per chunk ({floor})")
