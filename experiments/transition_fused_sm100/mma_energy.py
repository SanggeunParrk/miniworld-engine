"""Sustained energy per FLOP of tcgen05.mma by operand / accumulator type (148 CTAs, M128 N256, 32 B of K per instruction)."""
import torch, drv
from energy import measure
it = 4000
for n, kk in (("e_bf16_f32", 16), ("e_f16_f32", 16), ("e_f16_f16", 16), ("e_e4m3_f32", 32), ("e_e4m3_f16", 32)):
    k = drv.Kernel("build/mma_energy.cubin", n, 65536 + 1024)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    fn = lambda: k((148, 1, 1), (128, 1, 1), o, it)
    flop = 148 * it * 8 * 2 * 128 * 256 * kk
    r = measure(fn, flop, secs=3.0, reps=5)
    fn(); torch.cuda.synchronize()
    clk = o.float().median().item()
    print(f"{n:12s}: {r['TFLOPS']:6.0f} TFLOPS {r['W']:4.0f} W {r['pJ_per_flop']:.3f} pJ/FLOP | {clk/(it*8):.1f} clk/instr (ideal {128*256*kk*2/8192/ (2 if kk==32 else 1):.0f})", flush=True)
