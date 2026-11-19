# wstream.py: us a tile of the weight stream of a fused D 512 pair kernel (1.5 MB a CTA per tile = 96 stages of 16 KB), with the stage consumed at MMA pace (spin clocks a stage)
import sys, os, torch
from torch.utils.cpp_extension import load
m = load("wstream", sources=["/NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/s90-transition-inf/b200cfg/wstream.cu"], extra_cuda_cflags=["-gencode=arch=compute_100a,code=sm_100a", "-O3"], build_directory="/NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/s90-transition-inf/out/skelbuild", verbose=False)
w = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda"); out = torch.zeros(148, dtype=torch.int64, device="cuda")
per_tile = 96  # 1.5 MB / 16 KB
for spin in (0, 200, 400, 800):
    for nb in (148,):
        m.run(w, 4, per_tile, spin, out, nb); torch.cuda.synchronize()
        tiles = 24
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); m.run(w, tiles, per_tile, spin, out, nb); e1.record(); e1.synchronize()
        ms = e0.elapsed_time(e1)
        clk = out.double().median().item() / tiles
        print(f"spin {spin:4d} clk a stage, {nb} CTAs: {ms * 1000 / tiles:7.2f} us a tile (event), {clk:9.0f} clk a tile", flush=True)
