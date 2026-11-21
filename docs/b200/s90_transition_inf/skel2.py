# skel.py CS DIM: graph time (flush + replay, like floor.py) of the one-launch data-movement skeleton against 1 trivial kernel; clock64 stamps per phase
import sys, os, statistics, torch
from torch.utils.cpp_extension import load
cs, dim, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
os.makedirs("/NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/s90-transition-inf/out/skelbuild", exist_ok=True)
m = load(f"skel2_{cs}_{dim}_{mode}_{os.environ.get("XFULL", 0)}", sources=["/NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/s90-transition-inf/b200cfg/skel2.cu"], extra_cuda_cflags=[f"-DCS={cs}", f"-DDDIM={dim}", f"-DMODE={mode}", f"-DXFULL={os.environ.get('XFULL', 0)}", "-gencode=arch=compute_100a,code=sm_100a", "-O3"], build_directory="/NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/s90-transition-inf/out/skelbuild", verbose=False)
H = 2 * dim
w = torch.zeros(3 * dim * H * 2, dtype=torch.uint8, device="cuda"); x = torch.zeros(128 * dim * 2, dtype=torch.uint8, device="cuda")
y = torch.zeros(128 * dim * 2, dtype=torch.uint8, device="cuda"); st = torch.zeros(cs * 8, dtype=torch.int64, device="cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda"); z = torch.zeros(128, device="cuda")
m.run(w, x, y, st, 1); torch.cuda.synchronize()
s = st.view(cs, 8).cpu().double()
print("clock64 (cycles, median over CTAs): x in %.0f, xn gathered %.0f, weights in %.0f, h gathered %.0f, end %.0f" % tuple(s[:, i].median().item() for i in range(5)), flush=True)
def med(g):
    ts = []
    for _ in range(200):
        flush.zero_(); a, e = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); e.record(); e.synchronize(); ts.append(a.elapsed_time(e) * 1000)
    return statistics.median(ts)
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1): z.add_(1.0)
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2): m.run(w, x, y, st, 0)
print(f"CS {cs} D {dim}: 1 trivial kernel {med(g1):.1f} us; skeleton one-launch {med(g2):.1f} us", flush=True)
