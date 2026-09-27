"""SM clock / power while each backward variant replays for ~4 s: python power.py L [flags]"""
import subprocess, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402

L = int(sys.argv[1]); flags = sys.argv[2:]
ext = TB.build(extra=flags)
mod, x = TB.TA.fixture(L)
dy = torch.randn_like(x)
pk = TB.pack(mod)
bufs = {}
for name, fn in [("3k", lambda: TB.backward(ext, x, dy, pk, bufs)), ("two", lambda: TB.backward_2k(ext, x, dy, pk, bufs)),
                 ("seq", lambda: TB.backward_seq(ext, x, dy, pk, bufs)), ("ring", lambda: TB.backward_ring(ext, x, dy, pk, bufs, ndxp=64))]:
    for _ in range(5):
        fn()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(20):
                fn()
    torch.cuda.synchronize()
    t_end = time.time() + 2.0
    while time.time() < t_end:
        g.replay()
    torch.cuda.synchronize()
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits", "-lms", "100"],
                           stdout=subprocess.PIPE, text=True)
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n = 0; st.record(); t_end = time.time() + 4.0
    while time.time() < t_end:
        g.replay(); n += 1
    en.record(); en.synchronize()
    smi.terminate(); out = smi.communicate()[0].strip().split("\n")
    vals = [list(map(float, l.split(","))) for l in out[5:] if l.count(",") == 2]
    clk = sorted(v[0] for v in vals); pw = sorted(v[1] for v in vals)
    print(f"L{L} {name}: {st.elapsed_time(en) * 1e3 / (20 * n):.1f} us/call  SM clock med {clk[len(clk)//2]:.0f} MHz (min {clk[0]:.0f})  power med {pw[len(pw)//2]:.0f} W  temp {vals[-1][2]:.0f}C", flush=True)
