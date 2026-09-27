"""Our C = 128 fused kernel only, under graph replay."""
import os, torch
from wide_plan import WidePlan
from bench_front_wide import setup
with torch.no_grad():
    a = setup(384, 128)
    p = WidePlan(a, 128)
    p(); torch.cuda.synchronize(); print('eager ok', flush=True)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): p()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): p()
    print('captured', flush=True)
    for i in range(int(os.environ.get('MW_REPLAYS', '600'))):
        g.replay()
        if i % 100 == 0:
            torch.cuda.synchronize(); print('replay', i, flush=True)
    torch.cuda.synchronize(); print('C128 OK', flush=True)
