"""Is the failure call-count dependent, stream dependent, or capture dependent?"""
import os, torch
from wide_plan import WidePlan
from bench_front_wide import setup

with torch.no_grad():
    a = setup(384, 256)
    p = WidePlan(a, 256, allow_spills=True)
    for i in range(6):
        p(); torch.cuda.synchronize(); print('eager', i, flush=True)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            p(); print('side-issued', i, flush=True)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); print('side ok', flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        p()
    print('captured', flush=True)
    import os
    N=int(os.environ.get('MW_REPLAYS','600'))
    for i in range(N):
        g.replay()
        if i % 100 == 0:
            torch.cuda.synchronize()
            hit = p.counts.nonzero().flatten().tolist()
            if hit:
                print('STALL', [(h // 4, ['h', 'weights', 'gate', 'xres'][h % 4], int(p.counts[h])) for h in hit[:6]], flush=True)
                raise SystemExit
            print('replay', i, flush=True)
    torch.cuda.synchronize(); print('ours x300 OK', flush=True)
    raise SystemExit
    from bench_front_wide import baseline
    gb = torch.cuda.CUDAGraph()
    sb = torch.cuda.Stream(); sb.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sb):
        for _ in range(2):
            baseline(a)
    torch.cuda.current_stream().wait_stream(sb)
    with torch.cuda.graph(gb, stream=sb):
        baseline(a)
    print('baseline captured', flush=True)
    for i in range(300):
        gb.replay(); g.replay()
        if i % 50 == 0:
            torch.cuda.synchronize(); print('paired', i, flush=True)
    torch.cuda.synchronize(); print('ALL OK', flush=True)
