"""Kernel breakdown of the default TriAttn eval on the 5090 (E4 rows): python triattn_profile.py D,L,H ..."""
import sys, torch
import miniworld_engine
from miniworld_engine.modules import TriangleAttention
from torch.profiler import profile, ProfilerActivity
for c in sys.argv[1:]:
    d, L, h = map(int, c.split(","))
    m = TriangleAttention(d, n_head=h, master_weights=True, backend="cuda").cuda().eval()
    x = torch.randn(1, L, L, d, device="cuda", dtype=torch.bfloat16); mk = torch.ones(1, L, device="cuda", dtype=torch.bool)
    def f():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return m(x, mk)
    for _ in range(3): f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3): f()
        torch.cuda.synchronize()
    tot = sum(e.device_time_total for e in p.key_averages()) / 3
    print(f"== d{d} L{L} h{h}: total {tot:.0f} us / call")
    for e in sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:8]:
        print(f"  {e.device_time_total/3:9.0f} us  {e.count//3}x  {e.key[:100]}")
