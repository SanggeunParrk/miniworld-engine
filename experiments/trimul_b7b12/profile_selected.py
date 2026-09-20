"""One warm selected B7-B12 launch for Nsight Compute's profiler range."""
import argparse

from ring_plan import RingPlan
from warp_plan import WarpPlan, setup, torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--length', type=int, choices=(384, 768), required=True)
args = parser.parse_args()

with torch.no_grad():
    inputs = setup(args.length)
    if args.length == 384:
        plan = WarpPlan(inputs, count=264, splits=13,
                        source='front_prefetch_lnpair_storepipe')
    else:
        plan = RingPlan(inputs, count=264, splits=20, source='front_ring96_cache3')
    for _ in range(20):
        plan()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    plan()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
