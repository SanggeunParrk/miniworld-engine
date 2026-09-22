from harness import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--variant',default='baseline');a=p.parse_args()
with torch.no_grad():
 d=fixture();plan=Plan(d,a.variant)
 ref=N._bwd_launch(d['dy'],d['x'],d['xn'],d['rs'],d['c1'],d['gamma'],d['wa'],d['wb'],d['ws']);errors(plan(),ref)
 for _ in range(10):plan()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();plan();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 print('PROFILE_VALID',a.variant,plan.path,plan.k.regs,plan.k.lmem,flush=True)
