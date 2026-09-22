from warp_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=384);ap.add_argument('--source',default='front_rows512_r32_224');ap.add_argument('--splits',type=int,default=8);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);p=WarpPlan(a,splits=args.splits,source=args.source)
 for _ in range(20):p()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();p();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
