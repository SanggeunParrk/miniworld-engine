from front_plan import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=384);ap.add_argument('--source',default='front_selected');ap.add_argument('--splits',type=int,default=15);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);p=Plan(a,splits=args.splits,source=args.source)
 for _ in range(20):p()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();p();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
