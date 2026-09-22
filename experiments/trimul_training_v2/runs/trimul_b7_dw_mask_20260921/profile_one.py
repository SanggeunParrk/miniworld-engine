import argparse,torch
from gate_policy import Training,B
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--profile',action='store_true');n=ap.parse_args().length
with torch.no_grad():
 a=B.BASE.OLD.Q.setup(n);m=Training(a);m()
 for _ in range(5):m.p7()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();m.p7();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
