import argparse,torch
import opt_policy as P
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--variant',choices=['baseline','optimized'],required=True);args=ap.parse_args()
with torch.no_grad():
 a=P.B.BASE.OLD.Q.setup(args.length);m=P.B.Training(a) if args.variant=='baseline' else P.Training(a)
 _,k=m.forward();m.backward(k)
 for _ in range(5):m.p1()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();m.p1();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 print('PROFILE_DONE',args.variant,flush=True)
