import argparse,torch
import bench as B
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--variant',choices=['prior_xhat','xhat_fp32'],required=True);args=ap.parse_args()
with torch.no_grad():
 a=B.Q.setup(args.length);m=B.OLDN.Replacement(a,1) if args.variant=='prior_xhat' else B.Replacement(a,1)
 y,k=m.forward();m.backward(k)
 for _ in range(5):m.p1()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();m.p1();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 print('PROFILE_DONE',args.variant,flush=True)
