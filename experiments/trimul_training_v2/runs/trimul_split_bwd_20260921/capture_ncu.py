import argparse,json,torch
import bench as B
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--only',default='split_xn_s4');a=p.parse_args()
with torch.no_grad():
 data=B.Q.setup(a.length);m=B.Training(data,B.CONFIGS[a.only]);m();m();torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart();m.p7();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 print('PROFILE_DONE',flush=True)
