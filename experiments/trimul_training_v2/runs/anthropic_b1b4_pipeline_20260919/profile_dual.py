from dual import *
with torch.no_grad():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 384
 d,dy,s=data(n);p=Unified(d,dy,s,132,2);p();torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart();p();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
