from unified import *
with torch.no_grad():
 d,dy,s=data(384);p=Unified(d,dy,s,132,int(sys.argv[1]));p();torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart()
 p();torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStop()
