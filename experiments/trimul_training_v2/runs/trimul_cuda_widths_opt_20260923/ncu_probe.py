from width_plan import *
from fixture import setup
D=int(os.environ['NCU_WIDTH']);leaves,dy,mask,ds,*_=setup(D,384)
with torch.no_grad():
 m=Training(*leaves,mask,ds,dy);m();torch.cuda.synchronize()
 for _ in range(3):launch(m.ks['b7'],m.params7,m.grid7,D=D,gp=m.gp_native)
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();launch(m.ks['b7'],m.params7,m.grid7,D=D,gp=m.gp_native);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
