from width_plan import *
from fixture import setup
D=int(sys.argv[1]);N=384
leaves,dy,mask,ds,*_=setup(D,N)
with torch.no_grad():
 m=Training(*leaves,mask,ds,dy)
 for _ in range(3):m()
 torch.cuda.synchronize();t=m.floats[-1].cpu().tolist();print('STAGES',D,m.grid,[t[i+1]-t[i] for i in range(7)],[t[i+1]-t[i] for i in range(8,13)],flush=True)
