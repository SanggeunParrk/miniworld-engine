from width_autograd import *
from fixture import setup
leaves,dy,mask,ds,*_=setup(128,768)
with torch.no_grad():
 m=D128(leaves,mask,ds)
 for _ in range(2):m.forward();m.backward(dy)
 torch.cuda.synchronize();print('SANITIZE 128 768 DONE',flush=True)
