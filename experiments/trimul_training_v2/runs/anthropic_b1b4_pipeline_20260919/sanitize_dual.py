from dual import *
with torch.no_grad():
 d,dy,s=data(72);p=Unified(d,dy,s,4,2);ref=baseline(d,dy,s)
 for _ in range(3):p()
 torch.cuda.synchronize();er=[rel(x,y) for x,y in zip(p.outputs,ref)];assert max(er)<.001,er;assert torch.count_nonzero(p.workspace[-1])==0
 print('PASS multi-tile persistent prefetch',er,flush=True)
