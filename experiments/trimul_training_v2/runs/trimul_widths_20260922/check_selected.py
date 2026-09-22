from selected import *
import sys,os
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
D=int(sys.argv[1]);H=2*D;results=[]
with torch.no_grad():
 for n in (384,768):
  torch.manual_seed(n+D);x=torch.randn(1,n,n,D,device='cuda',dtype=torch.bfloat16);ws=[(torch.randn(s,device='cuda')/s[-1]**.5).bfloat16() for s in [(H,D)]*4+[(D,D),(D,H)]];gi=torch.ones(D,device='cuda');bi=torch.zeros_like(gi);go=torch.ones(H,device='cuda');bo=torch.zeros_like(go);mask=(torch.rand(1,n,n,device='cuda')>.15).float();args=(x,*ws,gi,bi,go,bo);p=Inference(*args,mask)
  y=p();ref=B.bidirectional_trimul_triton(*args,1e-5,1e-5,D,mask=mask);err=float((y.float()-ref.float()).norm()/ref.float().norm());assert err<.005,err
  st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(st):p()
  torch.cuda.current_stream().wait_stream(st);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=st):out=p()
  x.neg_();ws[0].neg_();e=p().clone();g.replay();torch.cuda.synchronize();assert torch.equal(e,out)
  results.append(dict(D=D,L=n,relative_l2=err,mutation_exact=True,job=os.environ.get('SLURM_JOB_ID')));print('SELECTED_PASS',D,n,err,flush=True)
(R/f'selected-check-D{D}.json').write_text(json.dumps(results,indent=2))
