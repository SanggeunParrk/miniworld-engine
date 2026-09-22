from native import *
import sys,torch.nn.functional as F
separate=len(sys.argv)>2 and sys.argv[2]=="separate"
D=int(sys.argv[1]);p=R/f'native-D{D}.json';save_xn=True
if D==64 and (R/'native-D64-nosave.json').exists():p=R/'native-D64-nosave.json';save_xn=False
if separate:p=R/f'native-D{D}-separate.json';save_xn=False
cfg=json.loads(p.read_text());checks=[]
with torch.no_grad():
 for n in (384,768):
  row=cfg['results'][str(n)];torch.manual_seed(n+D);x=torch.randn(n,n,D,device='cuda',dtype=torch.bfloat16);H=2*D
  wl,wlg,wr,wrg,wg,wp=[(torch.randn(s,device='cuda')/s[-1]**.5).bfloat16() for s in [(H,D)]*4+[(D,D),(D,H)]]
  gi=torch.ones(D,device='cuda');bi=torch.zeros_like(gi);go=torch.ones(H,device='cuda');bo=torch.zeros_like(go);mask=torch.ones(n,n,device='cuda')
  w1=torch.stack((torch.cat((wlg,wrg)).reshape(-1,32,D),torch.cat((wl,wr)).reshape(-1,32,D)),1).reshape(8*D,D)
  refxn=F.layer_norm(x.float(),(D,),gi,bi,1e-5).bfloat16();front=Front(refxn if separate else x,w1,mask,gi,bi,row['selected_k1'],emit_xn=save_xn,normalize=not separate);ab,xn=front();torch.cuda.synchronize();refxn=F.layer_norm(x.float(),(D,),gi,bi,1e-5).bfloat16()
  left=torch.sigmoid(F.linear(refxn,wlg))*F.linear(refxn,wl);right=torch.sigmoid(F.linear(refxn,wrg))*F.linear(refxn,wr);refab=torch.cat((left,right),-1).permute(2,0,1).contiguous();error=float((ab.float()-refab.float()).norm()/refab.float().norm());assert error<.01,error
  tri=B.packed_forward(ab[:H],ab[H:],D)
  if row['k3_route']=='Anthropic':
   oc=min([z for z in row['k3'] if 'us' in z],key=lambda z:z['us'])['config'];out=Output(tri,x,wp,wg,gi,bi,go,bo,oc);y=out();torch.cuda.synchronize();norm=F.layer_norm(tri.permute(1,2,0).float(),(H,),go,bo,1e-5).bfloat16();ref=x+F.linear(norm,wp)*torch.sigmoid(F.linear(refxn,wg));err=float((y.float()-ref.float()).norm()/ref.float().norm());assert err<.005,err
  mask.zero_();front();torch.cuda.synchronize();assert torch.count_nonzero(ab)==0
  checks.append(dict(L=n,relative_l2=error,cubin=front.path,emit_xn=save_xn));print('NATIVE_PROBE_PASS',D,n,flush=True)
(R/f'native-probe-D{D}{"-separate" if separate else ""}.json').write_text(json.dumps(checks,indent=2))
