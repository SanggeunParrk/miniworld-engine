from pathlib import Path
import argparse,gc,hashlib,json,platform,torch
import save_cost_core as C
import compare_cueq_training as Q
R=C.R
CONFIGS={
 'none':dict(ln=0,stats=0),
 'xn':dict(ln=1,stats=0),
 'xn_in_stats':dict(ln=1,stats=1),
 'xn_out_stats':dict(ln=1,stats=2),
 'xn_both_stats':dict(ln=1,stats=3),
 'xn_stats_out_pg':dict(ln=1,stats=3,output_pg=True),
 'xn_stats_in_pg':dict(ln=1,stats=3,input_pg=True),
 'xn_stats_all_pg':dict(ln=1,stats=3,input_pg=True,output_pg=True),
 'both_ln_stats_all_pg':dict(ln=3,stats=3,input_pg=True,output_pg=True),
}
def rel(x,y):return ((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-20)).item()
def references(d,kept):
 n=d['n'];m=n*n;_,_,xn,xo=C.LN.forward(d,3);xn=xn.reshape(m,128);xo=xo.reshape(m,256)
 x=d['x'].reshape(m,128).float();z=kept[1].permute(1,2,0).reshape(m,256).float()
 out=dict(xn=xn,xnout=xo,mi=x.mean(-1),ri=torch.rsqrt(x.var(-1,unbiased=False)+1e-5),mo=z.mean(-1),ro=torch.rsqrt(z.var(-1,unbiased=False)+1e-5));del x,z
 out['proj']=(xo.float()@d['wp'].float().t()).bfloat16();out['gate']=torch.sigmoid((xn.float()@d['leaves'][5].float().t()).bfloat16().float()).bfloat16()
 pre=torch.empty((1024,m),device='cuda',dtype=torch.bfloat16)
 for offset,idx in ((0,2),(1,1),(512,4),(513,3)):
  pre[offset:offset+512:2].copy_((xn.float()@d['leaves'][idx].float().t()).bfloat16().t())
 out['pre']=pre;return out

def validate(out,ref,refs):
 y,kept,saves=out;assert torch.equal(y,ref),'forward changed';e={'forward':dict(bit_exact=True)}
 for key,t in saves.items():
  if t is None:continue
  v=refs[key];rr=rel(t.reshape_as(v),v);lim=2e-6 if key in ('mi','ri','mo','ro') else 5e-4
  assert bool(torch.isfinite(t).all()) and rr<=lim,(key,rr,lim)
  e[key]=dict(relative_l2=rr,limit=lim,finite=True,bytes=t.numel()*t.element_size())
 return e

def main():
 p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--only');a=p.parse_args();n=a.length
 torch.backends.cuda.matmul.allow_tf32=False
 with torch.no_grad():
  data=Q.setup(n);d=data['d'];del data
  ref,kept=C.LN.S.forward(d);ref=ref.clone();refs=references(d,kept);check={};tuning={};im=om=0
  # Compare real store schedules on shared buffers before full-forward timing.
  for region in ('k1','k3'):
   gs={};outs={}
   if region=='k1':
    ab,pre=C.front(d,True,0);bufs=(ab,pre)
    for meth in (0,1):gs[str(meth)],outs[str(meth)]=Q.capture_outputs(lambda meth=meth:C.front(d,True,meth,bufs));assert torch.equal(outs[str(meth)][0],kept[0]);e=rel(outs[str(meth)][1],refs['pre']);assert e<=5e-4,(region,meth,e)
   else:
    y,saves=C.output(d,kept[1],1,3,True,0);bufs=(y,saves)
    for meth in (0,1):
     gs[str(meth)],outs[str(meth)]=Q.capture_outputs(lambda meth=meth:C.output(d,kept[1],1,3,True,meth,bufs));validate((outs[str(meth)][0],kept,outs[str(meth)][1]),ref,refs)
   if not a.check_only:
    blocks=[Q.paired(gs,iterations=100) for _ in range(2)];ts=Q.pool(blocks);winner=min(ts,key=lambda k:ts[k]['median_us']);tuning[region]=dict(blocks=blocks,times=ts,winner=int(winner));print('TUNING',region,{k:v['median_us'] for k,v in ts.items()},winner,flush=True)
    if region=='k1':im=int(winner)
    else:om=int(winner)
   del gs,outs,bufs;gc.collect()
  configs={k:dict(v,input_method=im,output_method=om) for k,v in CONFIGS.items() if a.only is None or k==a.only};fs={k:(lambda c=c:C.forward(d,**c)) for k,c in configs.items()}
  # Literal prior implementations to quantify any compiler effects in the new controls.
  fs['original_none']=lambda:(*C.LN.S.forward(d),{})
  def oldxn():
   y,k,xi,_=C.LN.forward(d,1);return y,k,dict(xn=xi)
  fs['original_xn']=oldxn
  graphs={};outputs={}
  for k,fn in fs.items():
   out=fn();check[k]=validate(out,ref,refs);print('CHECK',k,max((v.get('relative_l2',0),key) for key,v in check[k].items()),flush=True)
   if not a.check_only:graphs[k],outputs[k]=Q.capture_outputs(fn)
  if a.check_only:torch.cuda.synchronize();print('CHECK_ONLY_DONE',flush=True);return
  blocks=[Q.paired(graphs,iterations=200) for _ in range(3)];times=Q.pool(blocks);print('FULL',{k:v['median_us'] for k,v in times.items()},flush=True)
  result=dict(L=n,C=128,H=256,dropout=.25,configs=configs,tuning=tuning,checks=check,blocks=blocks,times=times,metadata=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,scope='forward only, live weight packing, both directions, mask/dropout25/residual, every selected saved tensor',excludes='backward, RNG generation, optimizer, CPU dispatch, compilation',gate_storage='input: BF16 gate logits; output: BF16 sigmoid(BF16 logit)',tiles='original Anthropic K1 fixed; prior no-save K3 fixed; store schedules tested'))
  # Mutate all data that can feed saved values, and check replay reads live tensors.
  d['x'].mul_(.97);d['gi'].mul_(1.02);d['go'].mul_(.98);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask'])
  ref2,kept2=C.LN.S.forward(d);ref2=ref2.clone();refs2=references(d,kept2);result['mutated_checks']={}
  for name,g in graphs.items():
   g.replay();torch.cuda.synchronize();result['mutated_checks'][name]=validate(outputs[name],ref2,refs2)
  result['retained_bytes']={k:sum(v.get('bytes',0) for v in e.values()) for k,e in check.items()}
  result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'bench.py',R/'save_cost_core.py',R/'save_k1.cu',R/'save_k3.cu',C.LN.R/'ln_save_core.py',C.LN.R/'ln_only_k3.cu',C.I.R/'original_k1.cu')}
  result['cubins']={}
  for label,c in configs.items():
   ks=[C.kernel('k1',pg=c.get('input_pg',False),method=im)[0],C.kernel('k3',c['ln'],c['stats'],c.get('output_pg',False),om)[0]];result['cubins'][label]=[dict(path=k.unit.cubin_path,sha256=hashlib.sha256(Path(k.unit.cubin_path).read_bytes()).hexdigest()) for k in ks]
  (R/('results-L%d.json'%n)).write_text(json.dumps(result,indent=2));print('DONE',n,flush=True)
if __name__=='__main__':main()
