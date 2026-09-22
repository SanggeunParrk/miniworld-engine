from pathlib import Path
import argparse,gc,hashlib,json,platform,sys,torch
import replace_plan as RP
import replace_core as RC
R=Path(__file__).resolve().parent;PREV=R.parent/'trimul_b1_shared_20260921'
sys.path.insert(0,str(PREV));import training as BASE
OLD=BASE.OLD;Q=OLD.Q;LN=OLD.LN
class Replacement:
 def __init__(self,a,fp32):
  self.a=a;self.d=a['d'];self.fp32=fp32;self.audit=False
  _,k=self.forward();ab,xhat,packed,rs,xn=k
  cfg=json.loads((PREV/('selected-L%d.json'%self.d['n'])).read_text())['config'];cfg['defines']['XHAT_FP32']=int(fp32)
  self.p1=RP.Plan(dict(self.d,x=xn),a['dy'],xhat,rs,**cfg)
  c=dict(OLD.CONFIGS['split_xn_pc1']);c.pop('saved');self.p7=OLD.R.Plan(self.d,a['dy'],a['dl'],a['dr'],a['dg'],xn=xn,**c);self.p7.mask=a['mask']
 def forward(self):
  d=self.d;packed=LN.F.pack(*d['leaves'][1:6]);d['wt'],d['w1']=packed[:5],packed[5]
  ab=LN.I.front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],True,(2,64,8,2,-1,232,2));tri=LN.B.packed_forward(ab[:256],ab[256:],128)
  y,s=RC.output(d,tri,ln=3,stats=2,method=int(self.fp32))
  if self.audit:self.audit_tri=tri
  return y,(ab,s['xnout'],packed,s['ro'],s['xn'])
 def backward(self,k):
  a,d=self.a,self.d;ab,xhat,packed,rs,xn=k;d['wt'],d['w1']=packed[:5],packed[5]
  self.p1.d=dict(d,x=xn);self.p1.bind(a['dy'],xhat,rs)
  dg,dwg,dt,dgo,dbo,dwp=self.p1();dl,dr=OLD.P.B.packed_backward(dt,ab[:256],ab[256:],128)
  self.p7.bind(dl,dr,dg,a['dy'],xn=xn);dx,dwl,dwlg,dwr,dwrg,dgi,dbi=self.p7()
  return dx.reshape_as(d['x']),dwl.t(),dwlg.t(),dwr.t(),dwrg.t(),dwg.t(),dwp,dgi,dbi,dgo,dbo
 def __call__(self):
  y,k=self.forward();return y,self.backward(k)
def clone(x):return x[0].clone(),tuple(t.clone() for t in x[1])
def errors(x,y):return OLD.H.errors(x,y)
def passed(e):return e['forward']['bit_exact'] and all(v['finite'] and v['relative_l2']<=5e-4 for v in e.values())
def mutate(a):
 d=a['d'];d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);d['wp'].mul_(.93);d['go'].mul_(1.07);d['go'][0]=0;d['bo'].add_(.017);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--check-only',action='store_true');args=ap.parse_args();n=args.length
 path=R/(('check-L%d.json' if args.check_only else 'results-L%d.json')%n)
 with torch.no_grad():
  a=Q.setup(n);d=a['d'];models={'baseline':BASE.Training(a),'xhat_bf16':Replacement(a,False),'xhat_fp32':Replacement(a,True)}
  result=dict(L=n,C=128,H=256,dropout=.25,checks={},mutated={},poison={},times={},blocks={},meta=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),raw_tri_in_backward=False,includes=['both directions','mask','dropout','residual','live packing','all eleven gradients'],excludes=['optimizer','RNG generation','compile','CPU dispatch']),production_ready=False)
  ref=clone(models['baseline']())
  for name,m in models.items():
   result['checks'][name]=errors(m(),ref);print('CHECK',name,passed(result['checks'][name]),{k:v['relative_l2'] for k,v in result['checks'][name].items()},flush=True)
   if name!='baseline':
    m.audit=True;y,k=m.forward();original=clone((y,m.backward(k)));tri=m.audit_tri
    assert all(t.data_ptr()!=tri.data_ptr() for t in k if isinstance(t,torch.Tensor))
    tri.fill_(float('nan'));changed=(y,m.backward(k));e=errors(changed,original);assert all(v['bit_exact'] for v in e.values()),(name,e)
    result['poison'][name]=dict(all_gradients_bit_exact_after_tri_nan=True,tri_absent_from_saved_tensors=True,bytes_saved=k[1].numel()*k[1].element_size()+k[3].numel()*4,baseline_tri_bytes=n*n*256*2)
    m.audit=False;del m.audit_tri;del tri
  path.write_text(json.dumps(result,indent=2))
  # Validate live graph input mutation, including gamma_out[0]=0, before timing.
  refs={};graphs={}
  for name,m in models.items():graphs[name],refs[name]=Q.capture_outputs(m)
  mutated_tensors=list({t.data_ptr():t for t in [d['x'],d['leaves'][1],d['leaves'][5],d['wp'],d['go'],d['bo'],a['dy'],d['ds'],d['mask'],a['mask']]}.values());snap=[t.clone() for t in mutated_tensors]
  mutate(a);ref=clone(models['baseline']());result['graph_vs_eager']={}
  for name,m in models.items():
   out=clone(m());result['mutated'][name]=errors(out,ref);graphs[name].replay();torch.cuda.synchronize();e=errors(refs[name],out);assert all(v['bit_exact'] for v in e.values()),(name,e);result['graph_vs_eager'][name]=e
   print('MUTATED',name,passed(result['mutated'][name]),{k:v['relative_l2'] for k,v in result['mutated'][name].items()},flush=True)
  for t,s in zip(mutated_tensors,snap):t.copy_(s)
  del graphs,refs,snap;gc.collect()
  valid=[name for name in models if passed(result['checks'][name]) and passed(result['mutated'][name])];result['valid']=valid;path.write_text(json.dumps(result,indent=2))
  if args.check_only:print('CHECK_ONLY_DONE',flush=True);return
  kept={name:m.forward()[1] for name,m in models.items()}
  # Time failing BF16 path as diagnostic only, never label as a valid speedup.
  result['timed_invalid']=[k for k in models if k not in valid]
  for scope in ('forward','b1','backward','forward_backward'):
   gs={}
   for name,m in models.items():
    if scope=='forward':f=m.forward
    elif scope=='b1':m.backward(kept[name]);f=m.p1
    elif scope=='backward':f=lambda m=m,name=name:m.backward(kept[name])
    else:f=m
    gs[name],_=Q.capture_outputs(f)
   blocks=[Q.paired(gs,iterations=200) for _ in range(3)];result['blocks'][scope]=blocks;result['times'][scope]=Q.pool(blocks);print('TIME',scope,{k:v['median_us'] for k,v in result['times'][scope].items()},flush=True)
   path.write_text(json.dumps(result,indent=2));del gs;gc.collect()
  result['cubins']={k:dict(b1=m.p1.k.unit.cubin_path) for k,m in models.items()};result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'bench.py',R/'replace_plan.py',R/'replace_core.py',*R.glob('*.cu'),*R.glob('*.cuh'),*R.glob('*.inc')]};path.write_text(json.dumps(result,indent=2));print('DONE',n,flush=True)
if __name__=='__main__':main()
