from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,platform,sys,torch
import role_plan as R
import saved_plans as SP
import compare_cueq_training as Q
P=R.P
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'trimul_ln_only_save_20260921'));import ln_save_core as LN
spec=importlib.util.spec_from_file_location('prior_bench',R.PREV/'bench.py');OLD=importlib.util.module_from_spec(spec);spec.loader.exec_module(OLD)
H=OLD.H
CONFIGS={
 'split_xn_s4':dict(saved=True,split=True,splits=16,pc=1,prod=32,cons=224,slices=4),
 'split_xn_s8':dict(saved=True,split=True,splits=16,pc=1,prod=32,cons=224,slices=8),
 'split_no_save':dict(saved=False,split=True,splits=8),
 'fused_xn':dict(saved=True,split=False,splits=3),
 'split_xn_pc2':dict(saved=True,split=True,splits=8),
 'split_xn_pc1':dict(saved=True,split=True,splits=16,pc=1,prod=32,cons=224),
}
class Training:
 def __init__(self,a,config):
  self.a=a;self.d=a['d'];c=dict(config);self.saved=c.pop('saved');self.config=config
  y,kept=self.forward();xn=kept[-1]
  cfg1=json.loads((R.PREV/('selected-L%d.json'%self.d['n'])).read_text())['b1']
  if self.saved:
   cfg1['defines']['USE_SAVED_XN']=1;self.p1=SP.B1(dict(self.d,x=xn),a['dy'],kept[1],**cfg1)
  else:self.p1=P.B1(self.d,a['dy'],kept[1],**cfg1)
  self.p7=R.Plan(self.d,a['dy'],a['dl'],a['dr'],a['dg'],xn=xn,**c);self.p7.mask=a['mask']
 def forward(self):
  if self.saved:
   y,kept,xn,_=LN.forward(self.d,1);return y,(*kept,xn)
  y,kept=P.S.forward(self.d);return y,(*kept,None)
 def backward(self,kept):
  a,d=self.a,self.d;ab,tri,packed,xn=kept;d['wt'],d['w1']=packed[:5],packed[5]
  self.p1.d=dict(d,x=xn) if self.saved else d;self.p1.bind(a['dy'],tri)
  dg,dwg,dt,dgo,dbo,dwp=self.p1();dl,dr=P.B.packed_backward(dt,ab[:256],ab[256:],128)
  self.p7.bind(dl,dr,dg,a['dy'],xn=xn);dx,dwl,dwlg,dwr,dwrg,dgi,dbi=self.p7()
  return(dx.reshape_as(d['x']),dwl.t(),dwlg.t(),dwr.t(),dwrg.t(),dwg.t(),dwp,dgi,dbi,dgo,dbo)
 def __call__(self):
  y,k=self.forward();return y,self.backward(k)

def check(out,ref):
 e=H.errors(out,ref);assert e['forward']['bit_exact'];assert all(v['finite'] and v['relative_l2']<=5e-4 for v in e.values()),e;return e

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--check-only',action='store_true');ap.add_argument('--only',choices=tuple(CONFIGS));args=ap.parse_args();n=args.length
 with torch.no_grad():
  a=Q.setup(n);d=a['d'];k3=json.loads((P.S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner'];cfg=json.loads((R.PREV/('selected-L%d.json'%n)).read_text());baseline=OLD.Training(a,k3,cfg['b1'],cfg['b7']);saved=P.S.A.Training(a,k3)
  y,ss=saved.forward_saved();ref=(y,P.S.C.backward(d,ss,a['dy']))
  configs={k:v for k,v in CONFIGS.items() if (args.only is None and k not in ('split_xn_s4','split_xn_s8')) or k==args.only};models={'baseline':baseline,**{k:Training(a,c) for k,c in configs.items()}};checks={}
  for name,m in models.items():checks[name]=check(m(),ref)
  print('CHECKS',{k:max((v['relative_l2'],kk) for kk,v in e.items()) for k,e in checks.items()},flush=True)
  if args.check_only:
   for name,m in models.items():check(m(),ref)
   torch.cuda.synchronize();print('CHECK_ONLY_DONE',flush=True);return
  result=dict(L=n,C=128,H=256,dropout=.25,configs=configs,checks=checks,times={},blocks={},traces={},metadata=dict(hostname=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,includes=['live weight packing','input LN store when selected','all launches and parameter reductions','both directions','mask/dropout25/residual','all eleven gradients'],excludes=['RNG generation','optimizer','CPU dispatch','compilation']))
  kept={k:m.forward()[1] for k,m in models.items()}
  scopes={'forward':{k:m.forward for k,m in models.items()},'backward':{k:(lambda k=k,m=m:m.backward(kept[k])) for k,m in models.items()},'forward_backward':models}
  for scope,fs in scopes.items():
   gs={};outs={}
   for name,f in fs.items():
    gs[name],outs[name]=Q.capture_outputs(f);gs[name].replay();torch.cuda.synchronize()
    if scope=='forward_backward':check(outs[name],ref)
   blocks=[Q.paired(gs,iterations=200) for _ in range(3)];result['blocks'][scope]=blocks;result['times'][scope]=Q.pool(blocks)
   print('RESULT',scope,{k:v['median_us'] for k,v in result['times'][scope].items()},flush=True)
   result['traces'][scope]={k:OLD.trace(g,ROOT/('trace-%s-%s-L%d.json'%(scope,k,n))) for k,g in gs.items()}
   if scope=='forward_backward':
    originals={k:t.clone() for k,t in {'x':d['x'],'wl':d['leaves'][1],'wg':d['leaves'][5],'dy':a['dy'],'ds':d['ds'],'mask':d['mask']}.items()}
    d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
    y2,ss2=saved.forward_saved();ref2=(y2,P.S.C.backward(d,ss2,a['dy']));result['mutated_reference']={};result['graph_vs_eager']={};result['mutated_vs_baseline']={}
    baseout=baseline();baseout=(baseout[0].clone(),tuple(x.clone() for x in baseout[1]))
    for name,m in models.items():
     eager=m();eager=(eager[0].clone(),tuple(x.clone() for x in eager[1]));result['mutated_vs_baseline'][name]=H.errors(eager,baseout)
     gs[name].replay();torch.cuda.synchronize();e=H.errors(outs[name],eager);assert all(x['bit_exact'] for x in e.values()),(name,e);result['graph_vs_eager'][name]=e
     e=H.errors(outs[name],ref2);result['mutated_reference'][name]=e
     print('MUTANT',name,max((x['relative_l2'],k) for k,x in e.items()),flush=True)
    result['mutated_vs_baseline_pass']={k:all(x['finite'] and x['relative_l2']<=5e-4 for x in v.values()) for k,v in result['mutated_vs_baseline'].items()}
    result['mutated_reference_pass']={k:all(x['finite'] and x['relative_l2']<=5e-4 for x in v.values()) for k,v in result['mutated_reference'].items()}
   (ROOT/('results-L%d.json'%n)).write_text(json.dumps(result,indent=2));del gs,outs;gc.collect()
  result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'bench.py',ROOT/'role_plan.py',ROOT/'saved_plans.py',*ROOT.glob('*.cu'),*ROOT.glob('*.cuh'),*ROOT.glob('*.inc'),LN.R/'ln_save_core.py',LN.R/'ln_only_k3.cu']}
  result['cubins']={}
  for name,m in models.items():
   units=[m.p1.k]+([u[0] for u in m.p7.units] if hasattr(m.p7,'units') else [m.p7.k]);result['cubins'][name]=[dict(path=k.unit.cubin_path,sha256=hashlib.sha256(Path(k.unit.cubin_path).read_bytes()).hexdigest()) for k in units]
  (ROOT/('results-L%d.json'%n)).write_text(json.dumps(result,indent=2));print('DONE',n,flush=True)
if __name__=='__main__':main()
