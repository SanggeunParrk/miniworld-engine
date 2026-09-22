from pathlib import Path
import argparse,gc,hashlib,json,platform,sys,torch
from gate_policy import Training,B,BASE
R=Path(__file__).resolve().parent
OLD=BASE.OLD;Q=OLD.Q;LN=OLD.LN
def clone(x):return x[0].clone(),tuple(t.clone() for t in x[1])
def errors(x,y):return OLD.H.errors(x,y)
def passed(e):return e['forward']['bit_exact'] and all(v['finite'] and v['relative_l2']<=5e-4 for v in e.values())
def mutate(a):
 d=a['d'];d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);d['wp'].mul_(.93);d['go'].mul_(1.07);d['go'][0]=0;d['bo'].add_(.017);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--check-only',action='store_true');args=ap.parse_args();n=args.length
 path=R/(('check-L%d.json' if args.check_only else 'results-L%d.json')%n)
 with torch.no_grad():
  a=Q.setup(n);d=a['d'];models={'baseline':B.Training(a),'optimized':Training(a)}
  result=dict(L=n,C=128,H=256,dropout=.25,checks={},mutated={},poison={},times={},blocks={},meta=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),raw_tri_in_backward={'baseline':True,'optimized':True},includes=['both directions','mask','dropout','residual','live packing','all eleven gradients'],excludes=['optimizer','RNG generation','compile','CPU dispatch']),production_ready=False)
  ref=clone(models['baseline']())
  for name,m in models.items():
   result['checks'][name]=errors(m(),ref);print('CHECK',name,passed(result['checks'][name]),{k:v['relative_l2'] for k,v in result['checks'][name].items()},flush=True)
  m=models['optimized'];m.audit=True;y,k=m.forward();ab,tri,packed,stats,xn=k
  assert tri.data_ptr()==m.audit_tri.data_ptr() and tri.dtype==torch.bfloat16
  assert stats.dtype==torch.float32 and stats.numel()==2*n*n and xn.dtype==torch.bfloat16
  m.backward(k)
  assert m.p1.xhat.data_ptr()==tri.data_ptr()
  result['saved_policy']=dict(tri_is_original_buffer=True,tri_dtype=str(tri.dtype),input_xn_dtype=str(xn.dtype),output_activation_allocated=False,stats_shape=list(stats.shape),stats_dtype=str(stats.dtype),output_side_bytes=tri.numel()*2+stats.numel()*4)
  m.audit=False;del m.audit_tri
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
  assert len(valid)==len(models),result['valid']
  result['timed_invalid']=[k for k in models if k not in valid]
  for scope in ('forward','b1','b7','backward','forward_backward'):
   gs={}
   for name,m in models.items():
    if scope=='forward':f=m.forward
    elif scope=='b1':m.backward(kept[name]);f=m.p1
    elif scope=='b7':m.backward(kept[name]);f=m.p7
    elif scope=='backward':f=lambda m=m,name=name:m.backward(kept[name])
    else:f=m
    gs[name],_=Q.capture_outputs(f)
   blocks=[Q.paired(gs,iterations=200) for _ in range(3)];result['blocks'][scope]=blocks;result['times'][scope]=Q.pool(blocks);print('TIME',scope,{k:v['median_us'] for k,v in result['times'][scope].items()},flush=True)
   path.write_text(json.dumps(result,indent=2));del gs;gc.collect()
  result['cubins']={k:dict(b1=m.p1.k.unit.cubin_path) for k,m in models.items()};result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'bench.py',R/'gate_policy.py',R/'role_plan.py',R/('selected-L%d.json'%n),*R.glob('*.cu'),*R.glob('*.cuh'),*R.glob('*.inc')]};path.write_text(json.dumps(result,indent=2));print('DONE',n,flush=True)
if __name__=='__main__':main()
