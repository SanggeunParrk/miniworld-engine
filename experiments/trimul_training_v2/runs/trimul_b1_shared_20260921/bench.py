from pathlib import Path
import argparse,json,sys,torch,platform,gc
import plan as N
sys.path.insert(0,str(N.OLD));import role_plan as R
import compare_cueq_training as Q
from compare_all_training import BoundB1
sys.path.insert(0,str(N.R.parent/'trimul_ln_only_save_20260921'));import ln_save_core as LN
names=('dg','dwg','dt','dgo','dbo','dwp');limits=(0,5e-4,2e-5,5e-6,5e-6,5e-4)
def error(out,ref):
 return {k:dict(rel=Q.rel(x,y),finite=bool(torch.isfinite(x).all()),bit_exact=bool(torch.equal(x,y)),limit=lim) for k,x,y,lim in zip(names,out,ref,limits)}
def valid(e):return all(v['finite'] and v['rel']<=v['limit'] for v in e.values())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--configs',default='[{"count":132}]');ap.add_argument('--name',default='first');ap.add_argument('--check-only',action='store_true');args=ap.parse_args();n=args.length
 with torch.no_grad():
  a=Q.setup(n);d=a['d'];_,kept,xn,_=LN.forward(d,1);tri=kept[1];sd=dict(d,x=xn)
  base=N.SP.B1(sd,a['dy'],tri,splits=24,defines=dict(GATE_SPLITS=36,TRAIN_L=n,B1_DWPROJ_PIPE=1,USE_SAVED_XN=1))
  ref=tuple(t.clone() for t in base());ind=tuple(t.clone() for t in BoundB1(d,a['dy'],a['s'],132,2,'dual_ln_prefetch')());torch.cuda.synchronize()
  base_error=error(ref,ind);assert valid(base_error),base_error
  gb,_=Q.capture_outputs(base);rows=[]
  for config in json.loads(args.configs):
   row=dict(config=config)
   try:
    p=N.Plan(sd,a['dy'],tri,**config);out=p();torch.cuda.synchronize();e=error(out,ref);ei=error(out,ind);row.update(errors=e,independent=ei,valid=valid(e) and valid(ei),cubin=p.k.unit.cubin_path)
    print('CHECK',config,e,flush=True)
    if row['valid'] and not args.check_only:
     g,go=Q.capture_outputs(p);g.replay();torch.cuda.synchronize();row['graph_errors']=error(go,ref)
     blocks=[Q.paired({'baseline':gb,'candidate':g},iterations=100) for _ in range(3)];row['blocks']=blocks;row['time']=Q.pool(blocks)
     print('TIME',config,{k:v['median_us'] for k,v in row['time'].items()},flush=True)
     # Replay after in-place changes to both operands and dropout scale.
     orig=[t.clone() for t in (sd['x'],tri,a['dy'],d['ds'])]
     sd['x'].mul_(.97);tri.mul_(1.03);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0))
     rr=tuple(t.clone() for t in base());g.replay();torch.cuda.synchronize();row['mutated']=error(go,rr);row['valid'] &= valid(row['mutated'])
     for t,v in zip((sd['x'],tri,a['dy'],d['ds']),orig):t.copy_(v)
     del g,go
    del p,out;gc.collect()
   except RuntimeError as ex:row.update(valid=False,error=str(ex));print('ERROR',str(ex),flush=True)
   rows.append(row);(N.R/('%s-L%d.json'%(args.name,n))).write_text(json.dumps(dict(L=n,baseline_error=base_error,rows=rows,host=platform.node(),gpu=torch.cuda.get_device_name()),indent=2))
if __name__=='__main__':main()
