from pathlib import Path
import argparse,json,sys,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'trimul_b7_sol90_20260921'));import baseline as H
import importlib.util
sp=importlib.util.spec_from_file_location('joint_b7_plan',R/'plan.py');JP=importlib.util.module_from_spec(sp);sp.loader.exec_module(JP);Plan=JP.Plan
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=384);ap.add_argument('--clusters',type=int,default=16);ap.add_argument('--diagnostic-timing',action='store_true');args=ap.parse_args();n=args.length
with torch.no_grad():
 a,m=H.setup(n);ref=tuple(t.clone() for t in m.p7());p=Plan(m.d,*[m.p7.inputs[i] for i in (3,0,1,2)],xn=m.p7.xn,clusters=args.clusters);p.mask=m.p7.mask;p.bind(*m.p7.inputs[:4],xn=m.p7.xn)
 print('LAUNCH',p.count,flush=True);out=p();torch.cuda.synchronize();e=[H.rel(x,y) for x,y in zip(out,ref)];exact=[torch.equal(x,y) for x,y in zip(out,ref)];limits=[2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6];valid=all(v<=t for v,t in zip(e,limits)) and all(torch.isfinite(x).all().item() for x in out)
 result=dict(L=n,clusters=p.clusters,relative_l2=e,bit_exact=exact,limits=limits,valid=valid,source_sha256=p.source_sha256,compiler_log=p.compiler_log,counters=p.counts.tolist());print(result,flush=True)
 (R/('check-L%d-C%d.json'%(n,p.clusters))).write_text(json.dumps(result,indent=2))
 if valid or args.diagnostic_timing:
  result['timed_invalid_diagnostic_only']=not valid
  graphs={name:H.Q.capture_outputs(fn)[0] for name,fn in [('baseline',m.p7),('joint',p)]};t=H.Q.pool([H.Q.paired(graphs,iterations=100) for _ in range(3)]);result['times']={k:v['median_us'] for k,v in t.items()};print(result['times'],flush=True);(R/('check-L%d-C%d.json'%(n,p.clusters))).write_text(json.dumps(result,indent=2))
