from pathlib import Path
import sys,torch,json,importlib.util,argparse
R=Path(__file__).resolve().parent;P=R.parent/'trimul_b1_epilogue_fixed_20260921';sys.path.insert(0,str(P));import epilogue_policy as B
spec=importlib.util.spec_from_file_location('gate_only_plan',R/'replace_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--profile',action='store_true');args=ap.parse_args();n=args.length;Q=B.BASE.OLD.Q
with torch.no_grad():
 a=Q.setup(n);m=B.Training(a);_,k=m.forward();m.backward(k);m.p1();ref=m.p1.partw[:,:16384].clone();cfg=json.loads((P/('selected-L%d.json'%n)).read_text())['config']
 p=RP.Plan(dict(a['d'],x=k[-1]),a['dy'],k[1],k[3],**cfg);p.outputs[0].copy_(m.p1.outputs[0]);p();torch.cuda.synchronize();assert torch.equal(p.partw[:,:16384],ref)
 if args.profile:
  for _ in range(10):p()
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();p();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 else:
  g0,_=Q.capture_outputs(m.p1);g1,_=Q.capture_outputs(p);blocks=[Q.paired({'full_b1':g0,'gate_only':g1},iterations=200) for _ in range(3)];timings=Q.pool(blocks)
  result=dict(L=n,gate_partial_bit_exact=True,times=timings,gate_stream_payload_bytes=512*n*n+132*16384*4,diagnostic_only=True,scope='same CTA-owned gate phase, writes gate partials; excludes final cross-CTA reduction',smem=p.smem,cubin=p.k.unit.cubin_path)
  (R/('results-L%d.json'%n)).write_text(json.dumps(result,indent=2));print({key:x['median_us'] for key,x in timings.items()},flush=True)
