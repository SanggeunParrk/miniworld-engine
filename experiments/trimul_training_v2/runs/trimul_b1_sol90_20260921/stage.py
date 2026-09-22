from pathlib import Path
import sys,importlib.util,torch,json,platform
import sol_policy as P
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('b1_stage_plan',R/'stage/replace_plan.py');RP=importlib.util.module_from_spec(spec);spec.loader.exec_module(RP)
Q=P.B.B.BASE.OLD.Q
result=dict(host=platform.node(),labels=['wait_raw','prepare_LN_gate_proj','dWproj_wait_dGate_store','dNorm_LNbwd_store_dTri','phase_A_total','grid_barrier_1','gate_phase','grid_barrier_2','final_reduce'],rows=[],note='clock64 instrumentation adds instructions; CTA mean/max stage cycles, not standalone production latency. Sum stages0..3 is within phase A; do not add phase A twice.')
with torch.no_grad():
 for n in (384,768):
  a=Q.setup(n);m=P.Training(a);_,k=m.forward();m.backward(k);ref=[t.clone() for t in m.p1()]
  cfg=json.loads((R/('selected-L%d.json'%n)).read_text())['config']
  for split in (0,1):
   cfg['defines']['B1_SPLIT_DN']=split;p=RP.Plan(dict(a['d'],x=k[-1]),a['dy'],k[1],k[3],**cfg);out=p();torch.cuda.synchronize();assert all(torch.equal(v,r) for v,r in zip(out,ref))
   for _ in range(5):p()
   p.counts.zero_()
   for _ in range(10):p()
   torch.cuda.synchronize();v=p.counts[512:].view(torch.int64).reshape(132,16).double().cpu()/10
   row=dict(L=n,split=split,mean_cycles=v.mean(0).tolist()[:9],max_cycles=v.max(0).values.tolist()[:9]);result['rows'].append(row);print(row,flush=True)
(R/'stage-node01.json').write_text(json.dumps(result,indent=2))
