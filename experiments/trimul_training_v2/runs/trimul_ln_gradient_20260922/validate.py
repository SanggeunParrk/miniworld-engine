from pathlib import Path
import importlib.util,json,torch,platform,hashlib,subprocess,threading,time,gc,collections,os
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('full_latest_local_policy',R/'policy.py');F=importlib.util.module_from_spec(s);s.loader.exec_module(F);P=F.P
Q=P.Q;H=P.H

def clone(v):return v[0].clone(),tuple(t.clone() for t in v[1])
def errors(v,ref):
 e=H.errors(v,ref)
 for k,x in e.items():
  limit=0 if k=='forward' else 2e-5 if k=='dx' else 5e-6 if k.startswith(('dgamma','dbeta')) else 5e-4
  x['limit']=limit;x['valid']=x['finite'] and x['relative_l2']<=limit
 return e

def units(p):
 kernels=[k for k,*_ in p.units] if hasattr(p,'units') else [p.k]
 return [dict(path=k.unit.cubin_path,sha256=hashlib.sha256(Path(k.unit.cubin_path).read_bytes()).hexdigest()) for k in kernels]

with torch.no_grad():
 a=Q.setup(384);d=a['d'];models={'historical_1048':P.Historical(a),'regression_1185':P.Regression(a),'latest_combined':P.Latest(a),'fixed':F.Fixed(a)}
 record=dict(L=384,C=128,H=256,dropout=.25,host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),includes=['fresh training forward saves','both directions','B1-B12','cuBLAS','all 11 gradients','live weight packing','pair mask','fixed dropout25%','residual'],excludes=['optimizer','dropout RNG generation','CPU dispatch','compilation'],checks={},times={},rounds={},cubins={n:dict(b1=units(m.p1),b7=units(m.p7)) for n,m in models.items()},selection=P.SELECTION)
 def save():(R/'validation.json').write_text(json.dumps(record,indent=2))
 save();graphs={};outputs={}
 for name,m in models.items():graphs[name],outputs[name]=Q.capture_outputs(m)
 tensors=list({t.data_ptr():t for t in [d['x'],d['leaves'][1],d['leaves'][5],d['wp'],d['go'],d['bo'],a['dy'],d['ds'],d['mask'],a['mask']]}.values());snap=[t.clone() for t in tensors]
 for case in range(3):
  if case:
   d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);d['wp'].mul_(.93);d['go'].mul_(1.07);d['go'][0]=0;d['bo'].add_(.017);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
  ref=clone(models['regression_1185']());record['checks'][str(case)]={}
  for name,m in models.items():
   eager=clone(m());graphs[name].replay();graphs[name].replay();torch.cuda.synchronize()
   es=errors(eager,ref);gs=errors(outputs[name],ref);vs=errors(outputs[name],eager)
   valid=all(x['valid'] for vals in (es,gs,vs) for x in vals.values()) and all(x['bit_exact'] for x in vs.values())
   record['checks'][str(case)][name]=dict(valid=valid,eager=es,graph=gs,graph_vs_eager=vs)
   print('CHECK',case,name,valid,{k:x['relative_l2'] for k,x in gs.items()},flush=True);save()
   assert all(x['finite'] for vals in (es,gs,vs) for x in vals.values()) and all(x['bit_exact'] for x in vs.values()),'Nonfinite or graph/eager mismatch'
   if name!='latest_combined':assert valid,(case,name)
 record['valid_models']=[n for n in models if all(c[n]['valid'] for c in record['checks'].values())]
 record['diagnostic_only_models']=[n for n in models if n not in record['valid_models']]
 record['production_ready']=False;record['fixed_strict_validation_passed']=all(c['fixed']['valid'] for c in record['checks'].values())
 for t,v in zip(tensors,snap):t.copy_(v)
 del snap
 uuid=str(torch.cuda.get_device_properties(0).uuid);uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid;record['gpu_uuid']=uuid
 phase=['idle'];stop=threading.Event();samples=[]
 def monitor():
  while not stop.is_set():
   label=phase[0];t=time.monotonic();p=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   samples.append(dict(t=t,phase=label,csv=p.stdout.strip(),returncode=p.returncode));stop.wait(.05)
 thread=threading.Thread(target=monitor);thread.start()
 try:
  phase[0]='full';rr=[Q.paired(graphs,warmup=100,iterations=300) for _ in range(6)];record['rounds']['full']=rr;record['times']['full']=Q.pool(rr);print('TIME full',{k:v['median_us'] for k,v in record['times']['full'].items()},flush=True);save()
  for scope in ('forward','backward','b1','b7'):
   phase[0]=scope;gs={};kept_by_name={};calls_by_name={};captured_by_name={}
   for name,m in models.items():
    kept=m.forward()[1];kept_by_name[name]=kept
    if scope=='forward':fn=m.forward
    elif scope=='backward':fn=lambda m=m,kept=kept:m.backward(kept)
    else:m.backward(kept);fn=m.p1 if scope=='b1' else m.p7
    calls_by_name[name]=fn;gs[name],captured_by_name[name]=Q.capture_outputs(fn)
   rr=[Q.paired(gs,warmup=100,iterations=300) for _ in range(5)];record['rounds'][scope]=rr;record['times'][scope]=Q.pool(rr);print('TIME',scope,{k:v['median_us'] for k,v in record['times'][scope].items()},flush=True);save()
   del gs,captured_by_name,calls_by_name,kept_by_name;gc.collect()
 finally:
  stop.set();thread.join();record['telemetry']=samples;save()
 record['traces']={}
 for name,g in graphs.items():
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
  path=R/('trace-'+name+'.json');prof.export_chrome_trace(str(path));trace=json.loads(path.read_text());names=collections.Counter(ev['name'] for ev in trace['traceEvents'] if ev.get('cat')=='kernel');assert names
  record['traces'][name]=dict(names)
 assert record['traces']['fixed'].get('b7_joint')==1
 assert record['traces']['latest_combined'].get('b1_fused')==1
 record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'validate.py',R/'policy.py',R.parent/'trimul_b1_gate_demote_20260922/policy.py',R.parent/'trimul_b7_weight_batch128_20260922/plan.py']};record['complete']=True;save();print('DONE',flush=True)
