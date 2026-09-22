import sys,json,gc,statistics,subprocess,os
from core import *
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'))
from bench_k3 import graph
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward
rows=json.loads((R/'results.json').read_text());(R/'results-short.json').write_text(json.dumps(rows,indent=2))
telemetry=subprocess.Popen(['nvidia-smi','-i',os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0],'--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu','--format=csv','-lms','200','-f',str(R/'telemetry.csv')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
 for row in rows:
  n=row['N'];cf=row['configs'];lc=tuple(cf['ln_anthropic']);tc=tuple(cf['ln_triton']);kc=tuple(cf['k1_split']);oc=tuple(cf['k3_split']);f1=tuple(cf['k1_original']);f3=tuple(cf['k3_original'])
  torch.manual_seed(67);x=torch.randn(1,n,n,128,device='cuda',dtype=torch.bfloat16);g=torch.rand(128,device='cuda');b=torch.randn_like(g);go=torch.rand(256,device='cuda');bo=torch.randn_like(go)
  w=torch.randn(1024,128,device='cuda',dtype=x.dtype)/128**.5;wp=torch.randn(128,256,device='cuda',dtype=x.dtype)/16;wg=torch.randn(128,128,device='cuda',dtype=x.dtype)/128**.5;mask=(torch.rand(n,n,device='cuda')>.2).float()
  def full(mode):
   z=x if mode=='original' else ln(x,g,b,lc) if mode=='split-anthropic-ln' else triton_ln(x,g,b,tc)
   a=front(z,w,mask,g,b,mode=='original',f1 if mode=='original' else kc);t=packed_forward(a[:256],a[256:],128)
   return output(t,z,wp,wg,g,b,go,bo,x,mode=='original',f3 if mode=='original' else oc,tma_residual=row.get('residual_tma',False) and mode!='original')
  keys=['original','split-anthropic-ln','split-triton-ln'];gs={k:graph(lambda k=k:full(k)) for k in keys};times={k:[] for k in keys}
  for _ in range(200):
   for k in keys:gs[k][0].replay()
  for r in range(18):
   order=keys[r%3:]+keys[:r%3]
   if (r//3)%2:order=order[::-1]
   for k in order:
    for _ in range(8):gs[k][0].replay()
    a=torch.cuda.Event(enable_timing=True);btime=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(200):gs[k][0].replay()
    btime.record();btime.synchronize();times[k].append(a.elapsed_time(btime)*1000/200)
  row['full']={k:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),samples_us=v) for k,v in times.items()}
  row['final_protocol']=dict(rounds=18,replays=200,initial_warmup=200,order='rotated and reversed; all six permutations')
  print(n,{k:dict(median=round(v['median_us'],2),min=round(v['min_us'],2),max=round(v['max_us'],2)) for k,v in row['full'].items()},flush=True)
  (R/'results.json').write_text(json.dumps(rows,indent=2));del gs;gc.collect();torch.cuda.empty_cache()
finally:telemetry.terminate();telemetry.wait()
