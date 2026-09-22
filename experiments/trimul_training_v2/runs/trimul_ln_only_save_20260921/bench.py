from pathlib import Path
import argparse,hashlib,json,platform,torch
import ln_save_core as C
import compare_cueq_training as Q
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--check-only',action='store_true');args=ap.parse_args();n=args.length

def capture(fn):return Q.capture_outputs(fn)
def rel(x,y):return (x.float()-y.float()).norm().item()/max(y.float().norm().item(),1e-30)
def normref(x,g,b):
 f=x.float();f=(f-f.mean(-1,keepdim=True))*torch.rsqrt(f.var(-1,unbiased=False,keepdim=True)+1e-5)
 return (f*g+b).bfloat16()
def validate(out,ref,lnin,lnout):
 y,xi,xo=out;torch.cuda.synchronize();assert torch.equal(y,ref),'forward changed'
 result=dict(output_bit_exact=True)
 for key,t,r in [('input_ln',xi,lnin),('output_ln',xo,lnout)]:
  if t is not None:
   e=rel(t.reshape_as(r),r);assert bool(torch.isfinite(t).all()) and e<=5e-4,(key,e);result[key]=dict(relative_l2=e,finite=True)
 return result
with torch.no_grad():
 a=Q.setup(n);d=a['d'];ref,kept=C.S.forward(d);ref=ref.clone();tri=kept[1]
 lin=normref(d['x'],d['gi'],d['bi']);lout=normref(tri.permute(1,2,0).reshape(n*n,256),d['go'],d['bo'])
 bufs=(torch.empty_like(d['x']),torch.empty_like(d['x']),torch.empty((n*n,256),device='cuda',dtype=torch.bfloat16))
 configs={};checks={};graphs={};outputs={}
 for mode in range(4):
  for method in ((0,) if mode==0 else (0,1)):
   label='%d_%s'%(mode,'tma' if method==0 else 'stg');configs[label]=dict(mode=mode,method=method,cfg=C.DEFAULT)
   active=(bufs[0],bufs[1] if mode&1 else None,bufs[2] if mode&2 else None)
   fn=lambda mode=mode,method=method,active=active:C.output(d,tri,mode,method,bufs=active)
   checks[label]=validate(fn(),ref,lin,lout)
   print('CHECK',label,checks[label],flush=True)
   if not args.check_only:graphs[label],outputs[label]=capture(fn)
 if args.check_only:print('CHECK_ONLY_DONE',flush=True);raise SystemExit
 blocks=[Q.paired(graphs,iterations=200) for _ in range(3)];times=Q.pool(blocks)
 winners={mode:min([k for k,c in configs.items() if c['mode']==mode],key=lambda k:times[k]['median_us']) for mode in range(4)}
 result=dict(L=n,C=128,H=256,dropout=.25,metadata=dict(gpu=torch.cuda.get_device_name(),hostname=platform.node(),torch=torch.__version__,saving='BF16 affine input/output LN activations only; both emitted in K3, K1 unchanged',other_saves='none added: no mean/rstd/preactivation/projection/gate; left/right/tri retained as before',timing='fixed common K3 config; two store schedules compared; live packing included in full forward; CUDA graphs, 600 interleaved samples per variant'),configs=configs,checks=checks,k3_blocks=blocks,k3_times=times,winners=winners)
 print('K3',{k:v['median_us'] for k,v in times.items()},'WINNERS',winners,flush=True)
 del graphs,outputs
 gs={};os={};fs={}
 for mode,label in winners.items():
  cfg=configs[label];fs[str(mode)]=lambda cfg=cfg:C.forward(d,**cfg)
 # Reference is literally the unmodified current no-save forward.
 fs['original']=lambda:C.S.forward(d)
 for name,fn in fs.items():
  gs[name],os[name]=capture(fn);gs[name].replay();torch.cuda.synchronize();assert torch.equal(os[name][0],ref),name
 blocks=[Q.paired(gs,iterations=200) for _ in range(3)];result['full_blocks']=blocks;result['full_times']=Q.pool(blocks)
 print('FULL',{k:v['median_us'] for k,v in result['full_times'].items()},flush=True)
 # Every saved activation must update during graph replay.
 tensors={'x':d['x'],'gi':d['gi'],'go':d['go'],'ds':d['ds'],'mask':d['mask']};original={k:v.clone() for k,v in tensors.items()}
 d['x'].mul_(.97);d['gi'].mul_(1.02);d['go'].mul_(.98);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask'])
 fresh,_=C.S.forward(d);fresh=fresh.clone();checks2={}
 for name,g in gs.items():
  g.replay();torch.cuda.synchronize();assert torch.equal(os[name][0],fresh),name
  if name=='original':continue
  _,kept2,xi,xo=os[name]
  rr=normref(d['x'],d['gi'],d['bi']);ro=normref(kept2[1].permute(1,2,0).reshape(n*n,256),d['go'],d['bo'])
  checks2[name]=validate((os[name][0],xi,xo),fresh,rr,ro)
 result['mutated_graph_checks']=checks2
 for k,t in tensors.items():t.copy_(original[k])
 result['retained_ln_bytes']={str(mode):n*n*2*((128 if mode&1 else 0)+(256 if mode&2 else 0)) for mode in range(4)}
 result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'bench.py',R/'ln_save_core.py',R/'ln_only_k3.cu')}
 result['cubins']={k:dict(path=C.kernel(v['mode'],v['method'])[0].unit.cubin_path,sha256=hashlib.sha256(Path(C.kernel(v['mode'],v['method'])[0].unit.cubin_path).read_bytes()).hexdigest()) for k,v in configs.items()}
 (R/('results-L%d.json'%n)).write_text(json.dumps(result,indent=2));print('DONE',n,flush=True)
