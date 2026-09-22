"""Experimental B7-B12 Plan. Selected source uses direct weight tensor maps."""
from front_core import *
from functools import lru_cache
import ctypes,hashlib,subprocess,re,fcntl
from miniworld_engine.kernels.trimul_inproj.cuda import anthropic_training as T

@lru_cache(None)
def load(count=132,splits=15,part=2,debug=0,source='front_roundonly'):
 inc=T._upstream()/'csrc';src=R/(source+'.cu')
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),f'-DUCOUNT={count}',f'-DDW_SPLITS={splits}',f'-DPART_ONLY={part}',f'-DDEBUG_SAVE={debug}']
 cfg_path=R/(source+'.launch.json')
 cfg=json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
 if 'ptxas_register_usage_level' in cfg:
  level=cfg['ptxas_register_usage_level'];assert isinstance(level,int) and 0<=level<=10
  flags.append(f'-Xptxas=--register-usage-level={level}')
 deps=(R/'front_primitives.cuh').read_bytes()+b''.join((inc/f).read_bytes() for f in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))
 if 'front_mn_primitives.cuh' in src.read_text() or 'warp_primitives.cuh' in src.read_text():deps+=(R/'front_mn_primitives.cuh').read_bytes()
 if 'warp_primitives.cuh' in src.read_text():deps+=(R/'warp_primitives.cuh').read_bytes()
 key=hashlib.sha256(src.read_bytes()+deps+str(flags).encode()).hexdigest();path=R/'build'/(key+'.cubin');path.parent.mkdir(exist_ok=True)
 command=['nvcc',*flags,str(src),'-o',str(path)];log=path.with_suffix('.ptxas.log')
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not path.exists():
   p=subprocess.run(command,capture_output=True,text=True);log.write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError(p.stderr)
   path.with_suffix('.json').write_text(json.dumps(dict(source=source,count=count,splits=splits,part=part,debug=debug,command=command),indent=2))
 compiler=log.read_text();print('PTXAS',str(path),compiler,flush=True)
 if re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)',compiler):raise RuntimeError('Spill regression, refusing GPU execution: '+str(log))
 if re.search(r'(?<!\d)[1-9]\d* bytes stack frame',compiler):raise RuntimeError('Local stack allocation, refusing GPU execution: '+str(log))
 launch=T._launch_module();drv=launch.BlockDriver(device=0);mod=drv.load(path.read_bytes());unit=launch.Unit(source,'sm_90a',0,drv.drv,mod,{},str(path))
 if '__device__ float sigmoid_lut[65536]' in src.read_text() or '#define COMPACT_SIGMOID_LUT 1' in src.read_text():unit.kernel('init_sigmoid_lut').launch((256,1,1),(256,1,1),[],0)
 k=unit.kernel('front_b7b12');k.set_max_dynamic_smem(229376)
 return k,unit.kernel('front_reduce')

class Plan:
 def __init__(self,a,count=132,splits=15,part=2,debug=0,source='front_selected'):
  if source=='front_selected':
   if count not in (66,132):raise ValueError('Selected configurations are validated for 66 or 132 CTAs')
   source='front_roundonly' if count==132 else 'front_rounding'
  self.a=a;d=a['d'];n=d['n'];m=n*n
  assert n>=64 and m%64==0 and 0<4*splits<count<=torch.cuda.get_device_properties(0).multi_processor_count
  assert part in (1,2) and torch.cuda.get_device_capability(0)==(9,0)
  self.count=count;self.splits=splits;self.part=part;self.debug=debug;self.source=source
  config_path=R/(source+'.launch.json');self.config=json.loads(config_path.read_text()) if config_path.exists() else {}
  assert not self.config.get('pre_split') or '#define PRE_SPLIT_TMA 1' in (R/(source+'.cu')).read_text()
  assert not self.config.get('direct_weights') or '#define DIRECT_FOUR_WEIGHTS 1' in (R/(source+'.cu')).read_text()
  self.k,self.reduce=load(count,splits,part,debug,source)
  self.wfront=None if self.config.get('direct_weights') else torch.cat((a['wlg'],a['wl'],a['wrg'],a['wr']),dim=1).contiguous()
  self.dx=torch.empty((m,128),device=d['x'].device,dtype=torch.bfloat16)
  self.dw=torch.empty((4,128,256),device=self.dx.device,dtype=self.dx.dtype)
  self.dgam=torch.empty(128,device=self.dx.device);self.dbeta=torch.empty_like(self.dgam)
  self.partw=torch.empty((4,splits*self.config.get('wgrad_slices',1),32768),device=self.dx.device)
  self.partln=torch.empty((count-4*splits,256),device=self.dx.device)
  self.counts=torch.zeros(2+(self.config.get('trace_words',4)*count if source.endswith('_trace') else 0),dtype=torch.int32,device=self.dx.device)
  self.debugdc=torch.empty((1024,m),device=self.dx.device,dtype=self.dx.dtype) if debug else self.dx
  self.debugxn=torch.empty_like(self.dx) if debug else self.dx
  self.outputs=(self.dx,*self.dw.unbind(0),self.dgam,self.dbeta)
  self.bind(a['dl'],a['dr'],a['dg'],a['dy'])

 def refresh_weights(self):
  a=self.a
  if self.config.get('direct_weights'):return
  # Caller performs this when parameters change; pointers/descriptors stay stable.
  self.wfront[:,:256].copy_(a['wlg']);self.wfront[:,256:512].copy_(a['wl'])
  self.wfront[:,512:768].copy_(a['wrg']);self.wfront[:,768:].copy_(a['wr'])

 def bind(self,dl,dr,dg,dy):
  a=self.a;d=a['d'];n=d['n'];m=n*n;launch=T._launch_module()
  tm=lambda t,box,dims,strides:launch.tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')
  row=lambda t:tm(t,[64,64],[128,m],[256])
  pre_map=tm(a['pre'],[64,1,128],[m,2,512],[m*2,m*4]) if self.config.get('pre_split') else tm(a['pre'],[64,256],[m,1024],[m*2])
  weight_maps=[tm(a[k],[64,64],[256,128],[512]) for k in ('wlg','wl','wrg','wr')] if self.config.get('direct_weights') else [tm(self.wfront,[64,64],[1024,128],[2048])]
  maps=[tm(dl,[64,128],[m,256],[m*2]),tm(dr,[64,128],[m,256],[m*2]),pre_map,row(a['xn']),row(dg),
    *weight_maps,tm(a['wg'],[64,64],[128,128],[256]),row(d['x']),row(dy),row(self.dx)]
  self.inputs=(dl,dr,dg,dy)
  self.p=launch.Struct([*maps,a['mask'],a['mu'],a['rs'],d['gi'],self.dw,self.dgam,self.dbeta,self.partw,self.partln,self.counts,self.debugdc,self.debugxn,m,n,m//64])

 def __call__(self):
  if self.part==2:
   launch=T._launch_module();drv=self.k.unit.drv;packed=launch._Packed([self.p]);stream=int(torch.cuda.current_stream().cuda_stream)
   drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.count,1,1,256,1,1,229376,drv.d.CUstream(stream),ctypes.addressof(packed.array)))
  else:
   self.k.launch((self.count,1,1),(256,1,1),[self.p],229376)
   self.reduce.launch(((131328+255)//256,1,1),(256,1,1),[self.p],0)
  return self.outputs

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--part',type=int,default=2);ap.add_argument('--count',type=int,default=132);ap.add_argument('--splits',type=int,default=15);ap.add_argument('--debug',type=int,default=0);ap.add_argument('--source',default='front_selected');ap.add_argument('--bench',action='store_true');args=ap.parse_args()
 with torch.no_grad():
  a=setup(args.length);ref,dc,dxn=baseline(a,True);p=Plan(a,args.count,args.splits,args.part,args.debug,args.source)
  out=p();torch.cuda.synchronize();e=errors(out,ref);print('ERRORS',json.dumps(e),flush=True)
  if args.debug:print('INTERMEDIATES',rel(p.debugdc,dc),rel(p.debugxn,dxn),flush=True)
  print('COUNTERS',p.counts.tolist(),flush=True)
  if args.bench:
   assert all(v['finite'] and v['relative_l2']<.001 for v in e.values()),e
   graphs={'baseline':capture(lambda:baseline(a)),'cuda':capture(p)};t=paired(graphs)
   record=dict(L=args.length,config=vars(args),errors=e,times=t)
   dest=R/f'{args.source}-L{args.length}-s{args.splits}-p{args.part}-debug{args.debug}.json';dest.write_text(json.dumps(record,indent=2))
   print('TIMES',{k:{q:v for q,v in z.items() if q!='samples_us'} for k,z in t.items()},flush=True)
