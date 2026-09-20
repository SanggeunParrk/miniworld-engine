"""Experimental B7-B12 Plan. Selected source uses direct weight tensor maps."""
from front_core import *
from functools import lru_cache
import ctypes,hashlib,subprocess,re,fcntl
import training_support as T

@lru_cache(None)
def load(count=264,splits=13,part=2,debug=0,source='front_prefetch_lnpair_storepipe'):
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
 def refresh_weights(self):
  # Both published variants bind the four live weight tensors directly.
  if not self.config.get('direct_weights'):
   raise RuntimeError('This checkpoint supports only direct weight tensor maps')
