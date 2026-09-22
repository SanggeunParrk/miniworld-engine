from core import *
@lru_cache(None)
def ukernel(count,part=0):
 inc=T._upstream()/'csrc';source=R/'unifiedasm.cu';flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-DROLE=0',f'-DUCOUNT={count}',f'-DPART_ONLY={part}']
 key=hashlib.sha256(source.read_bytes()+(R/'unified_v2.cu').read_bytes()+(R/'fused.cu').read_bytes()+b''.join(p.read_bytes() for p in sorted(R.glob('*.cuh')))+str(flags).encode()).hexdigest();path=R/'build'/(key+'.cubin')
 if not path.exists():
  z=subprocess.run(['nvcc',*flags,str(source),'-o',str(path)],capture_output=True,text=True);path.with_suffix('.ptxas.log').write_text(z.stdout+z.stderr)
  if z.returncode:raise RuntimeError(z.stderr)
  path.with_suffix('.json').write_text(json.dumps(dict(unified=True,count=count,flags=flags)))
 L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(path.read_bytes());unit=L.Unit('unified','sm_90a',0,drv.drv,mod,{},str(path));k=unit.kernel('unifiedasm_b1b4');k.set_max_dynamic_smem(188416);print('CUBIN',str(path),flush=True);return k,unit.kernel('unified_reduce')
class Unified(Plan):
 def __init__(self,d,dy,s,count,part=0):
  assert count<=d['n']**2//64
  super().__init__(d,dy,s,splits=count,compact=1,wgroups=2)
  self.k,self.reduce=ukernel(count,part);self.part=part;self.wgroups=4;self.smem=188416;self.grid=count

 def __call__(self):
  self.k.launch((self.grid,1,1),(512,1,1),[self.p],self.smem)
  if self.part:self.reduce.launch((194,1,1),(256,1,1),[self.p],0)
  return self.outputs
