"""Independent empirical BW calibration; not a proof of model-kernel SoL."""
from measure_experiment import *
src=R/'dual_bandwidth.cu'
flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-Xptxas=-v']
key=hashlib.sha256(src.read_bytes()+str(flags).encode()).hexdigest()
path=R/'build'/(key+'.cubin')
z=subprocess.run(['nvcc',*flags,str(src),'-o',str(path)],capture_output=True,text=True)
path.with_suffix('.ptxas.log').write_text(z.stdout+z.stderr)
assert z.returncode==0,z.stderr
assert not re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)',z.stderr)
L=T._launch_module();drv=L.BlockDriver(device=0);mod=drv.load(path.read_bytes())
unit=L.Unit('bandwidth','sm_90a',0,drv.drv,mod,{},str(path))
rows=[]
for length in (384,768):
 n=(length*length*2824+63)//64
 xs=[torch.randint(0,128,(n,16),device='cuda',dtype=torch.uint8) for _ in range(3)]
 out=torch.empty_like(xs[0]);funcs={}
 for reads in (1,3):
  k=unit.kernel('bandwidth'+str(reads))
  args=[xs[0],out,n] if reads==1 else [*xs,out,n]
  k.launch((528,1,1),(256,1,1),args,0)
  torch.cuda.synchronize()
  ref=xs[0] if reads==1 else xs[0]^xs[1]^xs[2]
  assert torch.equal(out,ref)
  for grid in (132,528,1056):
   funcs[f'r{reads}/g{grid}']=lambda k=k,args=args,grid=grid:k.launch((grid,1,1),(256,1,1),args,0)
 graphs={k:capture(f) for k,f in funcs.items()}
 times=paired_events(graphs)
 row=dict(L=length,tensor_bytes=n*16,times=times,
          TBps={k:(int(k[1])+1)*n*16/v['median_us']/1e6 for k,v in times.items()})
 rows.append(row);print('BW',length,row['TBps'],flush=True)
 (R/'sol-bandwidth.json').write_text(json.dumps(rows,indent=2))
