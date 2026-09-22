from pathlib import Path
import argparse,ctypes,hashlib,json,sys,torch,platform
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_b1_wait_folding_20260921'));import wait_policy as B
OLD=B.BASE.OLD;Q=OLD.Q;P=OLD.P

def rel(a,b):
 x,y=a.float().flatten(),b.float().flatten();return float(torch.linalg.vector_norm((x-y).flatten())/torch.linalg.vector_norm(y.flatten()).clamp_min(1e-20))

def single(plan,index):
 def call():
  L=OLD.R.T._launch_module();(k,count,threads,smem),p=plan.units[index],plan.params[index];drv=k.unit.drv;args=L._Packed([p]);stream=int(torch.cuda.current_stream().cuda_stream)
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(k.handle)),count,1,1,threads,1,1,smem,drv.d.CUstream(stream),ctypes.addressof(args.array)))
  return plan.outputs
 return call

def setup(n):
 a=Q.setup(n);m=B.Training(a);m();return a,m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--profile',action='store_true');args=ap.parse_args();n=args.length
 with torch.no_grad():
  a,m=setup(n);d=a['d']
  if args.profile:
   for _ in range(5):m.p7()
   torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();m.p7();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop();return
  record=dict(L=n,host=platform.node(),gpu=torch.cuda.get_device_name(),config=m.p7.cfg,units=[],times={},reference_checks={},saved_comparison={})
  for k,count,threads,smem in m.p7.units:
   path=Path(k.unit.cubin_path);record['units'].append(dict(cubin=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ctas=count,threads=threads,smem=smem,compiler=path.with_suffix('.ptxas.log').read_text()))
  k3=json.loads((P.S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner'];saved=P.S.A.Training(a,k3)
  for mutant in (False,True):
   if mutant:
    d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);a['dy'].mul_(.91);d['ds'].copy_(d['ds'].roll(1,0));d['mask'].copy_(1-d['mask']);a['mask'].copy_(d['mask'].reshape(-1))
   y,ss=saved.forward_saved();ref=(y.clone(),tuple(t.clone() for t in P.S.C.backward(d,ss,a['dy'])));out=m();e=OLD.H.errors(out,ref);record['reference_checks'][str(mutant)]=e
   _,_,_,_,xn=m.forward()[1];ctx,mu,rs=ss;refxn=ctx.saved_tensors[0]
   record['saved_comparison'][str(mutant)]={'xn_relative_l2':rel(xn,refxn),'xn_bit_exact':torch.equal(xn.reshape(-1),refxn.reshape(-1)),'pre_shape':list(ctx.saved_tensors[8].shape),'saved_shapes':[list(t.shape) for t in ctx.saved_tensors]}
   print('REFERENCE',mutant,{k:v['relative_l2'] for k,v in e.items()},flush=True)
  # Time pristine inputs in a fresh fixture, independent of accuracy mutations.
  a,m=setup(n)
  fns={'dw':single(m.p7,0),'dx_ln_residual':single(m.p7,1),'b7_b12':m.p7,'full_training':m}
  (R/('baseline-L%d.json'%n)).write_text(json.dumps(record,indent=2))
  for name,fn in fns.items():
   # Full forward/backward rebinds p7; time each graph before any later rebind.
   g,outputs=Q.capture_outputs(fn)
   t=Q.pool([Q.paired({name:g},iterations=150) for _ in range(3)]);record['times'][name]=t[name];print('TIME',name,t[name]['median_us'],flush=True);del g,outputs
  (R/('baseline-L%d.json'%n)).write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
