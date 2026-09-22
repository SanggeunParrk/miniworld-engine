"""Matched bidirectional cuEq-primitives composition vs selected CUDA training.

cuEq's public single-direction TMU is a different operation. This uses the
engine's shared-2H-LN cuEq composition, with identical supplied pair mask,
weights, residual and dropout mask. cuEq retains its own internal saves and
autograd. Report backward separately from forward; no one-direction shortcut.
"""
from measure_experiment import *
from integrate import backward_cuda
import importlib.metadata,os
sys.path.insert(0,str(R.parent/'anthropic_ln_equal_saves_20260919'))
import core_saved as C
import cuequivariance_ops_torch
from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm
from miniworld_engine.modules.functional import sigmoid_gate

def cueq_forward(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask,ds):
 xn=layer_norm_transpose(x,gi,bi,eps=1e-5,layout='bijd->bijd')
 ab=fused_sigmoid_gated_dual_gemm(xn,torch.cat((wlg,wrg)),torch.cat((wl,wr)),mask=mask,transpose_out=True)
 left,right=ab.chunk(2,dim=0);h=go.numel()//2
 outgoing=torch.einsum('dbik,dbjk->dbij',left[:h],right[:h])
 incoming=torch.einsum('dbki,dbkj->dbij',left[h:],right[h:])
 tri=torch.cat((outgoing,incoming),dim=0)
 norm=layer_norm_transpose(tri,go,bo,eps=1e-5,layout='dbij->bijd')
 update=sigmoid_gate(torch.nn.functional.linear(xn,wg),torch.nn.functional.linear(norm,wp))
 return update*ds+x

def pooled(blocks):
 out={}
 for key in blocks[0]:
  samples=sorted(t for block in blocks for t in block[key]['samples_us'])
  out[key]=dict(median_us=statistics.median(samples),p90_us=samples[int(.9*(len(samples)-1))],samples_us=samples)
 return out

def capture_on(fn,stream):
 stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(2):fn()
 torch.cuda.current_stream().wait_stream(stream)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):fn()
 return g

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--lengths',nargs='+',type=int,default=[384,768]);ap.add_argument('--output',default='cueq-comparison.json');args=ap.parse_args()
 metadata=dict(cuequivariance_torch=importlib.metadata.version('cuequivariance-torch'),cuequivariance_ops=importlib.metadata.version('cuequivariance-ops-torch-cu12'),torch=torch.__version__,gpu=torch.cuda.get_device_name(),tuning=os.environ.get('CUEQ_TRITON_TUNING','default'),dtype='bf16; FP32 LN parameters',dropout=.25,
   semantics='shared 256-channel output LN, same pair mask/weights/dropout/residual; vendor internal rounding and saves retained',compile='static fullgraph, Inductor cudagraphs off; explicit one-call CUDA Graph for all timed variants')
 print('METADATA',metadata,flush=True)
 compiled=torch.compile(cueq_forward,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
 rows=[]
 for n in args.lengths:
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n)
  with torch.no_grad():
   ref_y,_=C.forward(d,True,(3,64,2,2,1),(1,1))
   plan=Experiment(d,dy,s,132,2,'dual_ln_prefetch')
   ref_grad=C.backward(d,s,dy)
   cuda_grad=backward_cuda(d,s,dy,plan)
  mask=d['mask'].reshape(1,n,n).to(torch.bfloat16)
  ys={};checks={};streams={};functions={'cuda':lambda:backward_cuda(d,s,dy,plan),'triton_cublas':lambda:C.backward(d,s,dy)}
  for label,fn in [('cueq_eager',cueq_forward),('cueq_compile',compiled)]:
   print('BUILD',n,label,flush=True)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());streams[label]=stream
   # Autograd follows the forward stream. Build and capture its backward on
   # that same non-default stream, avoiding an implicit legacy-stream edge.
   with torch.cuda.stream(stream):
    # Independent leaves create AccumulateGrad on this forward's stream.
    # Reusing leaves touched by the manual reference keeps a legacy-stream
    # AccumulateGrad alive and prevents a backward-only graph capture.
    leaves=tuple(v.detach().clone().requires_grad_(True) for v in d['leaves'])
    inputs=(*leaves,mask,d['ds'].reshape(1,1,n,128))
    y=fn(*inputs);ys[label]=y
    grads=torch.autograd.grad(y,leaves,dy,retain_graph=True)
   torch.cuda.current_stream().wait_stream(stream)
   errors=[rel(a,b) for a,b in zip(grads,ref_grad)]
   checks[label]=dict(forward_relative_l2=rel(y,ref_y),gradient_relative_l2=errors,finite=all(torch.isfinite(g).all().item() for g in grads))
   print('CHECK',n,label,checks[label],flush=True)
   assert checks[label]['finite'] and max(errors)<.03,checks[label]
   functions[label]=lambda y=y,leaves=leaves:torch.autograd.grad(y,leaves,dy,retain_graph=True)
  with torch.no_grad():
   graphs={}
   for k,fn in functions.items():
    print('CAPTURE',n,k,flush=True)
    graphs[k]=capture_on(fn,streams[k]) if k in streams else capture(fn)
   blocks=[paired_events(graphs) for _ in range(3)]
  times=pooled(blocks)
  print('RESULT',n,[{k:round(v['median_us'],3) for k,v in b.items()} for b in blocks],flush=True)
  rows.append(dict(L=n,metadata=metadata,checks=checks,warmup_per_block=20,iterations_per_block=200,blocks=blocks,times=times))
  (R/args.output).write_text(json.dumps(rows,indent=2))
