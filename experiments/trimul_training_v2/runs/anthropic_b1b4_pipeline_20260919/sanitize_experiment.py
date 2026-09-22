"""Focused sanitizer run: real persistent loop, dropout on, changed graph inputs."""
from check_experiment import *
parser=argparse.ArgumentParser()
parser.add_argument('--source',default='dual_maskbits')
parser.add_argument('--length',type=int,default=384)
parser.add_argument('--count',type=int,default=132)
parser.add_argument('--part',type=int,default=2)
args=parser.parse_args()
with torch.no_grad():
 d,dy,s=data(args.length);change_inputs(d,dy,.25,20260920+args.length)
 ref=baseline(d,dy,s);p=Experiment(d,dy,s,args.count,args.part,args.source)
 print('CHECK',check(p(),ref),flush=True)
 assert not torch.count_nonzero(p.workspace[-1]).item()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):p()
 for seed in (20260921+args.length,20260922+args.length):
  change_inputs(d,dy,.25,seed);ref=baseline(d,dy,s);graph.replay();torch.cuda.synchronize()
  print('CHECK',seed,check(p.outputs,ref),flush=True)
  assert not torch.count_nonzero(p.workspace[-1]).item()
 print('CHECK counters_zero=true',flush=True)
