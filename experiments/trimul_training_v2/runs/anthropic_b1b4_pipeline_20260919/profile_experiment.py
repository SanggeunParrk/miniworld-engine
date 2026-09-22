"""Use --profile-from-start off to profile exactly one selected launch."""
import argparse
from dual_experiment import *
from check_experiment import change_inputs

parser=argparse.ArgumentParser()
parser.add_argument('--source',default='dual_dspref')
parser.add_argument('--length',type=int,default=384)
parser.add_argument('--count',type=int,default=132)
parser.add_argument('--part',type=int,default=2)
args=parser.parse_args()
with torch.no_grad():
    d,dy,saved=data(args.length)
    change_inputs(d,dy,.25,20260920+args.length)
    p=Experiment(d,dy,saved,args.count,args.part,args.source)
    for _ in range(20):p()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    p()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
