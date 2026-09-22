"""Focused sanitizer entry; checks repeated launches and counter reset."""
from check_front import *
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--count',type=int,default=132);ap.add_argument('--part',type=int,default=2);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);p=Plan(a,count=args.count,splits=15*args.count//132,part=args.part)
 for _ in range(2):p()
 torch.cuda.synchronize();e=check(p,a)
 print('SANITIZER_CHECK_PASS',args.length,args.count,args.part,e,flush=True)
