import argparse
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--length',type=int,default=64);ap.add_argument('--ring',action='store_true');args=ap.parse_args()
if args.ring:
 from check_ring import *
 cls=RingPlan;sp=20
else:
 from check_twocta import *
 cls=WarpPlan;sp=13
with torch.no_grad():
 a=setup(args.length);p=cls(a,count=264,splits=sp,source=args.source)
 for _ in range(2):p()
 torch.cuda.synchronize();print('SANITIZER_CHECK_PASS',args.length,check(p,a),flush=True)
