from check_twocta import *
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);args=ap.parse_args()
with torch.no_grad():
 a=setup(args.length);p=WarpPlan(a,count=264,splits=13,source='front_twocta_kindwg')
 for _ in range(2):p()
 torch.cuda.synchronize();print('SANITIZER_CHECK_PASS',args.length,check(p,a),flush=True)
