import argparse,json,torch
import bench as B
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--baseline',action='store_true');ap.add_argument('--profile',action='store_true');ap.add_argument('--config',default='{"count":132,"defines":{"GATE_PHASE":1,"LOWREG":1,"STREAM_LN":1}}');args=ap.parse_args()
with torch.no_grad():
 a=B.Q.setup(args.length);d=a['d'];_,kept,xn,_=B.LN.forward(d,1);sd=dict(d,x=xn);tri=kept[1]
 base=B.N.SP.B1(sd,a['dy'],tri,splits=24,defines=dict(GATE_SPLITS=36,TRAIN_L=args.length,B1_DWPROJ_PIPE=1,USE_SAVED_XN=1))
 p=base if args.baseline else B.N.Plan(sd,a['dy'],tri,**json.loads(args.config))
 ref=tuple(t.clone() for t in base());out=p();torch.cuda.synchronize();e=B.error(out,ref);assert B.valid(e),e
 if args.profile:
  p();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();p();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 else:
  for _ in range(2):p()
  g,out=B.Q.capture_outputs(p);g.replay();torch.cuda.synchronize();assert B.valid(B.error(out,ref))
  sd['x'].mul_(.91);tri.mul_(1.13);a['dy'].mul_(.79);d['ds'].copy_(d['ds'].roll(1,0));ref=tuple(t.clone() for t in base());g.replay();torch.cuda.synchronize();assert B.valid(B.error(out,ref)),B.error(out,ref)
 print('CHECK_DONE',e,flush=True)
