from pathlib import Path
import argparse,json,torch
import plans as P
import compare_cueq_training as Q
from compare_all_training import BoundB1
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('b1','b7'),required=True);ap.add_argument('--length',type=int,default=384)
ap.add_argument('--defines',default='{}');ap.add_argument('--part',type=int,default=2);ap.add_argument('--check-only',action='store_true');ap.add_argument('--splits',type=int);args=ap.parse_args()
splits=args.splits or json.loads((R/('tune-%s-L%d.json'%(args.kind,args.length))).read_text())['winner']['splits']
P.DEFINES=tuple(sorted(json.loads(args.defines).items()))
with torch.no_grad():
    a=Q.setup(args.length);d=a['d'];print('SETUP',args.kind,args.length,flush=True)
    if args.kind=='b1':
        saved=a['s'];old=BoundB1(d,a['dy'],saved,132,2,'dual_ln_prefetch')
        ref=tuple(x.clone() for x in old())
        plan=P.B1(d,a['dy'],saved[0].saved_tensors[11],part=args.part,splits=splits)
        names=('dg','dwg','dt','dgo','dbo','dwp');limits=(0,5e-4,2e-5,5e-6,5e-6,5e-4)
    else:
        ref=Q.baseline(a);plan=P.B7(d,a['dy'],a['dl'],a['dr'],a['dg'],part=args.part,splits=splits)
        names=('dx','dwl','dwlg','dwr','dwrg','dgi','dbi');limits=(2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6)
    print('LAUNCH',args.kind,flush=True)
    out=plan();torch.cuda.synchronize()
    errors={name:dict(relative_l2=Q.rel(x,y),max_absolute=(x.float()-y.float()).abs().max().item(),bit_exact=bool(torch.equal(x,y)),finite=bool(torch.isfinite(x).all()),limit=limit)
            for name,x,y,limit in zip(names,out,ref,limits)}
    print('ERRORS',errors,flush=True)
    assert all(v['finite'] and v['relative_l2']<=v['limit'] for v in errors.values()),errors
    if args.check_only:
        print('CHECK_ONLY_DONE',flush=True);raise SystemExit(0)
    (R/('check-%s-L%d.json'%(args.kind,args.length))).write_text(json.dumps(errors,indent=2))
    g,out=Q.capture_outputs(plan);g.replay();torch.cuda.synchronize()
    for x,y,lim in zip(out,ref,limits):assert Q.rel(x,y)<=lim
    ts=Q.paired({'new':g},iterations=100);print('TIME',ts['new']['median_us'],flush=True)
    print('DONE',flush=True)
