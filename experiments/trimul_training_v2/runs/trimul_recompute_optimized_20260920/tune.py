from pathlib import Path
import argparse,gc,json,torch
import plans as P
import compare_cueq_training as Q
from compare_all_training import BoundB1
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('b1','b7'),required=True);ap.add_argument('--length',type=int,default=384)
args=ap.parse_args()
with torch.no_grad():
    a=Q.setup(args.length);d=a['d']
    if args.kind=='b1':
        old=BoundB1(d,a['dy'],a['s'],132,2,'dual_ln_prefetch');ref=tuple(x.clone() for x in old())
        make=lambda s:P.B1(d,a['dy'],a['s'][0].saved_tensors[11],splits=s)
        splits=(20,24,28,32,36,40);names=('dg','dwg','dt','dgo','dbo','dwp');limits=(0,5e-4,2e-5,5e-6,5e-6,5e-4)
    else:
        ref=Q.baseline(a);make=lambda s:P.B7(d,a['dy'],a['dl'],a['dr'],a['dg'],splits=s)
        splits=(4,6,8,10,12,14);names=('dx','dwl','dwlg','dwr','dwrg','dgi','dbi');limits=(2e-5,5e-4,5e-4,5e-4,5e-4,5e-6,5e-6)
    records=[]
    for splits in splits:
        p=make(splits);out=p();torch.cuda.synchronize()
        es={name:dict(relative_l2=Q.rel(x,y),finite=bool(torch.isfinite(x).all()),limit=lim) for name,x,y,lim in zip(names,out,ref,limits)}
        valid=all(v['finite'] and v['relative_l2']<=v['limit'] for v in es.values())
        print('CHECK',splits,es,flush=True)
        if valid:
            g,out=Q.capture_outputs(p);t=Q.pool([Q.paired({'new':g},iterations=100) for _ in range(2)])['new']['median_us']
            records.append(dict(splits=splits,median_us=t,errors=es));print('TIME',splits,t,flush=True)
            del g,out
        else:records.append(dict(splits=splits,rejected=True,errors=es))
        del p;gc.collect()
    valid=[r for r in records if not r.get('rejected')]
    result=dict(kind=args.kind,L=args.length,winner=min(valid,key=lambda x:x['median_us']),candidates=records)
    (R/('tune-%s-L%d.json'%(args.kind,args.length))).write_text(json.dumps(result,indent=2))
    print('WINNER',result['winner'],flush=True)
