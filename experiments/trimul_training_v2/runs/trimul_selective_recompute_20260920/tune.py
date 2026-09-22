"""Retune both reduced epilogues within existing saved-kernel config spaces."""
from pathlib import Path
import argparse,json,torch
import adapter_selective as S
import compare_cueq_training as Q
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
with torch.no_grad():
    d=S.C.setup(n)
    k3=json.loads((S.A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner']
    y,ss=S.F.forward(d,k3=k3);_,kept=S.forward(d)
    xx=ss[0].saved_tensors
    xn=xx[0]
    refs={'front':(xn,ss[1],ss[2],xx[8]),'output':(xx[12],xx[13],xx[14],xx[16],xx[15])}
    choices={'front':list(S.C.S.front_candidates()),'output':list(S.T.candidates())}
    result=dict(L=n,complete=False,scope='Same configuration spaces as existing saved K1/K3, pruned by their original resource bounds; each accepted candidate checked bit-exact.')
    for kind,configs in choices.items():
        records=[]
        print('TUNE',kind,len(configs),flush=True)
        def call(cfg):return S.front(d,kept[0],cfg) if kind=='front' else S.output(d,kept[1],xn,cfg)
        for cfg in configs:
            out=call(cfg);torch.cuda.synchronize()
            exact=[bool(torch.equal(a,b)) for a,b in zip(out,refs[kind])]
            assert all(exact),(kind,cfg,exact)
            graph,out=Q.capture_outputs(lambda:call(cfg));graph.replay();torch.cuda.synchronize()
            assert all(torch.equal(a,b) for a,b in zip(out,refs[kind])),cfg
            time=Q.paired({'candidate':graph},iterations=50)['candidate']['median_us']
            records.append(dict(config=cfg,median_us=time,bit_exact=True))
            print('CONFIG',kind,cfg,time,flush=True)
            del graph,out
        top=sorted(records,key=lambda x:x['median_us'])[:4]
        graphs={str(i):Q.capture_outputs(lambda cfg=r['config']:call(cfg))[0] for i,r in enumerate(top)}
        # Every graph owns output storage through its private graph pool. The
        # input d/kept/xn stays strongly referenced for all paired replays.
        times=Q.pool([Q.paired(graphs,iterations=100) for _ in range(3)])
        for i,r in enumerate(top):r['final_median_us']=times[str(i)]['median_us']
        winner=min(top,key=lambda x:x['final_median_us'])
        result[kind]=dict(winner=winner['config'],median_us=winner['final_median_us'],candidates=records,finalists=top)
        (R/('tuning-L%d.json'%n)).write_text(json.dumps(result,indent=2))
        print('WINNER',kind,winner,flush=True)
        del graphs
    result['complete']=True
    (R/('tuning-L%d.json'%n)).write_text(json.dumps(result,indent=2))
