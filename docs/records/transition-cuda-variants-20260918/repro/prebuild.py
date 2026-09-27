import concurrent.futures,json,time
from pathlib import Path
from configs import seeds,smem
from miniworld_engine.kernels.transition.cuda.variants import extension
root=Path(__file__).parent
tasks=[(v,d,c) for d in (128,256,384,512) for v in ('streamed_k','full_k') for c in seeds(v,d)]
results=[]
def build(task):
    v,d,c=task;t=time.monotonic();r=dict(variant=v,D=d,config=c)
    if min(smem(v,d,c),smem(v,d,c,True))>232448:
        return dict(r,status='resource_excluded',reason='both directions exceed 232448 bytes',forward_smem=smem(v,d,c),backward_smem=smem(v,d,c,True))
    try:
        e=extension(v,d,c);actual=e.resources()
        assert actual['forward_smem']==smem(v,d,c),(actual,c)
        assert actual['backward_smem']==smem(v,d,c,True),(actual,c)
        r.update(status='built',path=e.__file__,resources=actual)
    except Exception as e:r.update(status='failed',error=str(e)[-5000:])
    r['seconds']=time.monotonic()-t
    return r
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures=[pool.submit(build,t) for t in tasks]
    for f in concurrent.futures.as_completed(futures):
        r=f.result();results.append(r)
        (root/'builds.json').write_text(json.dumps(results,indent=2)+'\n')
        print(len(results),len(tasks),json.dumps(r),flush=True)
