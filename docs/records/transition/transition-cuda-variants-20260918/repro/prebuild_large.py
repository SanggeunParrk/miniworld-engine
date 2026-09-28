import concurrent.futures,json
from pathlib import Path
from large_candidates import candidates
from miniworld_engine.kernels.transition.cuda.variants import extension
root=Path(__file__).parent
jobs=[(v,d,c) for v in ('streamed_k','full_k') for d in (384,512) for c in candidates(v,d)]
def run(t):
 v,d,c=t
 try:
  e=extension(v,d,c);return dict(variant=v,D=d,config=c,resources=e.resources(),status='built')
 except Exception as e:return dict(variant=v,D=d,config=c,status='failed',error=str(e)[-5000:])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 rows=list(pool.map(run,jobs))
(root/'builds-large.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
