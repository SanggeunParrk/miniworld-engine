"""Verify the reviewable explicit factory actually selects split4 and replays live inputs."""
from pathlib import Path
import sys,json,hashlib
root=Path('/workspace/experiments/trimul-large-d/runs');pre=root/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(pre));ns={'__file__':str(pre/'check.py')}
exec(compile((pre/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0],str(pre/'check.py'),'exec'),ns)
import torch
from selected_vast import make_plan
from validate_engine import capture,error
leaves,dy,mask,ds,ref,triton,names=ns['setup'](512,384)
with torch.no_grad(),ns['T'].native_context(leaves[0].device):
 plan=make_plan(leaves,mask,ds,dy)
 assert plan.input_splits==4 and plan.input_dw.index==0
 assert plan.input_dw.lib.cublasLtGetVersion()==120804
 y,g=plan();assert all(torch.isfinite(t).all() for t in (y,*g))
 graph,outputs=capture(plan)
 leaves[0].mul_(.91);leaves[1].mul_(1.02);dy.mul_(.87);mask.copy_(mask.roll(1,1));ds.copy_(ds.roll(1,2))
 y,g=plan();expected=[t.clone() for t in (y,*g)]
 graph.replay();torch.cuda.synchronize()
 errors={n:error(u,v) for n,u,v in zip(names,(outputs[0],*outputs[1]),expected)}
 assert max(errors.values())<5e-6,errors
 record={'passed':True,'width':512,'length':384,'input_splits':plan.input_splits,'lt_index':plan.input_dw.index,'graph_errors':errors,'selected_sha256':hashlib.sha256(Path(__file__).with_name('selected_vast.py').read_bytes()).hexdigest(),'input_split_sha256':hashlib.sha256(Path(__file__).with_name('input_split.py').read_bytes()).hexdigest()}
 Path('/workspace/vast-results/trimul-large-d/selected-entry-check.json').write_text(json.dumps(record,indent=2))
 print('SELECTED_ENTRY_PASS',max(errors.values()),flush=True)
