from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;rows=[]
old='    rk=tl.arange(0,BLOCK_K);ag=tl.zeros((BLOCK_M1,BLOCK_N),tl.float32)'
new='    # Cast before k * stride: packed bidirectional gradients exceed 2^31 elements.\n    rk=tl.arange(0,BLOCK_K).to(tl.int64);ag=tl.zeros((BLOCK_M1,BLOCK_N),tl.float32)'
for name in ('miniworld-engine-k1k3','miniworld-engine-tbwd'):
 p=Path('/home/psk6950')/name/'src/miniworld_engine/kernels/trimul_inproj/triton/backward_fused.py';s=p.read_text();assert s.count(old)==1 or new in s
 before=hashlib.sha256(s.encode()).hexdigest();s=s.replace(old,new);p.write_text(s);rows.append(dict(path=str(p),before=before,after=hashlib.sha256(s.encode()).hexdigest()))
(R/'installed-offset-fix.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
