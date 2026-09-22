from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;repo=Path('/home/psk6950/miniworld-engine-tbwd');target=repo/'src/miniworld_engine/kernels/transition/cuda/transition_fused_bwd_sm90a_kernel.cu';old=(R/'transition_fused_bwd_sm90a_kernel.cu').read_bytes();new=(R/'selected/transition_fused_bwd_sm90a_kernel.cu').read_bytes()
assert target.read_bytes() in (old,new),'Target changed independently: do not overwrite'
assert old.split(b'// partw [NDW]')[0]==new.split(b'// partw [NDW]')[0], 'Main kernel changed unexpectedly'
target.write_bytes(new)
record=dict(repo=str(repo),file=str(target),old_sha256=hashlib.sha256(old).hexdigest(),new_sha256=hashlib.sha256(new).hexdigest(),scope='Only partial reducer and its launch grid. Existing forward, main backward and dispatch preserved.')
(R/'installed.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
