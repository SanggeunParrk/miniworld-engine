import json
from pathlib import Path
root=Path(__file__).parent;op='transition_segmented_b2b_triton';name='NVIDIA H100 80GB HBM3 (sm90).json'
a,b=[json.loads((root/f'cache-D{d}'/op/name).read_text()) for d in (128,256)]
for k in ('op_identity','config_space_hash','env_identity','key_scheme','build_rev'):assert a[k]==b[k],k
for k in ('entries','entry_grids','measurements','grids'):a[k].update(b[k])
assert len(a['entries'])==12
p=root.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/autotune/data'/op/name
p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');print('Merged',len(a['entries']),'entries')
