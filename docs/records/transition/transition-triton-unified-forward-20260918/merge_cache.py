import json
from pathlib import Path
root=Path(__file__).parent
op='transition_b2b_residual_triton';name='NVIDIA H100 80GB HBM3 (sm90).json'
parts=[json.loads((root/f'cache-D{d}'/op/name).read_text().replace('"bfloat16|','"bfloat16+float32|')) for d in (128,256)]
merged=parts[0]
for other in parts[1:]:
 for field in ('op_identity','config_space_hash','env_identity','key_scheme','build_rev'):
  assert merged[field]==other[field],field
 for field in ('entries','entry_grids','measurements','grids'):
  merged[field].update(other[field])
assert len(merged['entries'])==8
out=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src/miniworld_engine/autotune/data')/op/name
out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(merged,indent=2,sort_keys=True)+'\n')
print('Merged',len(merged['entries']),'shape/mode entries',out)
