from pathlib import Path
from functools import lru_cache
import json,importlib.util
@lru_cache(None)
def load(path):
 spec=importlib.util.spec_from_file_location(Path(path).stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def old_extension(v,d,c):
 r=Path(__file__).parent.parent/'transition_cuda_variants_20260918';x=json.loads((r/f'tune-{v}-D{d}.json').read_text())
 for k in ('best_forward','best_backward'):
  if x[k]['config']==c:
   path=x[k]['extension'];assert '9502ec6a3ef5' in path,path
   return load(path)
 raise ValueError('No archived baseline for config')
