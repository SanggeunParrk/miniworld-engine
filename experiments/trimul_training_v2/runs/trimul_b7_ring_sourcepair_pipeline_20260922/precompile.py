import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent
T=SimpleNamespace(_upstream=lambda:R.parent/'trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5')
tree=ast.parse((R/'plan.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='L')
fn.body=fn.body[:stop];cls.body=[fn];exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'compile-only','exec'))
os.environ['B7_CONSUMERS']='10'
for mode in map(int,sys.argv[1:] or ['0','1','2','3']):Plan(None,None,None,None,None,None,mode=mode)
