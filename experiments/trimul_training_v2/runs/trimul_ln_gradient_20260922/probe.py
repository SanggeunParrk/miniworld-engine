from pathlib import Path
import importlib.util,sys,torch,hashlib,json
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('ln_fixed_probe',R/'policy.py');F=importlib.util.module_from_spec(s);sys.modules[s.name]=F;s.loader.exec_module(F)
with torch.no_grad():
 a=F.Q.setup(384);m=F.Fixed(a);out=m();ref=tuple(t.clone() for t in m.p7.outputs)
 for t in m.p7.outputs:t.fill_(float('nan'))
 got=m.p7();torch.cuda.synchronize()
 assert all(torch.equal(x,y) for x,y in zip(got,ref))
 assert int(m.p7.flags.count_nonzero())==0
 p=Path(m.p7.k.unit.cubin_path);print('SANITIZER_CUBIN',str(p),hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
 print('PASS same-cubin repeat / poison / flag reset',flush=True)
