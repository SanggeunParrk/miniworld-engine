from pathlib import Path
import importlib.util,sys,json,torch,hashlib
R=Path(__file__).resolve().parent

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
F=load('entry_fixed_policy',R/'policy.py');C=load('entry_current',R.parent/'trimul_training_current.py')
with torch.no_grad():
 a=F.Q.setup(384);fixed=F.Fixed(a);current=C.Training(a)
 y,g=fixed();ref=(y.clone(),tuple(t.clone() for t in g));y,g=current()
 assert all(torch.equal(x,z) for x,z in zip((y,*g),(ref[0],*ref[1])))
 oldhash=hashlib.sha256(Path(fixed.p7.k.unit.cubin_path).read_bytes()).hexdigest();newhash=hashlib.sha256(Path(current.p7.k.unit.cubin_path).read_bytes()).hexdigest();assert oldhash==newhash
 (R/'current_entry.json').write_text(json.dumps(dict(all_outputs_bit_exact=True,cubin_sha256=newhash,current_entry_sha256=hashlib.sha256((R.parent/'trimul_training_current.py').read_bytes()).hexdigest()),indent=2));print('CURRENT ENTRY PASS',newhash,flush=True)
