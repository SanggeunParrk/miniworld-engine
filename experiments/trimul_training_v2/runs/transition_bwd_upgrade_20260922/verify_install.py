from harness import *
p=Path('/home/psk6950/miniworld-engine-tbwd/src/miniworld_engine/kernels/transition/cuda/fused_sm90a.py')
s=importlib.util.spec_from_file_location('miniworld_engine.kernels.transition.cuda.transition_upgrade_installed',p);I=importlib.util.module_from_spec(s);sys.modules[s.name]=I;s.loader.exec_module(I)
with torch.no_grad():
 d=fixture();plan=Plan(d,'parallel_transpose');ref=tuple(v.clone() for v in plan())
 # Enter the native extension directly: old and new Python opaque names intentionally match.
 def call():return tuple(I._ext_for(d['x']).transition_fused_bwd(d['dy'],d['x'],d['xn'],d['rs'],d['c1'],d['gamma'],d['wa'],d['wb'],d['ws']))
 es=errors(call(),ref);assert all(v['bit_exact'] for v in es.values());g,out=capture(call);g.replay();torch.cuda.synchronize();assert all(torch.equal(x,y) for x,y in zip(out,ref))
 record=dict(job=os.environ.get('SLURM_JOB_ID'),source=str(p),checks=es,graph_exact=True,complete=True);(R/'installed_validation.json').write_text(json.dumps(record,indent=2));print('INSTALLED_PASS',flush=True)
