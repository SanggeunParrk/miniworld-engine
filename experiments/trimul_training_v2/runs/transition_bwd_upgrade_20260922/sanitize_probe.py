from harness import *
s=importlib.util.spec_from_file_location('miniworld_engine.kernels.transition.cuda.transition_upgrade_sanitize',R/'selected/fused_sm90a.py');C=importlib.util.module_from_spec(s);sys.modules[s.name]=C;s.loader.exec_module(C)
with torch.no_grad():
 for L in (384,768):
  d=fixture(L)
  outs=C._bwd_launch(d['dy'],d['x'],d['xn'],d['rs'],d['c1'],d['gamma'],d['wa'],d['wb'],d['ws']);torch.cuda.synchronize();assert all(bool(x.isfinite().all()) for x in outs);print('SANITIZE_PASS',L,flush=True)
