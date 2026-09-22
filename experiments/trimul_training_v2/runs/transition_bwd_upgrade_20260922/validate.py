from harness import *
s=importlib.util.spec_from_file_location('miniworld_engine.kernels.transition.cuda.transition_upgrade_selected',R/'selected/fused_sm90a.py');C=importlib.util.module_from_spec(s);sys.modules[s.name]=C;s.loader.exec_module(C)
record=dict(job=os.environ.get('SLURM_JOB_ID'),selection=json.loads((R/'selection.json').read_text()),cases={},times={})
def save():(R/'validation.json').write_text(json.dumps(record,indent=2))
with torch.no_grad():
 for L in (384,768):
  d=fixture(L);orig={k:d[k].clone() for k in ('x','gamma','beta','wa','wb','ws','dy')}
  def call(mod):return mod._bwd_launch(d['dy'],d['x'],d['xn'],d['rs'],d['c1'],d['gamma'],d['wa'],d['wb'],d['ws'])
  graphs={};outs={}
  for n,mod in [('baseline',N),('selected',C)]:graphs[n],outs[n]=capture(lambda mod=mod:call(mod))
  for case in range(12):
   for k,v in orig.items():d[k].copy_(v)
   torch.manual_seed(9190+case)
   if case:
    d['x'].normal_(0,[1.,.01,2.,.2][case%4]);d['dy'].normal_(0,.1 if case%3==0 else 1.)
    for k in ('wa','wb','ws'):d[k].add_(torch.randn_like(d[k])*.001)
    d['gamma'].normal_(1.,.15);d['beta'].normal_(0,.05);d['gamma'][::17]=0
   if case==10:d['x'].zero_()
   if case==11:d['dy'].zero_()
   refresh(d);ref=tuple(v.clone() for v in call(N));eager=tuple(v.clone() for v in call(C));checks=errors(eager,ref)
   graphs['selected'].replay();graphs['selected'].replay();torch.cuda.synchronize();ge=errors(outs['selected'],eager);assert all(v['bit_exact'] for v in ge.values())
   record['cases'][f'{L}-{case}']=dict(errors=checks,graph_eager_exact=True);save();print('PASS',L,case,flush=True)
  for k,v in orig.items():d[k].copy_(v)
  refresh(d);record['times'][str(L)]=paired(graphs,250,5);print('TIME',L,{k:v['median_us'] for k,v in record['times'][str(L)].items()},flush=True);save()
record['complete']=True;save()
