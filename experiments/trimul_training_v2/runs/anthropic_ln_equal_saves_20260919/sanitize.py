from core_saved import *
with torch.no_grad():
 for row in json.loads((R/'results.json').read_text()):
  for n in (64,72):
   d=setup(n);cf=row['configs'];outs=[]
   for f in (True,False):outs.append(front(d['x'],d['w1'],d['mask'],d['gi'],d['bi'],f,tuple(cf['fused' if f else 'split']),tuple(cf['ln'])))
   torch.cuda.synchronize();assert all(torch.equal(a,b) for a,b in zip(*outs))
print('PASS',flush=True)
