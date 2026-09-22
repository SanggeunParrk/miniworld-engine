import cutlass.cute as cute
import pathlib
original=cute.compile
folder=pathlib.Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/dual_bwd/ptx')
folder.mkdir(exist_ok=True)
def keeping(*args,**kwargs):
 kwargs['options']='--keep-ptx --dump-dir='+str(folder)
 return original(*args,**kwargs)
cute.compile=keeping
exec(pathlib.Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/dual_bwd/smoke.py').read_text())
for p in folder.glob('*.ptx'):
 text=p.read_text()
 print(str(p), 'TMA_LOAD=', text.count('cp.async.bulk.tensor'), 'WGMMA=',text.count('wgmma.mma_async'),flush=True)
