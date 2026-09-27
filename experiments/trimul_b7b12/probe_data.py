"""Same plan (WidePlan, C = 128), two data sources: synthetic and the real saved tensors."""
import sys, torch
from wide_plan import WidePlan
from front_core import capture
which = sys.argv[1] if len(sys.argv) > 1 else 'synthetic'
with torch.no_grad():
    if which == 'real':
        from front_core import setup as osetup
        o = osetup(384)
        a = dict(n=384, c=128, x=o['d']['x'].reshape(-1, 128), xn=o['xn'], mu=o['mu'], rs=o['rs'],
                 gamma=o['d']['gi'], pre=o['pre'], dl=o['dl'], dr=o['dr'], dg=o['dg'],
                 dy=o['dy'].reshape(-1, 128), wl=o['wl'], wlg=o['wlg'], wr=o['wr'], wrg=o['wrg'],
                 wg=o['wg'], mask=o['mask'])
    else:
        from bench_front_wide import setup
        a = setup(384, 128)
    src = 'front_prefetch_lnpair_storepipe' if len(sys.argv) > 2 else 'front_wideC'
    print('source', src, flush=True)
    p = WidePlan(a, 128, source=src)
    p(); torch.cuda.synchronize(); print('eager ok', flush=True)
    g = capture(p); print('captured', flush=True)
    for i in range(600):
        g.replay()
        if i % 200 == 0:
            torch.cuda.synchronize(); print('replay', i, flush=True)
    torch.cuda.synchronize(); print('OK', which, flush=True)
