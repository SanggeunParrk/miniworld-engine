import argparse
import gc
import json
from pathlib import Path
import torch
from common import inputs, baseline, identity, wide, _transition_ln_bwd, capture, paired

p = argparse.ArgumentParser()
p.add_argument('--widths', type=int, nargs='+', default=[256, 384, 512])
p.add_argument('--lengths', type=int, nargs='+', default=[384, 768])
args = p.parse_args()
out = Path('.bench/transition-wide-local/profile')
out.mkdir(parents=True, exist_ok=True)
for d in args.widths:
    for length in args.lengths:
        print('START', d, length, flush=True)
        v = inputs(d, length)
        x, gamma, beta, wa, wb, ws, dy = v
        ext = wide._ext_for(x)
        y, xn, rstd, c1, _ = wide._fwd_launch(x, gamma, beta, wa, wb, ws, 1e-5, True)
        wpack, wst, wab = wide._pack(wa, wb, 128), ws.t().contiguous(), torch.cat((wa, wb))
        hid, dab = ext.gate(xn, dy, wpack, wst, True)
        dxn = dab @ wab
        stages = {
            'full_fb': lambda: baseline(v),
            'fwd': lambda: wide._fwd_launch(x, gamma, beta, wa, wb, ws, 1e-5, True),
            'gate': lambda: ext.gate(xn, dy, wpack, wst, True),
            'dws': lambda: wide._mm_f32(dy.t(), hid),
            'dwab': lambda: wide._mm_f32(dab.t(), xn),
            'dx_mm': lambda: dab @ wab,
            'ln_bwd': lambda: _transition_ln_bwd(dxn, x, rstd, c1, gamma),
        }
        if d == 256:
            wt = wab.t().contiguous()
            stages['dxln'] = lambda: ext.dxln(dab, wt, x, dy, gamma, rstd, c1)
        graphs, outputs = {}, {}
        for name, fn in stages.items():
            graphs[name], outputs[name] = capture(fn)
        result = dict(D=d, L=length, identity=identity(), times=paired(graphs, 35))
        # CUDA activity timing of a complete step catches wrapper/copy overhead.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            graphs['full_fb'].replay()
            torch.cuda.synchronize()
        result['activity'] = [{'name':e.name, 'us':e.device_time_total} for e in prof.events()
                              if e.device_type == torch.autograd.DeviceType.CUDA]
        (out / f'D{d}-L{length}.json').write_text(json.dumps(result, indent=2))
        print('TIMES', d, length, {k:round(t['median_ms'], 4) for k,t in result['times'].items()}, flush=True)
        del graphs, outputs, stages, v, x, dy, xn, y, hid, dab, dxn
        gc.collect()
        torch.cuda.empty_cache()
