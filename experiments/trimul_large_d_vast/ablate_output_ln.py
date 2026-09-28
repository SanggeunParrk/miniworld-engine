"""Causal diagnostic: only change Triton's output-LN backward dispatch.

Frozen graphs retain the default and forced-atomic paths in the same process.
This is not a production change or a new native-kernel optimization.
"""
import argparse
from collections import defaultdict
import json
from short_common import *

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, required=True)
a = p.parse_args()
leaves, dy, mask, ds, _, triton, names = setup(a.width, 768)
import torch._functorch.config as fc
fc.donated_buffer = False
from miniworld_engine import settings
settings.configure(engine_backend='triton', trimul_sm90_kernels=(), autotune_miss_cap=24)
compiled = torch.compile(triton, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
def full():
    with torch.enable_grad():
        y = compiled(*leaves, mask, ds)
        return (y, *torch.autograd.grad(y, leaves, dy))
r = dict(D=a.width, L=768, complete=False, diagnostic_only=True,
         script_sha256=sha(__file__), configs={})
dest = OUT / f'ln-ablation-D{a.width}-L768.json'
def save():
    dest.write_text(json.dumps(r, indent=2))
graphs, values = {}, {}
with torch.no_grad():
    for label, setting in [('default', None), ('atomic', 'atomic')]:
        settings.configure(layernorm_out_bwd_path=setting)
        print('WARMUP', a.width, label, flush=True)
        graphs[label], outputs = capture(full)
        values[label] = [t.clone() for t in outputs]
        from miniworld_engine.kernels.layernorm_linear.triton import mmajor_bwd
        k = mmajor_bwd._ln_bwd_persistent_jit if label == 'default' else mmajor_bwd._ln_bwd_kernel
        r['configs'][label] = str(getattr(k, 'best_config', 'not available'))
    r['errors'] = {n: error(x, y) for n, x, y in zip(names, values['default'], values['atomic'])}
    save()
    # These existing Triton paths use different reduction/arithmetic orders.
    # Preserve the failed strict gate explicitly: this intervention diagnoses
    # timing and must not be promoted as a qualified equivalent replacement.
    r['strict_equivalent'] = strict(r['errors'])
    assert max(r['errors'].values()) < .001, r['errors']
    r['times'] = paired(graphs, 75)
    r['speedup'] = r['times']['default']['median_us'] / r['times']['atomic']['median_us']
    r['kernels'] = {}
    for label, graph in graphs.items():
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
        acc = defaultdict(float)
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                acc[e.name] += e.device_time_total / 3
        r['kernels'][label] = [dict(name=n, us=us) for n, us in sorted(acc.items(), key=lambda p: -p[1])]
    r['complete'] = True
    save()
    print('ABLATION', a.width, r['speedup'], {k: v['median_us'] for k, v in r['times'].items()}, r['errors'], flush=True)
