"""Matched CUDA Graph timings and activity traces for the length-scaling question.

Activity totals identify kernels, not hardware bottlenecks; NCU is unavailable.
The full-workload timing is measured separately from profiling.
"""
import argparse
from collections import defaultdict
import gc
import json
from short_common import *

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, required=True)
p.add_argument('--length', type=int, required=True)
a = p.parse_args()
leaves, dy, mask, ds, ref, triton, names = setup(a.width, a.length)
import torch._functorch.config as fc
fc.donated_buffer = False
from miniworld_engine import settings
settings.configure(engine_backend='triton', trimul_sm90_kernels=(), autotune_miss_cap=24)
r = dict(D=a.width, L=a.length, complete=False, scopes={}, script_sha256=sha(__file__))
dest = OUT / f'length-profile-D{a.width}-L{a.length}.json'
def save():
    dest.write_text(json.dumps(r, indent=2))

class Native128:
    def __init__(self):
        from miniworld_engine.kernels.trimul_inproj.cuda import h100_training
        self.module = h100_training
    def forward(self):
        self.y, *self.saved = self.module.forward(list(leaves), mask, ds)
        return self.y
    def backward(self):
        g = self.module.backward(list(leaves), mask, ds, self.saved, dy)
        return [g[0], *(w.t() for w in g[1].unbind()), *g[2:]]
    def __call__(self):
        return self.forward(), self.backward()

with torch.no_grad(), T.native_context(leaves[0].device):
    plan = Native128() if a.width == 128 else training(a.width)(leaves, mask, ds, dy)
    ny, ng = plan()
    expected = [t.clone() for t in (ny, *ng)]
    tls = tuple(t.detach().clone().requires_grad_(True) for t in leaves)
    compiled = torch.compile(triton, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
    def tfwd():
        with torch.enable_grad():
            return compiled(*tls, mask, ds)
    def tfull():
        with torch.enable_grad():
            y = tfwd()
            return y, torch.autograd.grad(y, tls, dy)
    print('WARMUP', a.width, a.length, flush=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sy = tfwd()
        ty, tg = tfull()
    torch.cuda.current_stream().wait_stream(stream)
    errors = {n: error(x, y) for n, x, y in zip(names, expected, (ty, *tg))}
    r['triton_errors'] = errors
    save()
    assert all(v < (.005 if n == 'y' else .01) for n, v in errors.items()), errors
    def tbwd():
        with torch.enable_grad():
            return torch.autograd.grad(sy, tls, dy, retain_graph=True)
    for scope, native, baseline in [('backward', plan.backward, tbwd), ('full', plan, tfull)]:
        graphs = {}
        outputs = {}
        for label, fn in [('triton', baseline), ('native', native)]:
            graphs[label], outputs[label] = capture(fn)
        row = {'times': paired(graphs, 51), 'kernels': {}}
        for label, graph in graphs.items():
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(OUT / f'length-trace-D{a.width}-L{a.length}-{scope}-{label}.json'))
            acc, counts = defaultdict(float), defaultdict(int)
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    acc[e.name] += e.device_time_total / 3
                    counts[e.name] += 1
            row['kernels'][label] = [dict(name=n, us=us, calls=counts[n] / 3)
                                     for n, us in sorted(acc.items(), key=lambda p: -p[1])]
        row['speedup'] = row['times']['triton']['median_us'] / row['times']['native']['median_us']
        r['scopes'][scope] = row
        save()
        print('RESULT', a.width, a.length, scope, row['speedup'],
              {k: v['median_us'] for k, v in row['times'].items()}, flush=True)
        del graphs, outputs, graph
        if scope == 'backward':
            sy = None
        gc.collect()
        torch.cuda.empty_cache()
    r['complete'] = True
    save()
