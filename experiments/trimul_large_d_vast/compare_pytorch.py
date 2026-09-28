"""Paired PyTorch reference/native graphs, including all parameter gradients.

Reference is fixture.ref (only torch/F operations), not the engine Triton path.
Compiler-generated Triton is an implementation detail of torch.compile/Inductor.
No production dispatch, numerical tolerance or native candidate is changed.
"""
import argparse
import gc
import inspect
import json
from short_common import *
from selected_vast import make_plan

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, choices=(128, 256, 384, 512), required=True)
p.add_argument('--length', type=int, choices=(384, 768), required=True)
p.add_argument('--mode', choices=('compiled', 'eager'), default='compiled')
p.add_argument('--with-backward', action='store_true',
               help='Optional retained-forward BWD diagnostic; full F+B is the main comparison')
a = p.parse_args()
leaves, dy, mask, ds, ref, _, names = setup(a.width, a.length)
import torch._functorch.config as fc
fc.donated_buffer = False
dest = OUT / f'pytorch-{a.mode}-D{a.width}-L{a.length}.json'
r = dict(D=a.width, L=a.length, mode=a.mode, complete=False, scopes={},
         with_backward=a.with_backward,
         gpu=torch.cuda.get_device_name(), gpu_index=os.environ.get('CUDA_VISIBLE_DEVICES'),
         torch_version=torch.__version__, cuda_version=torch.version.cuda,
         native_selection='installed_d128' if a.width == 128 else 'selected_vast.make_plan',
         input_splits=4 if (a.width, a.length) == (512, 384) else None,
         reference_source=inspect.getsource(ref),
         source_sha256={name: sha(Path(__file__).parent / name) for name in
                        ('compare_pytorch.py', 'short_common.py', 'selected_vast.py',
                         'input_split.py', 'runtime.py')})
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

def flatten(value, scope):
    return (value[0], *value[1]) if scope == 'full' else tuple(value)

def reference_pass(errors):
    # Existing independent BF16 mathematical-reference gates. These are distinct
    # from the unchanged strict native-to-native candidate qualification gates.
    return all(v < (.005 if n == 'y' else .01) for n, v in errors.items())

with torch.no_grad(), T.native_context(leaves[0].device):
    plan = Native128() if a.width == 128 else make_plan(leaves, mask, ds, dy)
    ny, ng = plan()
    expected = [t.clone() for t in (ny, *ng)]
    tls = tuple(t.detach().clone().requires_grad_(True) for t in leaves)
    reference = (torch.compile(ref, fullgraph=True, dynamic=False,
                              options={'triton.cudagraphs': False})
                 if a.mode == 'compiled' else ref)
    def pfwd():
        with torch.enable_grad():
            return reference(*tls, mask, ds)
    def pfull():
        with torch.enable_grad():
            y = pfwd()
            return y, torch.autograd.grad(y, tls, dy)
    def pbwd():
        with torch.enable_grad():
            return torch.autograd.grad(sy, tls, dy, retain_graph=True)

    print('WARMUP', a.mode, a.width, a.length, flush=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sy = pfwd() if a.with_backward else None
        py, pg = pfull()
    torch.cuda.current_stream().wait_stream(stream)
    wanted = [t.clone() for t in (py, *pg)]
    r['reference_errors'] = {n: error(x, y) for n, x, y in zip(names, expected, wanted)}
    save()
    assert reference_pass(r['reference_errors']), r['reference_errors']
    del expected, ny, ng, py, pg
    gc.collect()

    def capture_reference(fn):
        # Keep reference warmup and capture on one stream. The optional retained
        # forward diagnostic additionally requires its original forward stream.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = fn()
        graph.replay()
        torch.cuda.synchronize()
        return graph, outputs

    scopes = [('backward', plan.backward, pbwd), ('full', plan, pfull)] if a.with_backward else [('full', plan, pfull)]
    for scope, native, baseline in scopes:
        gp, po = capture_reference(baseline)
        actual = flatten(po, scope)
        target = wanted if scope == 'full' else wanted[1:]
        graph_errors = [error(x, y) for x, y in zip(actual, target)]
        r.setdefault('reference_graph_checks', {})[scope] = graph_errors
        save()
        assert max(graph_errors) < 5e-6, graph_errors
        gn, no = capture(native)
        row = dict(times=paired({'pytorch': gp, 'native': gn}, 75),
                   reference_graph_errors=graph_errors)
        row['speedup'] = row['times']['pytorch']['median_us'] / row['times']['native']['median_us']
        r['scopes'][scope] = row
        save()
        print('PYTORCH', a.mode, a.width, a.length, scope,
              {k: v['median_us'] for k, v in row['times'].items()}, row['speedup'], flush=True)

        if scope == 'full':
            # Verify graph liveness with changed input, weight, upstream gradient,
            # mask and dropout. Both graphs retain their original input pointers.
            for native_leaf, ref_leaf in zip(leaves[:7], tls[:7]):
                native_leaf.mul_(0.875)
                ref_leaf.copy_(native_leaf)
            leaves[0].add_(0.125)
            tls[0].copy_(leaves[0])
            dy.mul_(-0.75)
            mask.copy_(mask.roll(1, dims=1))
            ds.mul_(0.75)
            gp.replay()
            gn.replay()
            torch.cuda.synchronize()
            changed_errors = {n: error(x, y) for n, x, y in
                              zip(names, flatten(no, scope), flatten(po, scope))}
            r['changed_graph_errors'] = changed_errors
            assert reference_pass(changed_errors), changed_errors
            # The reference graph must also match a fresh call on changed data.
            fresh = pfull()
            replay_errors = {n: error(x, y) for n, x, y in
                             zip(names, flatten(po, scope), flatten(fresh, scope))}
            r['changed_reference_replay_errors'] = replay_errors
            assert max(replay_errors.values()) < 5e-6, replay_errors
            del fresh
        del gp, gn, po, no, actual, target
        if scope == 'backward':
            sy = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    r['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    r['loaded_cublas'] = sorted({line.split()[-1] for line in open('/proc/self/maps') if 'libcublas' in line})
    r['complete'] = True
    save()
