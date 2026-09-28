"""Transition pair-shape qualification and paired full-training CUDA graphs.

Performance reference: actual PyTorch module, compiled with Inductor. Engine
Triton is used only for the existing numerical acceptance contract. Every step
includes a fresh forward and gradients for the input and all five parameters.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from miniworld_engine import settings
from miniworld_engine.modules import Transition


def rel(got, want):
    got, want = got.detach().float(), want.detach().float()
    return float((got - want).norm() / want.norm().clamp_min(1e-20))


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def paired(graphs, repeats):
    times = {name: [] for name in graphs}
    names = list(graphs)
    for i in range(repeats + 10):
        for name in names[i % len(names):] + names[:i % len(names)]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            if i >= 10:
                times[name].append(start.elapsed_time(end))
    return {name: {"median_ms": statistics.median(ts), "min_ms": min(ts),
                   "max_ms": max(ts), "samples_ms": ts} for name, ts in times.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, choices=(128, 256, 384, 512), required=True)
    parser.add_argument('--length', type=int, choices=(384, 768), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=75)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d, length = args.width, args.length
    torch.manual_seed(729 + d + length)
    torch.backends.cuda.matmul.allow_tf32 = False
    settings.configure(engine_backend='auto', transition_residual_fusion=True,
                       transition_fused_sm90a=True)
    os.environ['MINIWORLD_TRANSITION_WIDE_SAVE_H'] = '0'
    native = Transition(d, n=4, implementation='miniworld').cuda().bfloat16()
    with torch.no_grad():
        for name, p in native.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -.5)
            elif name == 'ln_in.weight':
                p.copy_(1 + .2 * torch.randn_like(p))
            else:
                p.normal_(std=.2)
    x = torch.randn((1, length, length, d), device='cuda', dtype=torch.bfloat16).requires_grad_()
    dy = torch.randn_like(x)
    names = ['y', 'dx', *dict(native.named_parameters())]
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root / 'src/miniworld_engine/modules/transition/module.py']
    files += sorted((root / 'src/miniworld_engine/kernels/transition').rglob('*.py'))
    files += sorted((root / 'src/miniworld_engine/kernels/transition/cuda').rglob('*.cu'))
    files += sorted((root / 'src/miniworld_engine/kernels/transition/cuda').rglob('*.cuh'))
    result = dict(D=d, L=length, shape=list(x.shape), dtype=str(x.dtype), expansion=4,
                  complete=False, scope='forward + input and all parameter gradients',
                  gpu=torch.cuda.get_device_name(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  source_sha256={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    dest = args.out / f'D{d}-L{length}.json'

    def save():
        dest.write_text(json.dumps(result, indent=2))

    def make_step(module, xx):
        leaves = (xx, *module.parameters())
        def step():
            y = module(xx)
            return (y, *torch.autograd.grad(y, leaves, dy))
        return step

    step = make_step(native, x)
    print('START', d, length, 'sanitize', args.sanitize, flush=True)
    # Direct module entry must really select the native kernel before compilation.
    from miniworld_engine.kernels.transition.cuda import fused_sm90a, fused_wide_sm90a
    target = fused_sm90a if d == 128 else fused_wide_sm90a
    entry = 'transition_fused_sm90a' if d == 128 else 'transition_wide_sm90a'
    original, calls = getattr(target, entry), []
    def observed(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    setattr(target, entry, observed)
    initial = step()
    setattr(target, entry, original)
    assert calls, 'module silently fell back instead of using native Transition'
    result['dispatch_entry'] = f'{target.__name__}.{entry}'
    expected = tuple(t.detach().clone() for t in initial)
    del initial
    save()

    if not args.sanitize:
        # Use the repository's pre-existing native-versus-Triton tolerances.
        settings.configure(engine_backend='triton')
        triton = step()
        errors = {n: rel(g, w) for n, g, w in zip(names, expected, triton)}
        tolerances = {'y': .004, 'dx': .0005 if d == 128 else .005}
        result['triton_correctness_errors'] = errors
        save()
        assert all(v < tolerances.get(n, .002 if d == 128 else .006) for n, v in errors.items()), errors
        del triton
        settings.configure(engine_backend='auto')

    graphs, outputs, steps = {}, {}, {}
    variants = [0, 1] if d >= 384 else [0]
    for saved_h in variants:
        os.environ['MINIWORLD_TRANSITION_WIDE_SAVE_H'] = str(saved_h)
        key = 'save_h' if saved_h else 'native'
        if args.sanitize:
            current = step
        else:
            compiled = torch.compile(native, fullgraph=True, dynamic=False,
                                     options={'triton.cudagraphs': False})
            current = make_step(compiled, x)
        graphs[key], outputs[key] = capture(current)
        steps[key] = current
        errors = {n: rel(g, w) for n, g, w in zip(names, outputs[key], expected)}
        result.setdefault('native_graph_errors', {})[key] = errors
        save()
        for n, g, w in zip(names, outputs[key], expected):
            if n in ('ln_in.weight', 'ln_in.bias') and d >= 384:
                assert rel(g, w) < 1e-5, (key, n, rel(g, w))
            elif n == 'squeeze.weight' and saved_h:
                assert rel(g, w) < .002, (key, n, rel(g, w))
            else:
                assert torch.equal(g, w), (key, n, rel(g, w))
    del expected

    if not args.sanitize:
        reference = Transition(d, n=4, implementation='pytorch').cuda().bfloat16()
        reference.load_state_dict(native.state_dict())
        rx = x.detach().clone().requires_grad_()
        compiled_ref = torch.compile(reference, fullgraph=True, dynamic=False,
                                     options={'triton.cudagraphs': False})
        steps['pytorch'] = make_step(compiled_ref, rx)
        graphs['pytorch'], outputs['pytorch'] = capture(steps['pytorch'])
        fresh = steps['pytorch']()
        errors = {n: rel(g, w) for n, g, w in zip(names, outputs['pytorch'], fresh)}
        result['pytorch_graph_errors'] = errors
        assert max(errors.values()) < 5e-6, errors
        result['pytorch_vs_native_errors'] = {n: rel(g, w) for n, g, w in zip(names, outputs['native'], outputs['pytorch'])}
        del fresh
        result['times'] = paired(graphs, args.repeats)
        print('TIMES', d, length, {k: v['median_ms'] for k, v in result['times'].items()}, flush=True)
        save()

    # Changed inputs, all weights, LN affine, and upstream gradient at fixed pointers.
    with torch.no_grad():
        x.mul_(.875).add_(.125)
        dy.mul_(-.75)
        for p in native.parameters():
            p.mul_(.875)
        if not args.sanitize:
            rx.copy_(x)
            reference.load_state_dict(native.state_dict())
    for key, graph in graphs.items():
        os.environ['MINIWORLD_TRANSITION_WIDE_SAVE_H'] = str(int(key == 'save_h'))
        graph.replay()
        torch.cuda.synchronize()
        fresh = steps[key]()
        errors = {n: rel(g, w) for n, g, w in zip(names, outputs[key], fresh)}
        result.setdefault('changed_graph_errors', {})[key] = errors
        save()
        assert max(errors.values()) < 1e-5, (key, errors)
        del fresh
    if d >= 384:
        result['changed_save_h_errors'] = errors = {n: rel(g, w) for n, g, w in zip(names, outputs['save_h'], outputs['native'])}
        assert all(v < (.002 if n == 'squeeze.weight' else 1e-5) for n, v in errors.items()), errors
        result['retained_h_bytes_per_layer'] = length * length * 4 * d * 2
    if not args.sanitize:
        result['changed_pytorch_vs_native_errors'] = {n: rel(g, w) for n, g, w in zip(names, outputs['native'], outputs['pytorch'])}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            graphs['native'].replay()
            torch.cuda.synchronize()
        result['native_cuda_kernel_names'] = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
        assert result['native_cuda_kernel_names']
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['complete'] = True
    save()
    print('PASS', dest, flush=True)


if __name__ == '__main__':
    main()
