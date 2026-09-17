"""Before/after comparison using the production single-direction module harness."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch
import miniworld_engine  # Bind the requested package before the harness adds its src path.
from benchmarks.runners import bench
from miniworld_engine.kernels.trimul_inproj.triton import unidirectional as uni


class NoFabric:
    @staticmethod
    def setup_module(module):
        return module

    @staticmethod
    def backward(tensor, gradient):
        tensor.backward(gradient)


def paired(calls, functions):
    samples = {label: [] for label in calls}
    for round_id in range(12):
        for label in list(calls)[::(-1 if round_id % 2 else 1)]:
            uni.trimul_triton = functions[label]
            fn, grad_to_none = calls[label]
            stats = bench.bench_time(fn, warmup=20, rep=100, grad_to_none=grad_to_none)
            samples[label].append(stats['median_ms'])
    return {label: {'ms': statistics.median(values), 'samples_ms': values}
            for label, values in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--direction', choices=['outgoing', 'incoming'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pytorch', action='store_true')
    parser.add_argument('--graph', action='store_true', help='Supplementary dropout=0 CUDA graph benchmark')
    parser.add_argument('--require-tuned', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).parent / 'unidirectional_before.py'
    spec = importlib.util.spec_from_file_location('unidirectional_before', source)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    new_fn = uni.trimul_triton
    cache_lookups = []
    if args.require_tuned:
        from miniworld_engine.autotune import cache
        from miniworld_engine.autotune.configs import op_of
        required = {'trimul_output_f567_train_triton', 'trimul_input_dual_bwd_triton',
                    'trimul_input_ln_residual_bwd_triton'}
        original_miss, original_subset = cache._miss, cache._cached_subset

        def reject_miss(op, *a, **kw):
            assert op not in required, ('fused kernel cache miss', op, a[:3])
            return original_miss(op, *a, **kw)

        def observe_subset(tuner, configs, nargs, meta):
            answer = original_subset(tuner, configs, nargs, meta)
            op = op_of(tuner.configs)
            if op in required:
                assert answer, op
                cache_lookups.append({'op': op, 'bucket': cache.bucket_of_autotuner(tuner, nargs, meta),
                                      'configs': [cache.as_cfg_dict(c) for c in answer]})
            return answer
        cache._miss, cache._cached_subset = reject_miss, observe_subset
    conf = bench.BenchConfig(target='triangle_multiplication', level='module', mode='training',
                            metric='time', compile=True, cudagraph='manual' if args.graph else 'disabled', precision='bf16-mixed',
                            d_pair=128, n_layers=1, dropout=0 if args.graph else .25, mask_prob=.2,
                            trimul_direction=args.direction, min_seq_len=args.length,
                            max_seq_len=args.length)
    torch.backends.cuda.matmul.allow_tf32 = conf.allow_tf32
    torch.backends.cudnn.allow_tf32 = conf.allow_tf32
    calls, functions, rows = {}, {}, {}
    original_time = bench.bench_time
    label = ''

    def record_time(fn, **kwargs):
        calls[label] = (fn, kwargs.get('grad_to_none', []))
        return original_time(fn, **kwargs)

    bench.bench_time = record_time
    for label in ['before', 'after'] + (['pytorch'] if args.pytorch else []):
        uni.trimul_triton = old.trimul_triton if label == 'before' else new_fn
        functions[label] = uni.trimul_triton
        torch.compiler.reset()
        with bench.forward_stream(conf):
            row = bench.bench_module_triangle_multiplication(
                conf, args.length, 'pytorch' if label == 'pytorch' else 'triton', NoFabric())
        rows[label] = row._asdict()
        assert row.output_rel_frob < .025, rows[label]
        assert row.grad_rel_frob < .025, rows[label]
        print('HARNESS', label, row.value, flush=True)
    uni.trimul_triton = new_fn
    bench.bench_time = original_time
    result = {'length': args.length, 'direction': args.direction,
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'package': miniworld_engine.__file__, 'unidirectional_source': uni.__file__,
              'config': conf.model_dump(), 'harness': rows, 'paired': paired(calls, functions),
              'fused_cache_lookups': cache_lookups,
              'scope': 'One production module, forward+backward, no optimizer; fixed-shape '
                       'torch.compile(dynamic=False); graph/dropout per config; same GPU, '
                       '12 alternating rounds using the harness timer, 100 ms per sample.'}
    result['speedup'] = result['paired']['before']['ms'] / result['paired']['after']['ms']
    args.output.write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps({'paired': result['paired'], 'speedup': result['speedup']}), flush=True)


if __name__ == '__main__':
    main()
