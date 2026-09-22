"""Strict per-output correctness, dropout, empty/ragged CTA and graph tests."""
import argparse
from dual_experiment import *
from miniworld_engine import settings

settings.configure(engine_backend='triton', trimul_sm90_kernels=frozenset(), autotune_miss_cap=3)
NAMES = ('dg', 'dWg', 'dtri', 'dgamma', 'dbeta', 'dWp')
TOLERANCES = (0., 5e-4, 2e-5, 5e-6, 5e-6, 5e-4)


def check(outputs, reference):
    result = {}
    for name, tol, a, b in zip(NAMES, TOLERANCES, outputs, reference):
        assert torch.isfinite(a).all(), f'{name}: nonfinite'
        error = rel(a, b)
        # Equal FP values alone do not distinguish signed zero. Compare BF16 bits.
        exact = bool(torch.equal(a.view(torch.int16), b.view(torch.int16))) if name == 'dg' else None
        result[name] = dict(relative_l2=error, tolerance=tol, bit_exact=exact)
        assert error <= tol and exact is not False, (name, result[name])
    return result


def change_inputs(d, dy, dropout, seed):
    torch.manual_seed(seed)
    dy.normal_()
    scale = (torch.rand(d['ds'].shape, device=dy.device) >= dropout).to(d['ds'].dtype)
    d['ds'].copy_(scale / (1 - dropout))


def run_case(source, n, count, part, dropout):
    d, dy, saved = data(n)
    seed = 20260920 + n + count + part + int(dropout * 100)
    print('CASE', dict(source=source, L=n, count=count, part=part, dropout=dropout, seed=seed), flush=True)
    change_inputs(d, dy, dropout, seed)
    p = Experiment(d, dy, saved, count=count, part=part, source=source)
    reference = baseline(d, dy, saved)
    errors = [check(p(), reference)]
    assert not torch.count_nonzero(p.workspace[-1]).item()
    # The plan, inputs and output addresses stay fixed across graph replays.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            p()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        p()
    for replay in range(2):
        change_inputs(d, dy, dropout, seed + replay + 1)
        reference = baseline(d, dy, saved)
        graph.replay()
        torch.cuda.synchronize()
        errors.append(check(p.outputs, reference))
        assert not torch.count_nonzero(p.workspace[-1]).item(), 'Counter leaked across replay'
    record = dict(source=source, L=n, count=count, part=part, dropout=dropout,
                  forward_seed=341, gradient_seeds=[seed, seed+1, seed+2],
                  counters_zero=True, errors=errors)
    print('CHECK', json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--source', default='dual')
    a.add_argument('--lengths', type=int, nargs='+', default=[64, 384, 768])
    a.add_argument('--counts', type=int, nargs='+', default=[66, 132])
    a.add_argument('--parts', type=int, nargs='+', default=[1, 2])
    a.add_argument('--dropouts', type=float, nargs='+', default=[0., .25])
    args = a.parse_args()
    records = []
    with torch.no_grad():
        for n in args.lengths:
            for count in args.counts:
                for part in args.parts:
                    for dropout in args.dropouts:
                        records.append(run_case(args.source, n, count, part, dropout))
                        (R / f'{args.source}-strict-validation.json').write_text(json.dumps(records, indent=2))
