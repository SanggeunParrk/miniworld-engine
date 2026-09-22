"""Direct, matched bidirectional fwd+bwd: cuEq composition vs current B7-B12 CUDA.

Public cuEq TMU is single-direction. This composition preserves the engine's
shared 2H output LN, both contractions, pair mask, dropout and residual.
Both paths produce fresh saves and all eleven gradients inside each graph.
Weight-layout conversions are included in the primary CUDA comparison.
"""
from ring_plan import *
from integrate_front import backward_full
import argparse
import hashlib
import importlib.metadata
import os
import platform
import cuequivariance_ops_torch as cueq
from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm


def cueq_forward(x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, mask, ds):
    xn = layer_norm_transpose(x, gi, bi, eps=1e-5, layout='bijd->bijd')
    ab = fused_sigmoid_gated_dual_gemm(
        xn, torch.cat((wlg, wrg)), torch.cat((wl, wr)), mask=mask,
        transpose_out=True)
    left, right = ab.chunk(2, dim=0)
    h = go.numel() // 2
    outgoing = torch.einsum('dbik,dbjk->dbij', left[:h], right[:h])
    incoming = torch.einsum('dbki,dbkj->dbij', left[h:], right[h:])
    tri = torch.cat((outgoing, incoming), dim=0)
    norm = layer_norm_transpose(tri, go, bo, eps=1e-5, layout='dbij->bijd')
    update = torch.sigmoid(torch.nn.functional.linear(xn, wg)) * torch.nn.functional.linear(norm, wp)
    return update * ds + x


NAMES = ('dx', 'dWL', 'dWLg', 'dWR', 'dWRg', 'dWgate', 'dWproj',
         'dgamma_in', 'dbeta_in', 'dgamma_out', 'dbeta_out')
STRICT = (2e-5, *([5e-4] * 6), *([5e-6] * 4))


def check(out, ref, strict, eager=False):
    result = {}
    for name, x, y, limit in zip(('forward', *NAMES), (out[0], *out[1]),
                               (ref[0], *ref[1]), (0., *STRICT)):
        # Different vendor BF16 intermediate rounding: report separately from
        # the strict same-saves CUDA test. Never silently loosen CUDA limits.
        # Eager gate/multiply/residual each round to BF16. The primary compiled
        # comparison fuses them, as does our CUDA output kernel. Eager is only
        # a secondary timing/rounding diagnostic and has its own stated bound.
        limit = limit if strict else ((.005 if eager else .001) if name == 'forward' else .01)
        error = rel(x, y)
        finite = bool(torch.isfinite(x).all())
        result[name] = dict(relative_l2=error, limit=limit, finite=finite)
        assert finite and error <= limit, (name, result[name])
    return result


def capture_outputs(fn, stream=None):
    stream = stream or torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = fn()
    return graph, outputs


def pool(blocks):
    result = {}
    for name in blocks[0]:
        samples = sorted(t for block in blocks for t in block[name]['samples_us'])
        result[name] = dict(median_us=statistics.median(samples),
                            p90_us=samples[int(.9 * (len(samples)-1))],
                            samples_us=samples)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=int, choices=(384, 768), required=True)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--iterations', type=int, default=200)
    args = ap.parse_args()
    n = args.length
    output = R / ('cueq-training-L%d.json' % n)
    cueq.init_triton_cache()
    with torch.no_grad():
        a = setup(n)
        d = a['d']
        if n == 384:
            source = 'front_prefetch_lnpair_storepipe'
            p = WarpPlan(a, count=264, splits=13, source=source)
        else:
            source = 'front_ring96_cache3_glu_ahead_u4_early_writer32'
            p = RingPlan(a, count=264, splits=20, source=source)
        y, saves = C.forward(d, True, (3, 64, 2, 2, 1), (1, 1))
        ref = y, C.backward(d, saves, a['dy'])

    def repack_weights():
        _, wl, wlg, wr, wrg, wg, _, _, _, _, _ = d['leaves']
        d['wt'] = [q.t().contiguous() for q in (wl, wlg, wr, wrg, wg)]
        gate = torch.cat((wlg, wrg), 0)
        proj = torch.cat((wl, wr), 0)
        d['w1'] = torch.stack((gate.reshape(-1, 32, 128), proj.reshape(-1, 32, 128)), 1).reshape(1024, 128)

    def cuda_forward(repack=True):
        with torch.no_grad():
            if repack:
                repack_weights()
            return C.forward(d, True, (3, 64, 2, 2, 1), (1, 1))

    def cuda_train(repack=True):
        with torch.no_grad():
            y, saved = cuda_forward(repack)
            a['s'] = saved
            ctx, a['mu'], a['rs'] = saved
            (a['xn'], a['wl'], a['wlg'], a['wr'], a['wrg'], a['wg'],
             _, _, a['pre'], *_) = ctx.saved_tensors
            return y, backward_full(a, p)

    checks = {'cuda': check(cuda_train(), ref, True)}
    compiled = torch.compile(cueq_forward, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
    fwd_fns = {'cuda': cuda_forward}
    total_fns = {'cuda': cuda_train, 'cuda_prepacked': lambda: cuda_train(False)}
    streams = {}
    for label, fn in [('cueq_eager', cueq_forward), ('cueq_compile', compiled)]:
        # All AccumulateGrad nodes are born on the same stream used for graph
        # capture. Forward and backward are included together in every replay.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        streams[label] = stream
        with torch.cuda.stream(stream):
            leaves = tuple(v.detach().clone().requires_grad_(True) for v in d['leaves'])
            mask = d['mask'].reshape(1, n, n).to(torch.bfloat16)
            ds = d['ds'].reshape(1, 1, n, 128)

        def train(fn=fn, leaves=leaves, mask=mask, ds=ds):
            with torch.enable_grad():
                y = fn(*leaves, mask, ds)
                grads = torch.autograd.grad(y, leaves, a['dy'])
                return y, grads

        def forward(fn=fn, leaves=leaves, mask=mask, ds=ds):
            with torch.enable_grad():
                return fn(*leaves, mask, ds)

        print('BUILD', label, flush=True)
        with torch.cuda.stream(stream):
            out = train()
        torch.cuda.current_stream().wait_stream(stream)
        checks[label] = check(out, ref, False, eager=label == 'cueq_eager')
        print('CHECK', label, checks[label], flush=True)
        total_fns[label] = train
        fwd_fns[label] = forward

    record = dict(L=n, C=128, hidden_per_direction=128, packed_hidden=256,
        batch=1, dropout=.25, source=source, checks=checks, times={}, blocks={},
        metadata=dict(torch=torch.__version__, triton=importlib.metadata.version('triton'),
            cuequivariance_torch=importlib.metadata.version('cuequivariance-torch'),
            cuequivariance_ops=importlib.metadata.version('cuequivariance-ops-torch-cu12'),
            gpu=torch.cuda.get_device_name(), hostname=platform.node(),
            tuning=os.environ.get('CUEQ_TRITON_TUNING', 'default'),
            compile='cuEq static fullgraph; explicit one-call CUDA Graph for every timed path',
            semantics='matched shared-256-channel-LN bidirectional cuEq primitive composition, not two public TMU calls',
            dtype='BF16 activations/weights; FP32 LN parameters',
            current='Anthropic-derived saved forward, unchanged Triton/cuBLAS B1-B6, new CUDA B7-B12; excludes separate Claude B1-B4',
            includes='fresh forward saves, all 11 gradients, supplied pair/dropout masks, residual, weight-layout conversion in primary cuda and all cuEq timings',
            excludes=['optimizer', 'RNG generation', 'CPU/autograd dispatch', 'compilation'],
            source_sha256=hashlib.sha256((R/(source+'.cu')).read_bytes()).hexdigest(),
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for scope, fns in [('forward_backward', total_fns), ('forward_training', fwd_fns)]:
        graphs = {}
        replay_checks = {}
        for label, fn in fns.items():
            print('CAPTURE', scope, label, flush=True)
            g, out = capture_outputs(fn, streams.get(label))
            g.replay()
            torch.cuda.synchronize()
            if scope == 'forward_backward':
                replay_checks[label] = check(out, ref, label.startswith('cuda'), eager=label == 'cueq_eager')
                assert bool((p.counts[:2] == 0).all())
            graphs[label] = g
        blocks = [paired(graphs, iterations=args.iterations) for _ in range(args.blocks)]
        times = pool(blocks)
        record['times'][scope] = times
        record['blocks'][scope] = blocks
        record.setdefault('replay_checks', {})[scope] = replay_checks
        print('RESULT', scope, {k: round(v['median_us'], 3) for k, v in times.items()}, flush=True)
        output.write_text(json.dumps(record, indent=2))
        del graphs
    print('DONE', output, flush=True)


if __name__ == '__main__':
    main()
