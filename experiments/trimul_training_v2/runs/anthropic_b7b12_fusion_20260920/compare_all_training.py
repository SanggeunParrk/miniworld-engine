"""All selected training CUDA work versus pre-Anthropic Miniworld routes.

Every total graph produces fresh forward saves, executes B1-B4 CUDA, cuBLAS
B5-B6, and B7-B12 CUDA. Includes all weight-layout conversions. No source in
the independent B1-B4 or K1/K3 development directories is modified.
"""
from compare_cueq_training import *
import importlib.util
from miniworld_engine import settings
from miniworld_engine.autotune import trimul_sm90_config as sm90_cfg

B1ROOT = R.parent / 'anthropic_b1b4_pipeline_20260919'


def b1_class():
    # The historical experiments use unqualified imports named core/dual.
    # Load in a temporary namespace, restoring the inference helper's core.
    names = ('core', 'unified', 'dual', 'dual_experiment')
    previous = {k: sys.modules.get(k) for k in names}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, B1ROOT/(name+'.py'))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        return sys.modules['dual_experiment'].Experiment
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class BoundB1(b1_class()):
    def bind(self, d, dy, saved):
        """Rebind current forward saves, retaining reduction/output workspace."""
        xn, _, _, _, _, _, wp, go, _, _, _, tri, norm, mean, rs, gate, proj = saved[0].saved_tensors
        self.wpt = wp.t().contiguous()  # Per-step GPU packing is timed.
        dg, dwg, dt, dgamma, dbeta, dwp = self.outputs
        partw, partln, timestamps, counts = self.workspace
        m = d['n'] ** 2
        launch = T._launch_module()
        def tm(t, box, dims, strides):
            return launch.tensor_map(t, box, dims=dims, strides_bytes=strides,
                                     swizzle='128B', l2='128B')
        row = lambda t, c: tm(t, [64, 64], [c, m], [c*2])
        maps = [row(dy, 128), row(gate, 128), row(proj, 128), row(xn, 128),
                row(norm, 256), tm(tri, [64, 256], [m, 256], [m*2]),
                tm(self.wpt, [64, 64], [128, 256], [256]),
                tm(dt, [64, 16, 1], [m, 256, 1], [m*2, m*512])]
        self.p = launch.Struct([*maps, d['ds'], mean, rs, go, dg, dwg, dwp,
                                dgamma, dbeta, partw, partln, timestamps, counts,
                                m, d['n'], m//64, (m//64+31)//32])


def check_combined(out, ref):
    # The prior B1-B4 *full backward* contract is 5e-4 for all gradients.
    # The tighter B7 region contract applies only with identical B1-B6 inputs;
    # enforce it separately below rather than silently equating the scopes.
    result = {'forward': dict(relative_l2=rel(out[0], ref[0]), limit=0.,
                              finite=bool(torch.isfinite(out[0]).all()))}
    for name, x, y in zip(NAMES, out[1], ref[1]):
        result[name] = dict(relative_l2=rel(x, y), limit=5e-4,
                            finite=bool(torch.isfinite(x).all()))
    print('COMBINED_ERRORS', result, flush=True)
    assert all(v['finite'] and v['relative_l2'] <= v['limit'] for v in result.values()), result
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=int, choices=(384, 768), required=True)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--iterations', type=int, default=200)
    args = ap.parse_args()
    n = args.length
    output = R / ('all-training-L%d.json' % n)
    with torch.no_grad():
        a = setup(n)
        d = a['d']
        settings.configure(autotune_miss_cap=24)
        source = ('front_prefetch_lnpair_storepipe' if n == 384 else
                  'front_ring96_cache3_glu_ahead_u4_early_writer32')
        p7 = (WarpPlan(a, count=264, splits=13, source=source) if n == 384 else
              RingPlan(a, count=264, splits=20, source=source))
        p1 = BoundB1(d, a['dy'], a['s'], 132, 2, 'dual_ln_prefetch')
        y, saved = C.forward(d, True, (3, 64, 2, 2, 1), (1, 1))
        ref = y, C.backward(d, saved, a['dy'])

    def current_forward():
        with torch.no_grad():
            _, wl, wlg, wr, wrg, wg, _, _, _, _, _ = d['leaves']
            d['wt'] = [q.t().contiguous() for q in (wl, wlg, wr, wrg, wg)]
            gate, proj = torch.cat((wlg, wrg), 0), torch.cat((wl, wr), 0)
            d['w1'] = torch.stack((gate.reshape(-1, 32, 128),
                                    proj.reshape(-1, 32, 128)), 1).reshape(1024, 128)
            return C.forward(d, True, (3, 64, 2, 2, 1), (1, 1))

    def current_train(all_bwd=True):
        with torch.no_grad():
            y, saved = current_forward()
            a['s'] = saved
            ctx, a['mu'], a['rs'] = saved
            (a['xn'], a['wl'], a['wlg'], a['wr'], a['wrg'], a['wg'],
             _, _, a['pre'], lf, rf, *_) = ctx.saved_tensors
            if not all_bwd:
                return y, backward_full(a, p7)
            p1.bind(d, a['dy'], saved)
            dg, dwg, dt, dgo, dbo, dwp = p1()
            dl, dr = B.packed_backward(dt, lf, rf, 128)
            a['dl'], a['dr'], a['dg'] = dl.reshape(1, 256, n, n), dr.reshape(1, 256, n, n), dg
            p7.bind(dl, dr, dg, a['dy'])
            dx, dwl, dwlg, dwr, dwrg, dgi, dbi = p7()
            return y, (dx.reshape_as(d['x']), dwl.t(), dwlg.t(), dwr.t(), dwrg.t(),
                       dwg.t(), dwp, dgi, dbi, dgo, dbo)

    out = current_train()
    checks = {'current_all': check_combined(out, ref)}
    with torch.no_grad():
        b1ref_fn = BoundB1.__mro__[1].__init__.__globals__['baseline']
        b1ref = b1ref_fn(d, a['dy'], a['s'])
        checks['b1_region'] = {}
        for name, x, y, limit in zip(('dg', 'dWg', 'dtri', 'dgamma', 'dbeta', 'dWp'),
                                     p1.outputs, b1ref, (0., 5e-4, 2e-5, 5e-6, 5e-6, 5e-4)):
            error = rel(x, y)
            exact = bool(torch.equal(x.view(torch.int16), y.view(torch.int16))) if name == 'dg' else None
            checks['b1_region'][name] = dict(relative_l2=error, limit=limit, finite=bool(torch.isfinite(x).all()), bit_exact=exact)
        assert all(v['finite'] and v['relative_l2'] <= v['limit'] and v['bit_exact'] is not False
                   for v in checks['b1_region'].values()), checks['b1_region']
        # Same actual B1-B6 outputs on both sides for B7's strict contract.
        b7ref = baseline(a)
        es = errors(p7.outputs, b7ref)
        limits = dict(dx=2e-5, dWL=5e-4, dWLg=5e-4, dWR=5e-4, dWRg=5e-4, dgamma=5e-6, dbeta=5e-6)
        assert all(v['finite'] and v['relative_l2'] <= limits[k] for k,v in es.items()), es
        checks['b7_region'] = {k: dict(v, limit=limits[k]) for k,v in es.items()}
    print('CHECK_CURRENT_ALL', checks['current_all'], flush=True)
    streams = {}
    totals = {'current_all': current_train, 'previous_b7_only': lambda: current_train(False)}
    forwards = {'current_all': current_forward}
    configs_path = R.parent / ('trimul_sm90_round2_20260917/module/measured-configs-L%d.json' % n)
    measured = json.loads(configs_path.read_text())
    original_resolve = sm90_cfg.resolve
    calls = {}
    def measured_resolve(op, tensors, **kwargs):
        if op not in measured:
            return original_resolve(op, tensors, **kwargs)
        config = dict(measured[op])
        assert kwargs['feasibility'](config) is None
        calls[op] = calls.get(op, 0) + 1
        return config
    sm90_cfg.resolve = measured_resolve
    cueq.init_triton_cache()
    compiled_cueq = torch.compile(cueq_forward, fullgraph=True, dynamic=False,
                                   options={'triton.cudagraphs': False})

    def old_fn(*inputs):
        leaves, mask, ds = inputs[:-2], inputs[-2], inputs[-1]
        return B.bidirectional_trimul_triton(*leaves, 1e-5, 1e-5, 128,
                                             mask=mask, dropscale=ds, output_backend='triton')

    compiled_old = torch.compile(old_fn, fullgraph=True, dynamic=False,
                                  options={'triton.cudagraphs': False})
    for label, fn, native in [('old_triton', compiled_old, ()),
                               ('old_h100', compiled_old, ('front', 'f567', 'dual_bwd')),
                               ('cueq_compile', compiled_cueq, ())]:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        streams[label] = stream
        with torch.cuda.stream(stream):
            leaves = tuple(v.detach().clone().requires_grad_(True) for v in d['leaves'])
            mask = d['mask'].reshape(1, n, n).to(torch.bfloat16)
            ds = d['ds'].reshape(1, 1, n, 128)
        def forward(fn=fn, leaves=leaves, mask=mask, ds=ds, native=native):
            settings.configure(trimul_sm90_kernels=native, autotune_miss_cap=24)
            with torch.enable_grad():
                return fn(*leaves, mask, ds)
        def train(forward=forward, leaves=leaves):
            with torch.enable_grad():
                y = forward()
                return y, torch.autograd.grad(y, leaves, a['dy'])
        print('BUILD', label, flush=True)
        with torch.cuda.stream(stream):
            out = train()
        torch.cuda.current_stream().wait_stream(stream)
        checks[label] = check(out, ref, False)
        print('CHECK', label, checks[label], flush=True)
        totals[label], forwards[label] = train, forward

    record = dict(L=n, C=128, hidden_per_direction=128, packed_hidden=256,
        batch=1, dropout=.25, checks=checks, times={}, blocks={}, replay_checks={},
        metadata=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            source_b7=source, source_b1='dual_ln_prefetch',
            forward='latest selected training-saves fused input-LN Anthropic-derived forward; inference-only K1/K3 payload is not a training implementation',
            current='B1-B4 CUDA + unchanged cuBLAS B5-B6 + B7-B12 CUDA',
            old='preserved pre-Anthropic algorithm routes from current checkout, explicitly output_backend=triton; no historical package/environment recreation',
            includes='fresh forward saves; all 11 gradients; weight packing including B1 Wp transpose; pair mask/dropout/residual',
            excludes=['optimizer', 'RNG generation', 'CPU/autograd dispatch', 'compilation'],
            graph='all paths explicit CUDA Graph; previous Miniworld and cuEq static fullgraph compile',
            cueq_versions=[importlib.metadata.version('cuequivariance-torch'), importlib.metadata.version('cuequivariance-ops-torch-cu12')],
            old_h100_configs=measured, old_h100_config_file=str(configs_path),
            autotune='existing cache; default 24 heuristic candidates on Triton cache misses, not exhaustive retuning',
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [R/(source+'.cu'), B1ROOT/'dual_ln_prefetch.cu', Path(B.__file__)]}))
    for scope, fns in [('forward_backward', totals), ('forward_training', forwards)]:
        graphs = {}
        for label, fn in fns.items():
            print('CAPTURE', scope, label, flush=True)
            g, out = capture_outputs(fn, streams.get(label))
            g.replay()
            torch.cuda.synchronize()
            if scope == 'forward_backward':
                record['replay_checks'][label] = (check_combined(out, ref) if label == 'current_all' else
                    check(out, ref, label == 'previous_b7_only'))
                assert bool((p7.counts[:2] == 0).all())
                assert bool((p1.workspace[-1] == 0).all())
            graphs[label] = g
        blocks = [paired(graphs, iterations=args.iterations) for _ in range(args.blocks)]
        record['blocks'][scope] = blocks
        record['times'][scope] = pool(blocks)
        record['metadata']['old_h100_config_calls'] = calls.copy()
        output.write_text(json.dumps(record, indent=2))
        print('RESULT', scope, {k: v['median_us'] for k,v in record['times'][scope].items()}, flush=True)
        del graphs
    print('DONE', output, flush=True)


if __name__ == '__main__':
    main()
