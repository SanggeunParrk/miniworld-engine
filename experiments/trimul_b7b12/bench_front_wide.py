"""B7-B12 at one pair width: correctness against the Triton/cuBLAS path, then a paired CUDA-graph comparison.

Inputs are synthetic (both paths read the same saved tensors), so a width costs nothing but an argument. The C = 128
column reproduces the qualified configuration of front_prefetch_lnpair_storepipe on the same harness.
"""
import argparse, json, statistics, torch
from wide_plan import WidePlan, R, rel, capture, paired
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_dual_bwd, input_ln_residual_bwd
from miniworld_engine.autotune.shape_key import both_key

NAMES = ('dx', 'dWL', 'dWLg', 'dWR', 'dWRg', 'dgamma', 'dbeta')
LIMITS = dict(dx=2e-5, dWL=5e-4, dWLg=5e-4, dWR=5e-4, dWRg=5e-4, dgamma=5e-6, dbeta=5e-6)


def setup(n, c, seed=20260923):
    torch.manual_seed(seed + n + c)
    m, hs, dev, dt = n * n, 2 * c, 'cuda', torch.bfloat16
    x = torch.randn(m, c, device=dev, dtype=dt)
    gamma, beta = torch.rand(c, device=dev), torch.randn(c, device=dev)
    xf = x.float()
    mu = xf.mean(-1)
    rs = (xf.var(-1, unbiased=False) + 1e-5).rsqrt()
    xn = (((xf - mu[:, None]) * rs[:, None]) * gamma + beta).to(dt)
    w = {k: (torch.randn(c, hs, device=dev, dtype=dt) / c ** .5) for k in ('wl', 'wlg', 'wr', 'wrg')}
    return dict(n=n, c=c, x=x, xn=xn, mu=mu, rs=rs, gamma=gamma, beta=beta, **w,
                pre=torch.randn(4 * hs, m, device=dev, dtype=dt),
                dl=torch.randn(1, hs, n, n, device=dev, dtype=dt), dr=torch.randn(1, hs, n, n, device=dev, dtype=dt),
                dg=torch.randn(m, c, device=dev, dtype=dt), dy=torch.randn(m, c, device=dev, dtype=dt),
                wg=torch.randn(c, c, device=dev, dtype=dt) / c ** .5,
                mask=(torch.rand(m, device=dev) > .2).to(dt))


def baseline(a):
    n, m = a['n'], a['n'] ** 2
    dc, dwl, dwlg, dwr, dwrg, wstack = front_bwd_dW(a['dl'], a['dr'], a['pre'], a['xn'],
                                                    a['wl'], a['wlg'], a['wr'], a['wrg'], pair_mask=a['mask'])
    dxn = input_dual_bwd(a['dg'], dc.t(), a['wg'].t(), wstack, n)
    dx, dgam, dbeta = input_ln_residual_bwd(dxn, a['x'], a['gamma'], a['mu'], a['rs'], a['dy'], both_key(m))
    return (dx, dwl, dwlg, dwr, dwrg, dgam, dbeta)


def reference_fp32(a):
    """The region in fp32, straight from the saved tensors: the tolerance a bf16 kernel is judged against, since the
    Triton path it replaces is itself bf16 and its own error grows with the width."""
    c, n = a['c'], a['n']
    m, hs = n * n, 2 * c
    pre = a['pre'].float().reshape(2, hs, 2, m)
    dside = torch.stack((a['dl'], a['dr'])).float().reshape(2, hs, m) * a['mask'].float()
    sig = torch.sigmoid(pre[:, :, 0])
    dsig = dside * pre[:, :, 1] * sig * (1 - sig)
    dproj = dside * sig
    xn = a['xn'].float()
    dws = [dsig[0] @ xn, dproj[0] @ xn, dsig[1] @ xn, dproj[1] @ xn]          # (HS, C) per matrix
    dxn = (dsig[0].t() @ a['wlg'].float().t() + dproj[0].t() @ a['wl'].float().t()
           + dsig[1].t() @ a['wrg'].float().t() + dproj[1].t() @ a['wr'].float().t()
           + a['dg'].float() @ a['wg'].float().t())
    xh = (a['x'].float() - a['mu'][:, None]) * a['rs'][:, None]
    ha = dxn * a['gamma']
    dx = (ha - xh * (ha * xh).mean(-1, keepdim=True) - ha.mean(-1, keepdim=True)) * a['rs'][:, None] + a['dy'].float()
    return (dx, dws[1].t(), dws[0].t(), dws[3].t(), dws[2].t(), (dxn * xh).sum(0), dxn.sum(0))


def errors(out, ref):
    return {k: dict(relative_l2=rel(x, y), max_absolute=(x.float() - y.float()).abs().max().item(),
                    finite=bool(torch.isfinite(x).all())) for k, x, y in zip(NAMES, out, ref)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=int, default=384)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--count', type=int, default=None)
    ap.add_argument('--dwctas', type=int, default=None)
    ap.add_argument('--splits', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--allow-spills', action='store_true')
    ap.add_argument('--no-time', action='store_true')
    ap.add_argument('--only-kernel', action='store_true', help='launch the kernel alone, for compute-sanitizer')
    ap.add_argument('--ref-length', type=int, default=128, help='length the fp32 comparison runs at (0 skips it)')
    args = ap.parse_args()
    with torch.no_grad():
        a = setup(args.length, args.width)
        p = WidePlan(a, args.width, args.count, args.dwctas, args.splits, allow_spills=args.allow_spills)
        print('CONFIG', dict(c=p.c, split=p.split, dwctas=p.dwctas, dxcount=p.dxcount, splits=p.splits,
                             groups=p.g['groups'], shared=p.shared), flush=True)
        if args.only_kernel:
            p()
            torch.cuda.synchronize()
            print('KERNEL_OK', args.width, flush=True)
            raise SystemExit
        accuracy = {}
        if args.ref_length:
            b = setup(args.ref_length, args.width)
            q = WidePlan(a=b, c=args.width, count=args.count, dwctas=args.dwctas, splits=args.splits,
                         allow_spills=args.allow_spills)
            exact = reference_fp32(b)
            ours, triton = errors(q(), exact), errors(baseline(b), exact)
            torch.cuda.synchronize()
            accuracy = {k: dict(ours=ours[k]['relative_l2'], triton=triton[k]['relative_l2'],
                                finite=ours[k]['finite']) for k in ours}
            print('VS_FP32', args.ref_length, json.dumps(accuracy), flush=True)
            bad = [k for k, v in accuracy.items()
                   if not v['finite'] or v['ours'] > max(1.5 * v['triton'], 1e-6)]
            assert not bad, ('less accurate than the Triton path it replaces', bad, accuracy)
            del b, q, exact, ours, triton
        out, ref = p(), baseline(a)
        torch.cuda.synchronize()
        e = errors(out, ref)
        print('VS_TRITON', json.dumps(e, indent=None), flush=True)
        assert all(v['finite'] for v in e.values())
        assert p.split or bool((p.counts[:2] == 0).all()), p.counts[:2]
        print('PASS', args.width, flush=True)
        if args.no_time:
            raise SystemExit
        graphs = {'baseline': capture(lambda: baseline(a)), 'b7b12_cuda': capture(p)}
        blocks = [paired(graphs) for _ in range(3)]
        times = {name: dict(median_us=statistics.median(sorted(t for b in blocks for t in b[name]['samples_us'])))
                 for name in graphs}
        speedup = times['baseline']['median_us'] / times['b7b12_cuda']['median_us']
        record = dict(L=args.length, C=args.width, count=p.count, dwctas=p.dwctas, splits=p.splits, split=p.split,
                      shared=p.shared, vs_triton=e, vs_fp32=accuracy, ref_length=args.ref_length,
                      times=times, speedup=speedup,
                      device=torch.cuda.get_device_name())
        (R / f'records/front-wide-L{args.length}-C{args.width}{args.tag}.json').write_text(json.dumps(record, indent=2))
        print('FRONT_WIDE', args.length, args.width, {k: round(v['median_us'], 2) for k, v in times.items()},
              'speedup', round(speedup, 3), flush=True)
