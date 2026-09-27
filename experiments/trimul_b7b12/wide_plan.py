"""Width-parameterised B7-B12 plan (C = 128, 256, 384, 512) over front_wideC.cu.

The C = 128 harness (front_plan / warp_plan) stays exactly as it is; this one carries the width through the build defines,
the tensor maps and the partial buffers, so a width is one argument rather than a second kernel.
"""
from pathlib import Path
import ctypes, fcntl, hashlib, json, os, re, statistics, subprocess, torch
R = Path(__file__).resolve().parent
import training_support as T
from front_core import rel, capture, paired                       # timing and error helpers are width agnostic

SOURCE = 'front_wideC'


def build(c, count, dwctas, splits, part=2, source=SOURCE, role=0, depth=0, blocks=0, mrows=0, cspan=0, threads=0, extra=()):
    inc = T._upstream() / 'csrc'
    src = R / (source + '.cu')
    flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--cubin', '-lineinfo', '-Xptxas=-v', '-I' + str(inc),
             f'-DMW_C={c}', f'-DUCOUNT={count}', f'-DDW_SPLITS={splits}', f'-DDW_CTAS={dwctas}', f'-DPART_ONLY={part}', f'-DROLE_ONLY={role}', '-DDEBUG_SAVE=0'] + ([f'-DDW_DEPTH={depth}'] if depth else []) + ([f'-DDW_BLOCKS={blocks}'] if blocks else []) + ([f'-DMW_MROWS={mrows}'] if mrows else []) + ([f'-DDW_CSPAN={cspan}'] if cspan else []) + ([f'-DDW_THREADS={threads}'] if threads else []) + list(extra)
    deps = b''.join((R / f).read_bytes() for f in ('front_primitives.cuh', 'front_mn_primitives.cuh', 'warp_primitives.cuh'))
    deps += b''.join((inc / f).read_bytes() for f in ('tmn_kernels.cuh', 'tmn_ptx.cuh', 'common/tmn_math.cuh'))
    key = hashlib.sha256(src.read_bytes() + deps + str(flags).encode()).hexdigest()
    path = R / 'build' / (key + '.cubin')
    path.parent.mkdir(exist_ok=True)
    log = path.with_suffix('.ptxas.log')
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            done = subprocess.run(['nvcc', *flags, str(src), '-o', str(path)], capture_output=True, text=True)
            log.write_text(done.stdout + done.stderr)
            if done.returncode:
                raise RuntimeError(done.stderr)
    return path, log.read_text()


def load(c, count, dwctas, splits, part=2, source=SOURCE, allow_spills=False, role=0, depth=0, blocks=0, mrows=0, cspan=0, threads=0, extra=()):
    path, compiler = build(c, count, dwctas, splits, part, source, role, depth, blocks, mrows, cspan, threads, extra)
    print('PTXAS', c, compiler.strip().splitlines()[-1] if compiler.strip() else '', flush=True)
    if not allow_spills and re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)', compiler):
        raise RuntimeError('Spill regression, refusing GPU execution: ' + str(path.with_suffix('.ptxas.log')))
    launch = T._launch_module()
    drv = launch.BlockDriver(device=0)
    mod = drv.load(path.read_bytes())
    unit = launch.Unit(source, 'sm_90a', 0, drv.drv, mod, {}, str(path))
    k = unit.kernel('front_b7b12')
    k.set_max_dynamic_smem(229376)
    return k, unit.kernel('front_reduce'), compiler


def geometry(c):
    """Kernel side constants, mirrored from front_wideC.cu."""
    cnw = c // 128
    hs = 2 * c
    hch = hs // 64
    groups = 2 * hch * cnw
    dw_depth = 2 if cnw == 1 else int(os.environ.get('MW_DW_DEPTH', 3))
    mrows = int(os.environ.get('MW_MROWS', 2 if c == 256 else 1))
    glu_sep = int(os.environ.get('MW_XD_M2_GLU_SEP', 1))
    gw_depth = int(os.environ.get('MW_XD_GW_DEPTH', (2 if glu_sep else 3) if c == 256 else 2))
    dx_shared = (2 * 49152 + gw_depth * 32768 + (32768 if glu_sep else 0) + 32 * c + 2048) if mrows == 2 else \
                (2 * 40960 + gw_depth * 32768 + 49152 + 32 * c + 1024)
    shared = max(dw_depth * 57344, 114688 if cnw == 1 else dx_shared)
    return dict(cnw=cnw, hs=hs, hch=hch, groups=groups, shared=shared, dw_depth=dw_depth, gw_depth=gw_depth,
                mrows=mrows, dx_shared=dx_shared, ctas_per_sm=2 if cnw == 1 else 1)


def defaults(c):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    g = geometry(c)
    if g['cnw'] == 1:
        return dict(count=2 * sms, dwctas=8 * 13, splits=13)
    count = sms
    splits = max(1, round(count / 2 / g['groups'])) or 1
    jobs = g['groups'] * splits
    dwctas = jobs if jobs <= count // 2 + 8 else jobs // max(1, round(jobs / (count // 2)))
    return dict(count=count, dwctas=dwctas, splits=splits)


class WidePlan:
    """One plan per width. At C = 128 it is the single cooperative kernel the D = 128 work qualified; from C = 256 the two
    roles launch as separate kernels, because the dW role fits in 128 registers (two CTAs per SM) while the dx role holds
    C channels of accumulator (one CTA per SM), and at these widths both roles are latency bound, not bandwidth bound."""

    def __init__(self, a, c, count=None, dwctas=None, splits=None, part=2, source=SOURCE, allow_spills=False,
                 role=0, depth=0, split=None):
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.g = geometry(c)
        self.c, self.a, self.part = c, a, part
        self.split = (self.g['cnw'] > 1) if split is None else split
        self.threads = 256
        n = a['n']
        m = n * n
        hs, groups = self.g['hs'], self.g['groups']
        dev = a['x'].device
        if self.split:
            self.dw_threads = int(os.environ.get('MW_DW_THREADS', 256))
            self.cspan = int(os.environ.get('MW_DW_CSPAN', 1 if self.dw_threads == 256 else 2))
            self.blocks = int(os.environ.get('MW_DW_BLOCKS', 2 if (self.cspan == 1 and self.dw_threads == 256) else 1))
            self.dw_depth = depth or (2 if self.blocks > 1 else self.g['dw_depth'])
            cgroups = groups // self.cspan
            # Jobs are equal sized, so the dW kernel costs ceil(jobs / ctas) * (tiles / splits): pick the split count
            # that minimises that, instead of just filling one wave.
            slots = self.blocks * sms
            self.splits = splits or min(range(1, 17), key=lambda v: -(-(cgroups * v) // min(cgroups * v, slots)) / v)
            self.dwctas = min(cgroups * self.splits, slots)
            self.dxcount = count or sms
            dw_shared = self.dw_depth * (24576 + self.cspan * 16384 + 16384)
            if dw_shared * self.blocks > 232448:
                raise ValueError('dW ring does not fit at this occupancy')
            ahead = os.environ.get('MW_DW_GLU_AHEAD')
            dwx = tuple(f'-D{k[6:]}={v}' for k, v in os.environ.items() if k.startswith('MW_XD_') and k != 'MW_XD_MAXREG')
            kdw, _, c1 = load(c, self.dwctas, self.dwctas, self.splits, 0, source, allow_spills, 1, self.dw_depth, self.blocks,
                              cspan=self.cspan, threads=self.dw_threads,
                              extra=dwx if ahead is None else dwx + (f'-DDW_GLU_AHEAD={ahead}',))
            dxx = tuple(f'-D{k[6:]}={v}' for k, v in os.environ.items() if k.startswith('MW_XD_') and k != 'MW_XD_MAXREG')
            if os.environ.get('MW_XD_MAXREG'):
                dxx += (f"-maxrregcount={os.environ['MW_XD_MAXREG']}",)
            kdx, red, c2 = load(c, self.dxcount, 0, self.splits, 0, source, allow_spills, 2, 2, 1, self.g['mrows'],
                                self.cspan, extra=dxx)
            if os.environ.get('MW_VERBOSE'):
                print('PLAN dw', dict(ctas=self.dwctas, threads=self.dw_threads, cspan=self.cspan, depth=self.dw_depth,
                                      shared=dw_shared, splits=self.splits, blocks=self.blocks, cgroups=cgroups),
                      'dx', dict(ctas=self.dxcount, shared=self.g['dx_shared'], mrows=self.g['mrows'],
                                 gw_depth=self.g['gw_depth']), flush=True)
            self.launches = [(kdw, self.dwctas, dw_shared, self.dw_threads), (kdx, self.dxcount, self.g['dx_shared'], 256)]
            self.reduce = red
            self.compiler = c1 + c2
            self.count, self.shared = self.dwctas + self.dxcount, max(dw_shared, self.g['shared'])
        else:
            self.cspan = 1
            d = defaults(c)
            self.count = count or d['count']
            self.splits = splits or d['splits']
            self.dwctas = dwctas or (groups * self.splits)
            self.dxcount = self.count - self.dwctas
            self.shared = self.g['shared']
            k, red, self.compiler = load(c, self.count, self.dwctas, self.splits, part, source, allow_spills, role, depth)
            self.k, self.reduce = k, red
            self.launches = [(k, self.count, self.shared, self.threads)]
        if self.dxcount <= 0:
            raise ValueError('no CTAs left for the dx role')
        self.dx = torch.empty((m, c), device=dev, dtype=torch.bfloat16)
        self.dw = torch.empty((4, c, hs), device=dev, dtype=torch.bfloat16)
        self.dgam = torch.empty(c, device=dev)
        self.dbeta = torch.empty_like(self.dgam)
        self.partw = torch.empty((groups, self.splits * 2, 16384), device=dev)
        self.partln = torch.empty((self.dxcount, 2 * c), device=dev)
        self.counts = torch.zeros(4096, dtype=torch.int32, device=dev)
        self.debug = self.dx
        self.outputs = (self.dx, *self.dw.unbind(0), self.dgam, self.dbeta)
        self.bind(a['dl'], a['dr'], a['dg'], a['dy'])

    def bind(self, dl, dr, dg, dy):
        a, c, hs = self.a, self.c, self.g['hs']
        n = a['n']
        m = n * n
        L = T._launch_module()
        tm = lambda t, box, dims, strides: L.tensor_map(t, box, dims=dims, strides_bytes=strides, swizzle='128B', l2='128B')
        row = lambda t: tm(t, [64, 64], [c, m], [c * 2])
        maps = [tm(dl, [64, 64], [m, hs], [m * 2]), tm(dr, [64, 64], [m, hs], [m * 2]),
                tm(a['pre'], [64, 128], [m, 4 * hs], [m * 2]), row(a['xn']), row(dg),
                *[tm(a[k], [64, 64], [hs, c], [hs * 2]) for k in ('wlg', 'wl', 'wrg', 'wr')],
                tm(a['wg'], [64, 64], [c, c], [c * 2]), row(a['x']), row(dy), row(self.dx)]
        self.p = L.Struct([*maps, a['mask'], a['mu'], a['rs'], a['gamma'], self.dw, self.dgam, self.dbeta,
                           self.partw, self.partln, self.counts, self.debug, self.debug, m, n, m // 64])

    def __call__(self):
        L = T._launch_module()
        if not self.split and self.part == 2:
            k = self.launches[0][0]
            drv = k.unit.drv
            packed = L._Packed([self.p])
            stream = int(torch.cuda.current_stream().cuda_stream)
            drv._unwrap('cuLaunchCooperativeKernel', drv.d.cuLaunchCooperativeKernel(
                drv.d.CUfunction(int(k.handle)), self.launches[0][1], 1, 1, self.launches[0][3], 1, 1, self.launches[0][2],
                drv.d.CUstream(stream), ctypes.addressof(packed.array)))
            return self.outputs
        only = os.environ.get('MW_ONLY_LAUNCH')                  # '0' dW, '1' dx; for bisecting under the sanitizer
        for i, (k, grid, shared, threads) in enumerate(self.launches):
            if only is None or int(only) == i:
                k.launch((grid, 1, 1), (threads, 1, 1), [self.p], shared)
        if not os.environ.get('MW_NO_REDUCE'):
            values = self.g['groups'] * 16384 + 2 * self.c
            self.reduce.launch(((values + 255) // 256, 1, 1), (256, 1, 1), [self.p], 0)
        return self.outputs
