"""Common A/B launch wrapper for untouched and newly named CUDA variants.

All allocation, descriptor construction, and Wp packing happen once. Timed
CUDA Graph replay performs only the device launches. Each plan is single-stream
and non-reentrant, as are the original plans.
"""
from core import *
from dual import Unified as Original
import re
import os


@lru_cache(None)
def load_variant(source, count, part):
    inc = T._upstream() / 'csrc'
    src = R / (source + '.cu')
    flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--cubin', '-lineinfo',
             '-Xptxas=-v', '-I' + str(inc), '-DROLE=0',
             f'-DUCOUNT={count}', f'-DPART_ONLY={part}']
    deps = [(R / 'dual_primitives.cuh').read_bytes()]
    deps += [(inc / p).read_bytes() for p in
             ('tmn_kernels.cuh', 'tmn_ptx.cuh', 'common/tmn_math.cuh')]
    key = hashlib.sha256(src.read_bytes() + b''.join(deps) + str(flags).encode()).hexdigest()
    path = R / 'build' / (key + '.cubin')
    path.parent.mkdir(exist_ok=True)
    log = path.with_suffix('.ptxas.log')
    command = ['nvcc', *flags, str(src), '-o', str(path)]
    if not path.exists():
        p = subprocess.run(command, capture_output=True, text=True)
        log.write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError(p.stderr)
        path.with_suffix('.json').write_text(json.dumps(dict(
            source=source, count=count, part=part, flags=flags, command=command), indent=2))
    # Reject any variant that spills; record the compiler evidence in every log.
    compiler = log.read_text()
    if re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)', compiler):
        # Diagnostic-only escape hatch (timestamp probes add registers); never for candidates.
        if os.environ.get('B1B4_ALLOW_SPILL') == '1' and source.endswith('_probe'):
            print('WARNING spill tolerated for diagnostic probe', source, flush=True)
        else:
            raise RuntimeError('Spill regression: ' + str(log) + '\n' + compiler)
    print('PTXAS', source, count, part, str(path), compiler, flush=True)
    launch = T._launch_module()
    drv = launch.BlockDriver(device=0)
    mod = drv.load(path.read_bytes())
    unit = launch.Unit(source, 'sm_90a', 0, drv.drv, mod, {}, str(path))
    k = unit.kernel('dual_b1b4')
    k.set_max_dynamic_smem(231424)
    return k, unit.kernel('unified_reduce')


class Experiment(Original):
    def __init__(self, d, dy, saved, count=132, part=2, source='dual'):
        assert part in (1, 2) and 0 < count
        assert torch.cuda.get_device_capability(0) == (9, 0)
        # Empty CTAs must still zero/publish their partials and join the barrier.
        # Unlike the original host wrapper, count > number of row tiles is valid.
        config_path = R / (source + '.launch.json')
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.counter_words = config.get('counter_words', 2)
        assert self.counter_words in (2, 4)
        if self.counter_words != 2:
            assert f'#define COUNTER_WORDS {self.counter_words}' in (R / (source + '.cu')).read_text()
        self.threads = config.get('threads', 256)
        assert self.threads in (256, 384, 512)
        if self.threads != 256:
            assert f'#define CTA_THREADS {self.threads}' in (R / (source + '.cu')).read_text()
        self.row_tile = config.get('row_tile', 64)
        assert self.row_tile in (32, 64)
        if self.row_tile != 64:
            assert f'#define ROW_TILE {self.row_tile}' in (R / (source + '.cu')).read_text()
        self.dtri_channels = config.get('dtri_channels', 16)
        assert self.dtri_channels in (16, 32, 64, 128)
        if self.dtri_channels != 16:
            assert f'#define DTRI_CHANNELS {self.dtri_channels}' in (R / (source + '.cu')).read_text()
        # Optional appended CUtensorMap over ds[L,128] (dual_ds_tma lineage): the
        # kernel's Params must end with 'CUtensorMap dsm;' when this is set.
        self.ds_tensor_map = bool(config.get('ds_tensor_map', False))
        if self.ds_tensor_map:
            assert 'CUtensorMap dsm;' in (R / (source + '.cu')).read_text()
        # Optional appended 32-row-box maps for a DW role that streams 32-row tiles (dual_dw32 lineage).
        self.dw_row32_maps = bool(config.get('dw_row32_maps', False))
        if self.dw_row32_maps:
            assert 'CUtensorMap dy32,gate32,proj32,xn32,norm32;' in (R / (source + '.cu')).read_text()
        # Diagnostic streaming floor: 32-row boxes for all six inputs plus a 32-row dtri store map.
        self.stream32_maps = bool(config.get('stream32_maps', False))
        if self.stream32_maps:
            assert 'CUtensorMap dy32,gate32,proj32,xn32,norm32,tri32,dtri32;' in (R / (source + '.cu')).read_text()
        self._init_workspace(d, dy, saved, count)
        self.k, self.reduce = load_variant(source, count, part)
        self.source = source
        self.part, self.wgroups, self.smem, self.grid = part, self.threads // 128, 231424, count

    def __call__(self):
        if self.threads == 256:
            return super().__call__()
        if self.part == 2:
            import ctypes
            assert self.grid <= torch.cuda.get_device_properties(0).multi_processor_count
            launch = T._launch_module()
            dr = self.k.unit.drv
            packed = launch._Packed([self.p])
            stream = int(torch.cuda.current_stream().cuda_stream)
            dr._unwrap('cuLaunchCooperativeKernel', dr.d.cuLaunchCooperativeKernel(
                dr.d.CUfunction(int(self.k.handle)), self.grid, 1, 1,
                self.threads, 1, 1, self.smem, dr.d.CUstream(stream),
                ctypes.addressof(packed.array)))
        else:
            self.k.launch((self.grid, 1, 1), (self.threads, 1, 1), [self.p], self.smem)
            self.reduce.launch((194, 1, 1), (256, 1, 1), [self.p], 0)
        return self.outputs

    def _init_workspace(self, d, dy, saved, count):
        super()._init_workspace(d, dy, saved, count)
        xn, wl, wlg, wr, wrg, wg, wp, go, pre, lf, rf, tri, norm, mean, rs, gate, proj = saved[0].saved_tensors
        dg, dwg, dt, dgamma, dbeta, dwp = self.outputs
        partw, partln, _, counts = self.workspace
        if self.counter_words != 2:
            counts = torch.zeros(self.counter_words, dtype=torch.int32, device=dy.device)
        self.timestamps = torch.empty((count, 12), dtype=torch.int64, device=dy.device)
        self.workspace = (partw, partln, self.timestamps, counts)
        launch = T._launch_module()
        m = d['n'] ** 2
        tm = lambda t, box, dims, strides, swizzle='128B': launch.tensor_map(
            t, box, dims=dims, strides_bytes=strides, swizzle=swizzle, l2='128B')
        tile = self.row_tile
        row = lambda t, c: tm(t, [64, tile], [c, m], [c * 2])
        tri_swizzle = '64B' if tile == 32 else '128B'
        maps = [row(dy, 128), row(gate, 128), row(proj, 128), row(xn, 128),
                row(norm, 256), tm(tri, [tile, 256], [m, 256], [m * 2], tri_swizzle),
                tm(self.wpt, [64, 64], [128, 256], [256]),
                tm(dt, [tile, self.dtri_channels, 1], [m, 256, 1], [m * 2, m * 512], tri_swizzle)]
        self.p = launch.Struct([*maps, d['ds'], mean, rs, go, dg, dwg, dwp,
                                dgamma, dbeta, partw, partln, self.timestamps,
                                counts, m, d['n'], m // tile, (m // tile + 31) // 32])
        if getattr(self, 'dw_row32_maps', False):
            row32 = lambda t, c: tm(t, [64, 32], [c, m], [c * 2])
            self.maps32 = [row32(dy, 128), row32(gate, 128), row32(proj, 128), row32(xn, 128), row32(norm, 256)]
            self.p = launch.Struct([*self.p.fields, *self.maps32])
        if getattr(self, 'stream32_maps', False):
            row32 = lambda t, c: tm(t, [64, 32], [c, m], [c * 2])
            self.maps32 = [row32(dy, 128), row32(gate, 128), row32(proj, 128), row32(xn, 128), row32(norm, 256),
                           tm(tri, [32, 256], [m, 256], [m * 2], '64B'), tm(dt, [32, 16, 1], [m, 256, 1], [m * 2, m * 512], '64B')]
            self.p = launch.Struct([*self.p.fields, *self.maps32])
        if getattr(self, 'ds_tensor_map', False):
            ds = d['ds']
            assert ds.dtype == torch.bfloat16 and ds.is_contiguous() and tuple(ds.shape) == (d['n'], 128)
            self.dsmap = tm(ds, [64, 64], [128, d['n']], [256])
            self.p = launch.Struct([*self.p.fields, self.dsmap])
