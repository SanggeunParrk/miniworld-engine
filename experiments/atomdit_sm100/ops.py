"""Host side of the sm_100a atom DiT attention kernels (cubins launched through drv). Shapes: q, k, v [A, N, 128] bf16 (4 heads x 32,
head h in columns 32 h .. 32 h + 31), bias [4, N, N] bf16 (head-major), bias_t [4, N(key), N(query)]."""
from pathlib import Path
import torch
import drv

HERE = Path(__file__).resolve().parent
NH, DH, DM = 4, 32, 128
NSM = None


def nsm():
    global NSM
    if NSM is None:
        NSM = torch.cuda.get_device_properties(0).multi_processor_count
    return NSM


def _tm(t, rows, box_rows, cols=DM, box_cols=DH):
    return drv.TensorMap(t, [cols, rows], cols * 2, [box_cols, box_rows])


class Fwd:
    def __init__(self, cubin=HERE / "build" / "attn_fwd.cubin"):
        self.k = drv.Kernel(str(cubin), "augattn_fwd2_sm100", 232448)

    def bind(self, q, k, v, bias):
        A, N, _ = q.shape
        assert N % 128 == 0
        q2, k2, v2 = (t.reshape(A * N, DM) for t in (q, k, v))
        O = torch.empty(A * N, DM, device=q.device, dtype=torch.float32)
        LSE = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
        bsrc = bias.contiguous()
        maps = (_tm(q2, A * N, 128), _tm(k2, A * N, 64), _tm(v2, A * N, 64), drv.TensorMap(bsrc, [N, NH * N], N * 2, [64, 128]),
                drv.TensorMap(O, [DM, A * N], DM * 4, [32, 128], swizzle=128, dtype="f32"))
        items = ((A + 1) // 2) * NH * (N // 128)
        grid = (min(nsm(), items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, O, LSE, int(N), int(A))
        run.keep = (maps, q2, k2, v2, bsrc)
        return run, O, LSE


class Bwd:
    """The three backward passes: attn_dkv (dK, dV), attn_dq (dQ), attn_dbias (dbias summed over samples, fp32 [4, N, N])."""
    def __init__(self, dkv=HERE / "build" / "attn_dkv.cubin", dq=HERE / "build" / "attn_dq.cubin", dbias=HERE / "build" / "attn_dbias.cubin"):
        self.kkv = drv.Kernel(str(dkv), "augattn_dkv_sm100", 232448)
        self.kq = drv.Kernel(str(dq), "atom_dq_sm100", 232448)
        self.kb = drv.Kernel(str(dbias), "atom_dbias_sm100", 232448)

    def bind(self, q, k, v, do, bias, bias_t, LSE, Dd):
        """do [A, N, 128] bf16; LSE [A, 4, N] (log2) and D = rowsum(dO O) per head [A, 4, N] fp32."""
        A, N, _ = q.shape
        q2, k2, v2, do2 = (t.reshape(A * N, DM) for t in (q, k, v, do))
        DQ, DK, DV = (torch.empty(A * N, DM, device=q.device, dtype=torch.float32) for _ in range(3))
        DB = torch.empty(NH, N, N, device=q.device, dtype=torch.float32)
        f32 = lambda t, bc, sw: drv.TensorMap(t, [DM, A * N], DM * 4, [bc, 128], swizzle=sw, dtype="f32")
        mb = drv.TensorMap(bias, [N, NH * N], N * 2, [64, 128])
        mbt = drv.TensorMap(bias_t, [N, NH * N], N * 2, [64, 128])
        m128 = tuple(_tm(t, A * N, 128) for t in (q2, k2, v2, do2))
        mkv = (_tm(q2, A * N, 64), m128[1], m128[2], _tm(do2, A * N, 64), mbt, f32(DK, 32, 128), f32(DV, 32, 128))
        mq = m128 + (mb, f32(DQ, 16, 64))
        mbias = m128 + (mb,)
        g_kv = (min(nsm(), A * NH * (N // 128)), 1, 1)
        g_q = (min(nsm(), A * NH * (N // 128)), 1, 1)
        g_b = (min(nsm(), NH * (N // 128) ** 2), 1, 1)

        def run_kv():
            self.kkv(g_kv, (384, 1, 1), *mkv, LSE, Dd, DK, DV, None, int(N), int(A))

        def run_q():
            self.kq(g_q, (384, 1, 1), *mq, LSE, Dd, int(N), int(A))

        def run_b():
            self.kb(g_b, (384, 1, 1), *mbias, LSE, Dd, DB, int(N), int(A))

        def run():
            run_kv(); run_q(); run_b()
        run.parts = (run_kv, run_q, run_b)
        run.keep = (mkv, mq, mbias, q2, k2, v2, do2, bias, bias_t)
        return run, DQ, DK, DV, DB
