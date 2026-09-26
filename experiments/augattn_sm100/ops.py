"""Host side of the sm_100a augmented pair-bias attention kernels (cubins launched through drv)."""
import math
from pathlib import Path
import torch
import drv
from common import H, D

HERE = Path(__file__).resolve().parent
LOG2E = 1.0 / math.log(2.0)


def bias_prep(bias):
    """[H, L, L] -> bf16 bias * log2 e (log2 units, as the kernels read it)."""
    return (bias.float() * LOG2E).to(torch.bfloat16).contiguous()


class Fwd:
    def __init__(self, cubin=HERE / "build" / "attn_fwd.cubin"):
        self.k = drv.Kernel(str(cubin), "augattn_fwd_sm100", 229888)

    def bind(self, q, k, v, bias, qwid=48, prep=False):
        """q, k, v [A, 1, L, 16, 48] bf16 contiguous; bias [H, L, L]. Returns run(), O [A*L, 768] fp32, LSE [A, H, L] (log2)."""
        A, _, L, _, _ = q.shape
        assert A % 2 == 0 and L % 128 == 0
        tm = drv.TensorMap
        q2, k2, v2 = (t.reshape(A * L, H * D) for t in (q, k, v))
        bb = torch.empty(H, L, L, device=q.device, dtype=torch.bfloat16)
        O = torch.empty(A * L, H * D, device=q.device, dtype=torch.float32)
        LSE = torch.empty(A, H, L, device=q.device, dtype=torch.float32)
        bsrc = bb if prep else bias.contiguous()
        maps = tuple(tm(t, [H * D, A * L], H * D * 2, [qwid, 128]) for t in (q2, k2, v2)) + (tm(bsrc, [L, H * L], L * 2, [64, 128]),)
        grid = ((A // 2) * H * (L // 128), 1, 1)

        def run():
            if prep:
                bb.copy_(bias_prep(bias))
            self.k(grid, (384, 1, 1), *maps, O, LSE, int(L), int(A))
        run.keep = (maps, q2, k2, v2, bb, bsrc)
        run.k_launch = lambda: self.k(grid, (384, 1, 1), *maps, O, LSE, int(L), int(A))
        return run, O, LSE


class Fwd2(Fwd):
    """attn_fwd2: persistent CTAs (one per SM), 64-key blocks, double-buffered S."""
    def __init__(self, cubin=HERE / "build" / "attn_fwd2.cubin", ctas=None):
        self.k = drv.Kernel(str(cubin), "augattn_fwd2_sm100", 232448)
        self.ctas = ctas

    def bind(self, q, k, v, bias, qwid=48, prep=False):
        A, _, L, _, _ = q.shape
        assert A % 2 == 0 and L % 128 == 0
        tm = drv.TensorMap
        q2, k2, v2 = (t.reshape(A * L, H * D) for t in (q, k, v))
        O = torch.empty(A * L, H * D, device=q.device, dtype=torch.float32)
        LSE = torch.empty(A, H, L, device=q.device, dtype=torch.float32)
        bsrc = bias.contiguous()
        maps = (tm(q2, [H * D, A * L], H * D * 2, [D, 128]), tm(k2, [H * D, A * L], H * D * 2, [D, 64]),
                tm(v2, [H * D, A * L], H * D * 2, [D, 64]), tm(bsrc, [L, H * L], L * 2, [64, 128]),
                tm(O, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, dtype="f32"), tm(O, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, dtype="f32"))
        items = (A // 2) * H * (L // 128)
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grid = (min(self.ctas or nsm, items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, O, LSE, int(L), int(A))
        run.keep = (maps, q2, k2, v2, bsrc)
        run.k_launch = run
        return run, O, LSE


class Dqb:
    """attn_dqb: backward dQ (fp32, reduced over key chunks into a zeroed buffer) and dbias [H, L, L] fp32 (summed over samples on chip)."""
    def __init__(self, cubin=HERE / "build" / "attn_dqb.cubin", ctas=None):
        self.k = drv.Kernel(str(cubin), "augattn_dqb_sm100", 232448)
        self.ctas = ctas

    def bind(self, q, k, v, do, bias, LSE, Dd, zeroed=False):
        """q, k, v, do [A, 1, L, 16, 48] bf16; bias [H, L, L] bf16; LSE (log2), D = rowsum(dO O) [A, H, L] fp32.
        zeroed: DQ is zero-filled elsewhere (attn_dkv's dq_zero) -- otherwise run() clears it first."""
        A, L = q.shape[0], q.shape[2]
        assert L % 128 == 0
        tm = drv.TensorMap
        q2, k2, v2, do2 = (t.reshape(A * L, H * D) for t in (q, k, v, do))
        maps0 = tuple(tm(t, [H * D, A * L], H * D * 2, [D, 128]) for t in (q2, k2, v2, do2))
        DQ = torch.empty(A * L, H * D, device=q.device, dtype=torch.float32)
        DB = torch.empty(H, L, L, device=q.device, dtype=torch.float32)
        bsrc = bias.contiguous()
        maps = maps0 + (tm(bsrc, [L, H * L], L * 2, [64, 128]),
                        tm(DQ, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, dtype="f32"),
                        tm(DQ, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, dtype="f32"))
        items = H * (L // 128) * (L // 128)
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grid = (min(self.ctas or nsm, items), 1, 1)

        def k_launch():
            self.k(grid, (384, 1, 1), *maps, LSE, Dd, DQ, DB, int(L), int(A))

        def run():
            if not zeroed:
                DQ.zero_()
            k_launch()
        run.keep = (maps, q2, k2, v2, do2, bsrc)
        run.k_launch = k_launch
        return run, DQ, DB


class Dkv:
    """attn_dkv: backward dK, dV [A*L, 768] fp32 (TMA stores). Needs the bias transposed, [H, L(key), L(query)] bf16."""
    def __init__(self, cubin=HERE / "build" / "attn_dkv.cubin", ctas=None):
        four = "dkv4" in str(cubin)                    # attn_dkv4: four dS warpgroups, 640 threads
        self.k = drv.Kernel(str(cubin), "augattn_dkv4_sm100" if four else "augattn_dkv_sm100", 232448)
        self.nth = 640 if four else 384
        self.ctas = ctas

    def bind(self, q, k, v, do, bias_t, LSE, Dd, dq_zero=None):
        """dq_zero: an fp32 [A*L, 768] buffer the kernel zero-fills on the way (attn_dqb's dQ, so dqb can run without a memset)."""
        A, _, L, _, _ = q.shape
        four = self.nth == 640
        assert L % 128 == 0
        tm = drv.TensorMap
        q2, k2, v2, do2 = (t.reshape(A * L, H * D) for t in (q, k, v, do))
        DK = torch.empty(A * L, H * D, device=q.device, dtype=torch.float32)
        DV = torch.empty_like(DK)
        maps = (tm(q2, [H * D, A * L], H * D * 2, [D, 64]), tm(k2, [H * D, A * L], H * D * 2, [D, 128]),
                tm(v2, [H * D, A * L], H * D * 2, [D, 128]), tm(do2, [H * D, A * L], H * D * 2, [D, 64]),
                tm(bias_t, [L, H * L], L * 2, [64, 128]))
        if not four:                                    # fp32 output maps: columns 0-31 (128-B swizzle) and 32-47 (64-B swizzle)
            maps = maps + (tm(DK, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, dtype="f32"),
                           tm(DK, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, dtype="f32"),
                           tm(DV, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, dtype="f32"),
                           tm(DV, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, dtype="f32"))
        items = A * H * (L // 128)
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grid = (min(self.ctas or nsm, items), 1, 1)

        def run():
            if four:
                self.k(grid, (self.nth, 1, 1), *maps, LSE, Dd, DK, DV, int(L), int(A))
            else:
                self.k(grid, (self.nth, 1, 1), *maps, LSE, Dd, DK, DV, dq_zero, int(L), int(A))
        run.keep = (maps, q2, k2, v2, do2, bias_t)
        run.k_launch = run
        return run, DK, DV
