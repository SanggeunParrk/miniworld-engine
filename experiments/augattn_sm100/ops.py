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
                tm(v2, [H * D, A * L], H * D * 2, [D, 64]), tm(bsrc, [L, H * L], L * 2, [64, 128]))
        items = (A // 2) * H * (L // 128)
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grid = (min(self.ctas or nsm, items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, O, LSE, int(L), int(A))
        run.keep = (maps, q2, k2, v2, bsrc)
        run.k_launch = run
        return run, O, LSE
