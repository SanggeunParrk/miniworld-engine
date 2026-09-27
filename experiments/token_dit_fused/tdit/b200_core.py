"""The token DiT's inference attention core on sm_100a (B200): experiments/augattn_sm100/src/attn_inf.cu, launched through
augattn_sm100/drv.py. Same contract as the Triton gated2 core and the sm_90a CUDA core: reads q | k | v | g as column views of
the [S L, 4 D] q|k|v|g GEMM output (logits pre-scaled into exp2 units), the block's head-major hoisted bias, and writes
sigmoid(g) * o over q in bf16."""
import sys
from pathlib import Path

import torch

_AUG = Path(__file__).resolve().parents[2] / "augattn_sm100"
_CUBIN = _AUG / "build" / "attn_inf.cubin"


class InfCore:
    def __init__(self, cubin=_CUBIN, pdl=False):
        if str(_AUG) not in sys.path:
            sys.path.insert(0, str(_AUG))
        import drv
        self.drv = drv
        self.k = drv.Kernel(str(cubin), "augattn_inf_sm100", 232448, pdl=pdl)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.runs = {}

    def _bind(self, qkvg, bias, block, S, L, H):
        tm = self.drv.TensorMap
        M, D4 = qkvg.shape
        D, DH = D4 // 4, D4 // 4 // H
        rs = D4 * qkvg.element_size()                    # row stride of the q|k|v|g buffer, bytes
        q, k, v, g = (qkvg[:, i * D:(i + 1) * D] for i in range(4))
        bv = bias[block * H:(block + 1) * H]             # this block's heads, [H, L, L] -> rows H L of L keys
        maps = (tm(q, [D, M], rs, [DH, 128]), tm(k, [D, M], rs, [DH, 64]), tm(v, [D, M], rs, [DH, 64]),
                tm(bv, [L, H * L], L * bias.element_size(), [64, 128]), tm(g, [D, M], rs, [DH, 128]))
        items = ((S + 1) // 2) * H * (L // 128)
        grid = (min(self.nsm, items), 1, 1)
        mq, mk, mv, mb, mg = maps

        def run():
            self.k(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), int(S))
        run.keep = (maps, qkvg, bias)
        return run

    def __call__(self, qkvg, bias, block, S, H):
        L = qkvg.shape[0] // S
        key = (qkvg.data_ptr(), bias.data_ptr(), block, S, L)
        run = self.runs.get(key)
        if run is None:
            run = self.runs[key] = self._bind(qkvg, bias, block, S, L, H)
        run()
        return qkvg


def supported(dtype, L, D, H):
    return (torch.cuda.get_device_capability() == (10, 0) and dtype is torch.bfloat16 and D == 768 and H == 16 and L % 128 == 0
            and _CUBIN.exists())
