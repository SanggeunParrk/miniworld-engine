"""The H100 fused SWA atom block (h100_fused.py = team-gm 14f2c73 swa_fused_triton) with its three CUDA kernels served by the sm_100a
ports -- exactly the H100 configuration (qkvg forward, out-proj + FFN forward and FFN backward in CUDA, everything else in Triton).

    import b200_block; b200_block.install()      # h100_fused.swa_block / block_fwd / block_bwd now call qkvg_fwd.cu / ffn_fwd.cu / ffn_bwd.cu
"""
import torch
import h100_fused as SF
from ops import QkvgFwd, FfnFwd, FfnBwd, tiling

_PACK = {}


def cached(fn, *ts):
    """fn(*ts) cached on the tensors' storage and version (the packed weights are rebuilt only when a weight changes)."""
    key = (fn.__name__,) + tuple((t.data_ptr(), t._version, tuple(t.shape)) for t in ts)
    hit = _PACK.get(key)
    if hit is None:
        if len(_PACK) > 64:
            _PACK.clear()
        hit = _PACK[key] = fn(*ts)
    return hit


class _Qkvg:
    def __init__(self):
        self.k = QkvgFwd()

    def qkvg_fwd(self, qf, mod, cos, sin, W, A, B, S, eps, qk_eps, save):
        assert A >= 5, "the sm_100a tiling needs A >= 5 (smaller A keeps the Triton path: see install())"
        q = qf.view(A * B, S, -1)
        run, (Qh, Kh, Vh, G, X, PQ, PK) = self.k.bind(q, mod, cos, sin, None, None, A, B, save=save, W=W)
        run()
        return Qh, Kh, Vh, G, X, PQ, PK


class _Fwd:
    def __init__(self):
        self.k = FfnFwd()

    def ffn_fwd(self, qf, G, O, mod, wo, wab64, wd, A, B, S, eps, save):
        run, (out, q1, att, y, ffn) = self.k.bind(qf, G, O, mod, wo, wab64, wd, A, B, save=save, packed=True)
        run()
        return out, q1, att, y, ffn


class _Bwd:
    def __init__(self):
        self.k = FfnBwd()

    def ffn_bwd(self, dy, q1, Y, FF, mod, dmod, wab32, wdt, wabt, A, B, S, eps, _sp):
        run, (dq1, dffn, hh, dab, _) = self.k.bind(dy, q1, Y, FF, mod, None, None, A, B, dmod=dmod, packed=(wab32, wdt, wabt))
        run()
        return dq1, dffn, hh, dab


class _HoistTF32(torch.autograd.Function):
    """silu(c) Wmod^T in fp32. Both operands hold bf16 values (c and Wmod are bf16; silu is evaluated in bf16), which TF32 represents
    exactly, so the TF32 tensor-core GEMM forms the same fp32 products as the fp32 SIMT GEMM (only the summation order differs).
    The backward stays fp32 (dmod is not bf16-valued)."""
    @staticmethod
    def forward(ctx, a, w):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            out = a @ w.t()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        ctx.save_for_backward(a, w)
        return out

    @staticmethod
    def backward(ctx, g):
        a, w = ctx.saved_tensors
        return g @ w, g.t() @ a


def hoist_mod_tf32(c1, wmod):
    a = torch.nn.functional.silu(c1).float()
    return _HoistTF32.apply(a.reshape(-1, a.shape[-1]), wmod.float()).contiguous()


def install():
    SF.hoist_mod = hoist_mod_tf32
    # weight packing per call -> cached per weight version (block_fwd / block_bwd call these with the raw weights every time)
    _cat, _p64, _p = torch.cat, SF._pack_ffn64, SF._pack_ffn
    SF._pack_ffn64 = lambda wu: cached(_p64, wu)
    SF._pack_ffn = lambda wu, wd: cached(_p, wu, wd)
    SF.torch = type("T", (), {"__getattr__": lambda self, k: getattr(torch, k),
                              "cat": staticmethod(lambda ts, *a, **k: cached(lambda *x: _cat(list(x), *a, **k), *ts) if (not a and not k and
                                                  all(not t.requires_grad or True for t in ts) and all(t.dim() == 2 for t in ts)) else _cat(ts, *a, **k))})()
    SF._CUDA.update(qkvg=_Qkvg(), fwd=_Fwd(), bwd=_Bwd())
    SF._qkvg_fwd_cuda_ok = lambda C, H, dt: C == 128 and H == 4 and dt == torch.bfloat16
    SF._ffn_fwd_cuda_ok = lambda C, N, dt: C == 128 and N == 256 and dt == torch.bfloat16
    SF._ffn_bwd_cuda_ok = lambda C, N, dt: C == 128 and N == 256 and dt == torch.bfloat16 and SF.FFN_DW != "fused"
