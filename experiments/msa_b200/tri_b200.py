"""TriangleAttention on the B200 kernels: the attention core (forward with LSE, backward: KV-owned dK/dV + query-owned dQ/dbias) is
ours; LayerNorm, the five projections, the gate and to_out stay torch / cuBLAS for now (the next kernels to fuse)."""
import os, pathlib, torch
from torch.utils.cpp_extension import load
from einops import rearrange
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

_src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        tag = os.environ.get("TA_TAG", "cur")
        d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / f"triattn_{tag}"; d.mkdir(parents=True, exist_ok=True)
        _EXT = load(f"triattn_{tag}", [str(_src / "triattn_sm100.cu")], extra_include_paths=[str(_src)], build_directory=str(d),
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
    return _EXT


class _Core(torch.autograd.Function):
    """out = softmax(scale q.k + bias) v on [B, N, S, H, 32] tensors, bias [B, H, S, S] bf16."""
    @staticmethod
    def forward(ctx, q, k, v, bias, scale):
        e = ext()
        need = torch.is_grad_enabled() or any(t.requires_grad for t in (q, k, v, bias))
        out, lse, _ = e.triattn_fwd(q, k, v, bias, scale, bool(ctx.needs_input_grad[0] or ctx.needs_input_grad[3]) or need)
        ctx.save_for_backward(q, k, v, bias, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, bias, out, lse = ctx.saved_tensors
        e = ext()
        dout = dout.contiguous()
        delta = e.triattn_delta(dout, out)
        dq, dk, dv, db = e.triattn_bwd(q, k, v, bias, dout, lse, delta, ctx.scale)
        return dq, dk, dv, db.to(bias.dtype), None


class TriangleAttentionB200(TriangleAttention):
    def _attention(self, pair, mask=None):
        if not self.starting:
            pair = rearrange(pair, "B I J D -> B J I D").contiguous()
        B, L, _, _ = pair.shape
        H = self.n_head
        x = self.ln_pair(pair)
        q = self.to_query(x).view(B, L, L, H, -1)
        k = self.to_key(x).view(B, L, L, H, -1)
        v = self.to_value(x).view(B, L, L, H, -1)
        bias = self.to_bias(x).permute(0, 3, 1, 2)                       # [B, H, L (query j), L (key k)]
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
        bias = bias.to(torch.bfloat16).contiguous()
        o = _Core.apply(q, k, v, bias, q.shape[-1] ** -0.5).view(B, L, L, -1)
        out = self.to_out(torch.sigmoid(self.to_gate(x)) * o)
        if not self.starting:
            out = rearrange(out, "B J I D -> B I J D").contiguous()
        return out
