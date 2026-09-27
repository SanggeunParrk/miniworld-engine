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


_MOD = None


def mod_ext():
    global _MOD
    if _MOD is None:
        d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "trimod"; d.mkdir(parents=True, exist_ok=True)
        _MOD = load("trimod", [str(_src / "triattn_mod_sm100.cu")], extra_include_paths=[str(_src)], build_directory=str(d),
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
    return _MOD


class _Fused(torch.autograd.Function):
    """The whole TriangleAttention (starting node, no dropout) on B200 kernels:
    front (LN + q|k|v|g + head-major masked bias) -> attention (+ LSE) -> tail (gate, to_out, residual);
    backward: gate_bwd (do, dg, delta, dWo) -> attention bwd (dq, dk, dv, dbias) -> head_bwd (dgrad + LN bwd + residual, y) -> cuBLAS dW."""
    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, H):
        B, L, _, C = pair.shape
        x = pair.contiguous().view(-1, C)
        w4 = torch.cat([wq, wk, wv, wg], 0).contiguous()
        m = mod_ext()
        q, k, v, g, bias = m.tri_front(x, lnw, lnb, eps, w4, wb, mask if mask is not None else torch.empty(0, device=pair.device, dtype=torch.bool), B, L)
        shp = (B, L, L, H, C // H)
        training = torch.is_grad_enabled() or ctx.needs_input_grad[0]
        o, lse, _ = ext().triattn_fwd(q.view(shp), k.view(shp), v.view(shp), bias, (C // H) ** -0.5, bool(any(ctx.needs_input_grad)))
        out = m.tri_tail(g, o.view(-1, C), x, wo.contiguous())
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x, lnw, lnb, w4, wb, wo, q, k, v, g, bias, o, lse)
            ctx.dims = (B, L, C, H, eps)
        return out.view(B, L, L, C)

    @staticmethod
    def backward(ctx, dy):
        x, lnw, lnb, w4, wb, wo, q, k, v, g, bias, o, lse = ctx.saved_tensors
        B, L, C, H, eps = ctx.dims
        m, e = mod_ext(), ext()
        dy = dy.contiguous().view(-1, C)
        do, dg, delta, dwo = m.tri_gate_bwd(dy, g, o.view(-1, C), wo.contiguous(), B, L)
        shp = (B, L, L, H, C // H)
        dq, dk, dv, db = e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, (C // H) ** -0.5)
        dq, dk, dv = dq.view(-1, C), dk.view(-1, C), dv.view(-1, C)
        dpair = dy.clone()
        y, dgam, dbet = m.tri_head_bwd(dq, dk, dv, dg, db, x, w4, wb, lnw, lnb, eps, dpair, B, L)
        dw = [t.t() @ y for t in (dq, dk, dv, dg)]
        dwb = db.view(B, H, -1).permute(1, 0, 2).reshape(H, -1).to(torch.bfloat16) @ y
        return (dpair.view(B, L, L, C), dgam.to(lnw.dtype), dbet.to(lnb.dtype), *(t.to(w4.dtype) for t in dw), dwb.to(wb.dtype),
                dwo.to(wo.dtype), None, None, None)


class TriangleAttentionB200Fused(TriangleAttention):
    def forward(self, pair, mask=None):
        if not self.starting:
            pair = rearrange(pair, "B I J D -> B J I D").contiguous()
        out = _Fused.apply(pair, self.ln_pair.weight, self.ln_pair.bias, self.to_query.weight, self.to_key.weight, self.to_value.weight,
                           self.to_gate.weight, self.to_bias.weight, self.to_out.weight, mask, self.ln_pair.eps, self.n_head)
        if not self.starting:
            out = rearrange(out, "B J I D -> B I J D").contiguous()
        return out
