"""A100 atom DiT training through explicitly qualified CUDA + cuBLAS paths.

The default serving gate is the qualified A48,B1,N4096 atom geometry. N8192
remains the independent opt-in research path and is not served by this default.
Both residuals are owned by the same operators as the original modules. BF
activations and BF/FP32 master weights retain pinned FP32 norm affine weights.
Native casts, graph-live packs, fixed dQ/dKV schedules and both eager/compiled
fanout rounding are explicit; no benchmark environment flags enable this path.
"""
from __future__ import annotations
import warnings
import torch
from torch.autograd.function import once_differentiable
from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque,device_constant
from miniworld_engine.integrations import adaln_sm80 as ad
from miniworld_engine.integrations import augattn_sm80 as aa
from miniworld_engine.integrations import conditioned_transition_sm80 as ct
from miniworld_engine.integrations import atom_augattn_sm80_a100 as rolledaa
from miniworld_engine.integrations.atom_dit_sm80_join_fused import _bwd_fake
from miniworld_engine.integrations.conditioned_transition_sm80 import _adaln,_tail,_ATOM_TRAIN_ROW_THRESHOLD,_wgrad,Branch,_mm_op,_mm,like,BF

_FAILED=False
@device_constant
def _ready(index):
    global _FAILED
    if _FAILED:return False
    try:
        from miniworld_engine.kernels.augmented_attention.cuda import atom_core_qresident_sm80,atom_core_order_sm80,atom_post_grad_delta_sm80,atom_pair_reduce_vec_sm80,atom_pair_uniform_no_db_sm80,atom_sample_uniform_sm80,atom_qkvg_resident_sm80,atom_output_fused_sm80
        from miniworld_engine.kernels.adaln.cuda import atom_adaln_join_sm80
        from miniworld_engine.kernels.conditioned_transition.cuda import ct_atom_cond_wgrad_fused_sm80
        from miniworld_engine.kernels.transition.cuda import parameter_cast_sm80 as cast
        for m in (atom_core_qresident_sm80,atom_core_order_sm80,atom_post_grad_delta_sm80,atom_pair_reduce_vec_sm80,atom_pair_uniform_no_db_sm80,atom_sample_uniform_sm80,atom_qkvg_resident_sm80,atom_output_fused_sm80,atom_adaln_join_sm80,ct_atom_cond_wgrad_fused_sm80,cast):m._ext()
    except Exception as exc:
        _FAILED=True
        warnings.warn(f'A100 atom training CUDA kernels unavailable: {exc!r}',RuntimeWarning,stacklevel=2)
        return False
    return True

def serves(module,single,cond,pair,mask,compute_dtype=None):
    from miniworld_engine.modules.dispatch import KernelBackend
    if _FAILED or not settings.current().atom_dit_sm80 or settings.current().engine_backend=='triton':return False
    if not torch.is_grad_enabled() or module.attention._backend != KernelBackend.TRITON:return False
    if module.attention.use_qk_norm or compute_dtype not in (None,torch.bfloat16):return False
    if not single.is_cuda or torch.cuda.get_device_capability(single.device)!=(8,0):return False
    if tuple(single.shape)!=(48,1,4096,128) or tuple(cond.shape)!=tuple(single.shape) or tuple(pair.shape)!=(1,4096,4096,16):return False
    if any(t.dtype is not torch.bfloat16 or t.device!=single.device for t in (single,cond,pair)):return False
    if aa._mask_kind(mask,48,1,4096) is None or (mask is not None and mask.device!=single.device):return False
    if (module.attention.n_head,module.attention.to_query.weight.shape[0],module.transition.expand_a.weight.shape[0])!=(4,128,256):return False
    if any(n.eps!=1e-5 for n in (module.attention.ada_ln_in.ln_in,module.attention.ada_ln_in.ln_cond,module.attention.ln_pair,module.transition.ada_ln_in.ln_in,module.transition.ada_ln_in.ln_cond)):return False
    dtype=module.attention.to_query.weight.dtype
    if dtype not in (torch.bfloat16,torch.float32):return False
    for name,p in module.named_parameters():
        if p.device!=single.device or p.dtype is not (torch.float32 if name.endswith('ln_cond.weight') or name=='attention.ln_pair.weight' else dtype):return False
    if settings.current().condtrans_sm80_save_param_cast or settings.current().adaln_sm80_save_param_cast:return False
    # These two standard debug projection overrides must agree with the
    # fixed qualified schedules; otherwise retain the original module route.
    s=settings.current()
    if s.augattn_sm80_atom_qkvg not in (None,True) or s.augattn_sm80_atom_qkvg_tiled not in (None,False) or s.augattn_sm80_atom_output_fused not in (None,True):return False
    return aa.serves(module.attention,single,pair,mask,compute_dtype,cond) and _ready(single.device.index if single.device.index is not None else torch.cuda.current_device())

@opaque(fake=ct._train_bwd_fake, name="atom_dit_sm80_transition_bwd")
def _transition_bwd(dy: torch.Tensor, x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws1: torch.Tensor, sb1: torch.Tensor, wb1: torch.Tensor, wa: torch.Tensor,
              wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor, aff: torch.Tensor, cst: torch.Tensor, sbt: torch.Tensor, xst: torch.Tensor,
              xa: torch.Tensor, ab: torch.Tensor, z: torch.Tensor, g: torch.Tensor, wcat: torch.Tensor, wab: torch.Tensor, residual: bool, fp32_grads: bool = False, native_cast: bool = False) -> list[torch.Tensor]:
    """[dx, dcond, d lnw, dWs1, d sb1, dWb1, dWa, dWb, dWs, dWsc, d bsc] (each in its input's dtype; the weights' and biases' fp32 when ``fp32_grads``: fp32 master parameters, unrounded) from the gradient dy [M, d] and the forward's saved tensors."""
    ka, kt = _adaln(), _tail()
    d, nd = x.shape[1], wa.shape[0]
    with torch.cuda.device(x.device):
        if native_cast:
            from miniworld_engine.kernels.transition.cuda.parameter_cast_sm80 import result
            ws1, sb1, wb1, wa, wb, ws, wsc, bsc = result((ws1, sb1, wb1, wa, wb, ws, wsc, bsc))
        if x.shape[0] > _ATOM_TRAIN_ROW_THRESHOLD and kt.atom_supported(x, cond, wa) and ka.atom_supported(x, cond):          # same decomposition as train_fwd and its fake
            dz, dg, dcond2, hh, dab, pbsc = kt.atom_bwd_gate(dy, z, cond, xa, wab, ws, wsc, bsc)
            dxa = kt.atom_bwd_dxa(dab, wab)
            if (x.shape[0] == 48 * 8192 and dz.dtype is BF
                    and dz.shape[1] == 128 and hh.shape[1] == 256):
                if True:
                    from miniworld_engine.kernels.conditioned_transition.cuda import ct_atom_wgrad_wide_sm80
                    dws = ct_atom_wgrad_wide_sm80.result(dz, hh)
                else:
                    from miniworld_engine.kernels.conditioned_transition.cuda import ct_atom_wgrad_split_sm80
                    dws = ct_atom_wgrad_split_sm80.result(dz, hh)
            else:
                dws = _wgrad(dz, hh)
            fused_cond_wgrad = x.shape[0] in (48 * 4096, 48 * 8192)
            dwab = _wgrad(dab, xa)
            if not fused_cond_wgrad:
                dwsc = _wgrad(dg, cond)
            dx, dcond, dsc, aff, psb1, plw = ka.atom_bwd(dxa, x, cond, xst, cst, lnw, ws1, wb1, sb1, dy if residual else None, dcond2)
            if fused_cond_wgrad:
                from miniworld_engine.kernels.conditioned_transition.cuda import ct_atom_cond_wgrad_fused_sm80
                dws1, dwb1, dwsc = ct_atom_cond_wgrad_fused_sm80.result(dsc, dxa, dg, aff, cond)
            else:
                dws1, dwb1 = _wgrad(dsc, aff), _wgrad(dxa, aff)
        else:
            dz, dg, pbsc = kt.gate_res_bwd(dy, z, g)
            br = Branch(x.device)                                          # the weight gradients (and the gate's data gradient) beside the data gradient's chain of row passes and GEMMs
            dwsc = br.run(lambda: _wgrad(dg, cond), dg, cond)
            dcond2 = br.run(lambda: _mm_op(dg, wsc), dg, wsc)
            dcond2_ready = br.mark()
            dab, h = kt.swiglu_bwd(_mm_op(dz, ws), ab)                     # dh = dz Ws [M, n d] stays in the operand dtype
            dws = br.run(lambda: _wgrad(dz, h), dz, h)
            dwab = br.run(lambda: _wgrad(dab, xa), dab, xa)
            dxa = _mm_op(dab, wab)                                         # the AdaLN's dy [M, d]
            dm, dx, psb1 = ka.bwd_x(dxa, x, xst, sbt, sb1, dy if residual else None, x.dtype)
            dw1 = br.run(lambda: _wgrad(dm, aff), dm, aff)
            dca = _mm(dm, wcat)
            br.wait(dcond2_ready, dcond2)
            dcond, plw = ka.cond_bwd(dca, cond, cst, lnw, dcond2)
            br.join()
            dws1, dwb1 = dw1[:d], dw1[d:]
        dbsc, dsb1, dlnw, dws1, dwb1, dwa, dwb, dws, dwsc = ka.finish(            # the column sums and the casts of the weight gradients: one launch
            [pbsc, psb1, plw], [like(bsc, fp32_grads), like(sb1, fp32_grads), lnw], [dws1, dwb1, dwab[:nd], dwab[nd:], dws, dwsc],
            [like(t, fp32_grads) for t in (ws1, wb1, wa, wb, ws, wsc)])
    return [dx, dcond, dlnw, dws1, dsb1, dwb1, dwa, dwb, dws, dwsc, dbsc]

@opaque(fake=_bwd_fake,name='atom_dit_sm80_adaln_fanout_bwd_production')
def _bwd(dy: torch.Tensor,x: torch.Tensor,cond: torch.Tensor,lnw: torch.Tensor,ws: torch.Tensor,sb: torch.Tensor,wb: torch.Tensor,aff: torch.Tensor,cst: torch.Tensor,sbt: torch.Tensor,xst: torch.Tensor,wcat: torch.Tensor,res: torch.Tensor,dc_transition: torch.Tensor,dc_attention: torch.Tensor,masters: bool,native: bool,compiled: bool) -> list[torch.Tensor]:
    from miniworld_engine.kernels.adaln.cuda import atom_adaln_join_sm80 as fused
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_atom_cond_wgrad_fused_sm80 as wg
    if native:
        from miniworld_engine.kernels.transition.cuda.parameter_cast_sm80 import result
        ws,sb,wb=result((ws,sb,wb))
    dx,dc,dsc,aff,psb,plw=fused.result(dy,x,cond,xst,cst,lnw,ws,wb,sb,res,dc_transition,dc_attention,not compiled)
    if dsc.shape[0] == 48 * 4096:
        dws,dwb=wg.result_two(dsc,dy,aff)
    else:
        dws,dwb=ad._wgrad(dsc,aff),ad._wgrad(dy,aff)
    dsb,dlnw,dws,dwb=ad._kernels().finish([psb,plw],[ad.like(sb,masters),lnw],[dws,dwb],[ad.like(ws,masters),ad.like(wb,masters)])
    return [dx,dc,dlnw,dws,dsb,dwb]

class _Whole(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,cond,pair,kmask,kind,compiled,eps_ax,eps_ac,eps_tx,eps_tc,*params):
        a,b,n,d=x.shape
        aps=list(params[:4]);pps=list(params[4:14]);tps=list(params[14:])
        assert len(params)==23 and len(tps)==9
        masters=aps[1].dtype is torch.float32
        native=masters
        xs=x.reshape(-1,d);cs=cond.reshape(-1,d)
        normalized,*saved_a=ad.train_fwd(xs,cs,*aps,eps_ax,eps_ac,native,False)
        normalized=normalized.view(x.shape)
        attended,*saved_p=rolledaa._fwd(normalized,cond,pair,x,kmask,None,pps,a,b,n,4,kind,True)
        result,*saved_t=ct.train_fwd(attended.reshape(-1,d),cs,*tps,eps_tx,eps_tc,True,native,False)
        ctx.save_for_backward(x,cond,pair,*params,normalized,attended,*saved_a,*saved_p,*saved_t)
        ctx.meta=(kmask,kind,compiled,a,b,n,d,masters,native)
        return result.view(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        v=ctx.saved_tensors
        x,cond,pair=v[:3];params=list(v[3:26]);normalized,attended=v[26:28]
        saved_a=list(v[28:33]);saved_p=list(v[33:40]);saved_t=list(v[40:])
        assert len(saved_a)==5 and len(saved_p)==7 and len(saved_t)==10
        kmask,kind,compiled,a,b,n,d,masters,native=ctx.meta
        aps=params[:4];pps=params[4:14];tps=params[14:]
        xs=x.reshape(-1,d);cs=cond.reshape(-1,d)
        tgrads=_transition_bwd(dy.contiguous().reshape(-1,d),attended.reshape(-1,d),cs,*tps,*saved_t,True,masters,native)
        res,dc_transition=tgrads[:2]
        pgrads=rolledaa._bwd(res.view(x.shape),normalized,cond,pair,kmask,pps,saved_p,a,b,n,4,kind)
        normdy,dc_attention,dpair=pgrads[:3]
        agrads=_bwd(normdy.contiguous().reshape(-1,d),xs,cs,*aps,*saved_a,res,dc_transition,dc_attention.contiguous().reshape(-1,d),masters,native,compiled)
        return (agrads[0].view(x.shape),agrads[1].view(cond.shape),dpair,None,None,None,None,None,None,None,*agrads[2:],*pgrads[3:],*tgrads[2:])

def block(module,single,cond,pair,mask=None):
    assert single.shape[:2]==(48,1) and single.shape[2] == 4096
    assert single.shape[-1]==128 and cond.shape==single.shape and cond.dtype is single.dtype is torch.bfloat16
    assert pair.shape==(1,single.shape[2],single.shape[2],16)
    assert not settings.current().condtrans_sm80_save_param_cast
    assert not settings.current().adaln_sm80_save_param_cast
    ada=module.attention.ada_ln_in
    aps=[ada.ln_cond.weight,ada.to_scale.weight,ada.to_scale.bias,ada.to_bias.weight]
    kind=aa._mask_kind(mask,*single.shape[:3])
    kmask=None if kind==0 else (mask[0] if mask.ndim==3 and kind==1 else mask)
    eps=(float(ada.ln_in.eps),float(ada.ln_cond.eps),float(module.transition.ada_ln_in.ln_in.eps),float(module.transition.ada_ln_in.ln_cond.eps))
    return _Whole.apply(single,cond,pair,kmask,kind,torch.compiler.is_compiling(),*eps,*aps,*aa._params(module.attention),*ct._params(module.transition))
