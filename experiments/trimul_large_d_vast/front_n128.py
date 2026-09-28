"""Two gate/projection groups per m64n128 WGMMA, original K order and epilogues."""
import copy
import re
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from wide_saved_front_overlap import SavedOverlapFront

def build(plan, cfg=(1,64,2,1,2,2)):
    d,n=plan.p.D,plan.p.n
    assert d in (256,384,512) and n==384 and cfg[5]==2
    a,b,slots,sk,mb=cfg[:5]
    base=copy.copy(plan.f.front if d==256 else plan.baseline_front)
    base.cfg=list(cfg);base.smem=F.k1_smem(d,cfg)
    extra=slots*sk*8192
    smem=base.smem+a*b//64*4096*(1 if d!=256 else -1)+extra
    if smem*mb>232448:raise ValueError('shared-memory feasibility')
    base.threads=128*(a*b//64+1);tj=n//b;tiles=n//a*tj
    base.grid=min(tiles,torch.cuda.get_device_properties(plan.p.x.device).multi_processor_count*mb)
    fields=base.params.fields.copy()
    fields[0]=F.tm(base.x if base.normalize else base.xn,[64,b,a],[d,n,n],[d*2,n*d*2])
    fields[1]=F.tm(plan.f.w,[64,128],[d,8*d],[d*2])
    fields[11:13]=[tj,tiles]
    base.params=T._launch_module().Struct(fields)
    original_compile=T.compile_text
    def compile_n128(source,flags):
        pos=source.index('TMN_DEVI void mw_shared_mma')
        head,body=source[:pos],source[pos:]
        stop=body.index('template <class G,')
        regs=','.join('%%%d'%i for i in range(64))
        outputs=','.join('"+f"(v[%d])'%i for i in range(64))
        helper='TMN_DEVI void mw_shared_mma(float (&v)[64],uint64_t a,uint64_t b,int ac){\n'
        helper+='asm volatile("{.reg .pred p;setp.ne.b32 p,%66,0;wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {'+regs+'},%64,%65,p,1,1,0,0;}" : '+outputs+' : "l"(a),"l"(b),"r"(ac));\n}\n'
        body=helper+body[stop:]
        if d==256:
            # D256 recomputes the preactivations in backward. Preserve only its
            # already-qualified input-statistics save, without a new pre buffer.
            start=body.index('      static_assert(TMAST,')
            end=body.index('      uint32_t pk[4][2];',start)
            body=body[:start]+body[end:]
            body=body.replace('cw * 12288 + 8192','cw * 4096')
            body=body.replace('cw*12288+8192','cw*4096')
            body=body.replace('tma_store_wait_read<1>()','tma_store_wait_read<0>()')
            body=body.replace('SMEM_STAGE=NCWG*12288','SMEM_STAGE=NCWG*4096')
            body=body.replace('<C,true,1,false,true,1>','<C,true,1,true,true,1>')
        body=body.replace('NBLK = G::NBLK,','NBLK = G::NBLK / 2,')
        body=body.replace('SLOT_BYTES = G::SLOT_BYTES,','SLOT_BYTES = G::SLOT_BYTES * 2,')
        body=body.replace('sW + G::SMEM_W;','sW + G::SMEM_W * 2;')
        body=body.replace('kk * 8192, &p.tm_w','kk * 16384, &p.tm_w')
        body=body.replace('64 * b);','128 * b);')
        body=body.replace('K*8192+Q*32','K*16384+Q*32')
        body=body.replace('float acc0[32], acc1[32];','float acc0[64], acc1[64];')
        body=body.replace('auto issue_block = [&](float (&ac)[32]', 'auto issue_block = [&](float (&ac)[64]')
        body=body.replace('i < 32; ++i) ac[i]', 'i < 64; ++i) ac[i]')
        body=body.replace('i < 32; ++i) acc0[i]', 'i < 64; ++i) acc0[i]')
        def expand(match):
            acc,block,wi,release=match.groups()
            return (f'epilogue(*reinterpret_cast<float(*)[32]>({acc}),2*({block}),{wi},false);'
                    f'epilogue(*reinterpret_cast<float(*)[32]>({acc}+32),2*({block})+1,{wi},{release});')
        body=re.sub(r'epilogue\((acc[01]), (b(?: \+ 1)?), (w_iter(?: \+ SPB)?), (false|true)\);',expand,body)
        # The original wrapper adds one preactivation staging slot per consumer.
        marker='static constexpr int SMEM=Base::SMEM+NCWG*4096;'
        assert marker in body
        body=body.replace(marker,'static constexpr int SMEM=Base::SMEM'+('+' if d!=256 else '-')+'NCWG*4096+Base::SMEM_W;')
        return original_compile(head+body,flags)
    T.compile_text=compile_n128
    unused=plan.p.x.new_empty((64,64)) if d==256 else plan.pre
    try:op=SavedOverlapFront(base,unused)
    finally:T.compile_text=original_compile
    op.smem=smem;op.k.set_max_dynamic_smem(smem)
    drv=op.k.unit.drv;fn=drv.d.CUfunction(int(op.k.handle))
    op.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,op.threads,smem)))
    return op
