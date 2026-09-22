"""SM90 implementation of Triton B9+B10 with identical rounding and buffers.

Explicit TMA loads and WGMMA. Logical BLOCK_M1 below the instruction's 64-row
minimum is padded in shared memory; output is predicated to the logical tile.
Stages are real independent shared-memory K slots, loaded as bounded batches.
No operand transposes are materialized. The second GEMM accepts the production
column-major dconc.T and column-major W_stack.T views directly.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
import torch
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from quack import copy_utils


class DualBackwardSm90:
    def __init__(self, n, kg, kp, BLOCK_M1, BLOCK_N, BLOCK_K, GROUP_M,
                 num_warps=4, num_stages=2):
        self.n, self.kg, self.kp = n, kg, kp
        self.logical_m, self.bn, self.bk = BLOCK_M1, BLOCK_N, BLOCK_K
        self.group, self.warps, self.stages = GROUP_M, num_warps, num_stages
        self.pm = max(BLOCK_M1, 64 * (num_warps // 4))
        self.shared_storage = None

    @cute.kernel
    def kernel(self, ag, tg, af, tf, aw, tw, av, tv, y,
               lg: cute.ComposedLayout, lf: cute.ComposedLayout,
               lw: cute.ComposedLayout, lv: cute.ComposedLayout,
               mg: cute.TiledMma, mf: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        nm = cute.ceil_div(y.shape[0], self.logical_m)
        nn = cute.ceil_div(self.n, self.bn)
        first = pid // (self.group * nn) * self.group
        actual = min(self.group, nm - first)
        local = pid % (self.group * nn)
        mi, ni = first + local % actual, local // actual
        # TMA origin uses logical tile size while the descriptor loads physical
        # m64/m128. Oversized rows are never stored and cannot affect other rows.
        row_origin = mi * self.logical_m
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sg = storage.sg.get_tensor(lg.outer, swizzle=lg.inner)
        sf = storage.sf.get_tensor(lf.outer, swizzle=lf.inner)
        sw = storage.sw.get_tensor(lw.outer, swizzle=lw.inner)
        sv = storage.sv.get_tensor(lv.outer, swizzle=lv.inner)
        bar = storage.bar.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(ag); cpasync.prefetch_descriptor(af)
                cpasync.prefetch_descriptor(aw); cpasync.prefetch_descriptor(av)
                cute.arch.mbarrier_init(bar, 1)
        cute.arch.mbarrier_init_fence(); cute.arch.barrier()
        # domain_offset moves the TMA coordinate origin, not the base pointer.
        gg = cute.local_tile(cute.domain_offset((row_origin, 0), tg), (self.pm,self.bk), (0,None))
        gf = cute.local_tile(cute.domain_offset((row_origin, 0), tf), (self.pm,self.bk), (0,None))
        gw = cute.local_tile(tw, (self.bn,self.bk), (ni,None))
        gv = cute.local_tile(tv, (self.bn,self.bk), (ni,None))
        loadg,_,_ = copy_utils.tma_get_copy_fn(ag,0,cute.make_layout(1),gg,sg)
        loadf,_,_ = copy_utils.tma_get_copy_fn(af,0,cute.make_layout(1),gf,sf)
        loadw,_,_ = copy_utils.tma_get_copy_fn(aw,0,cute.make_layout(1),gw,sw)
        loadv,_,_ = copy_utils.tma_get_copy_fn(av,0,cute.make_layout(1),gv,sv)
        thrg, thrf = mg.get_slice(tidx), mf.get_slice(tidx)
        accg = cute.make_fragment(thrg.partition_shape_C((self.pm,self.bn)),Float32)
        accf = cute.make_fragment(thrf.partition_shape_C((self.pm,self.bn)),Float32)
        accg.fill(0);accf.fill(0)
        xg=thrg.make_fragment_A(thrg.partition_A(sg)); wg=thrg.make_fragment_B(thrg.partition_B(sw))
        xf=thrf.make_fragment_A(thrf.partition_A(sf)); vf=thrf.make_fragment_B(thrf.partition_B(sv))
        mag, maf = cute.make_mma_atom(mg.op), cute.make_mma_atom(mf.op)
        mag.set(warpgroup.Field.ACCUMULATE, True);maf.set(warpgroup.Field.ACCUMULATE,True)
        phase=Int32(0)
        # The gate GEMM completes before rounding; the front GEMM is accumulated
        # separately. Batches recycle stage slots only after WGMMA wait_group(0).
        for kind in cutlass.range_constexpr(2):
            if cutlass.const_expr(kind==0):
                loops=cutlass.const_expr((self.kg+self.bk-1)//self.bk)
            else:
                loops=cutlass.const_expr((self.kp+self.bk-1)//self.bk)
            for base in cutlass.range(0,loops,self.stages):
                count=min(self.stages,loops-base)
                if warp==0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(bar,count*(self.pm+self.bn)*self.bk*2)
                    for s in cutlass.range(self.stages):
                        if s<count:
                            if cutlass.const_expr(kind==0):
                                loadg(src_idx=base+s,dst_idx=s,tma_bar_ptr=bar)
                                loadw(src_idx=base+s,dst_idx=s,tma_bar_ptr=bar)
                            else:
                                loadf(src_idx=base+s,dst_idx=s,tma_bar_ptr=bar)
                                loadv(src_idx=base+s,dst_idx=s,tma_bar_ptr=bar)
                cute.arch.mbarrier_wait(bar,phase)
                phase=phase^1
                cute.arch.fence_view_async_shared();warpgroup.fence()
                for s in cutlass.range(self.stages):
                    if s<count:
                        for k in cutlass.range_constexpr(cute.size(xg.shape[2])):
                            if cutlass.const_expr(kind==0):
                                cute.gemm(mag,accg,xg[None,None,k,s],wg[None,None,k,s],accg)
                            else:
                                cute.gemm(maf,accf,xf[None,None,k,s],vf[None,None,k,s],accf)
                warpgroup.commit_group();warpgroup.wait_group(0)
                cute.arch.barrier()
        coords=thrg.partition_C(cute.make_identity_tensor((self.pm,self.bn)))
        result=(accf.load()+accg.load().to(BFloat16).to(Float32)).to(BFloat16)
        out=cute.make_fragment_like(accg,BFloat16);out.store(result)
        for i in cutlass.range(cute.size(out),unroll_full=True):
            row=row_origin+coords[i][0];col=ni*self.bn+coords[i][1]
            if coords[i][0]<self.logical_m and row<y.shape[0] and col<self.n:
                y[row,col]=out[i]

    @cute.jit
    def __call__(self,g,f,w,v,y,stream:cuda.CUstream):
        def layout(rows,major):
            orient=LayoutEnum.ROW_MAJOR if major else LayoutEnum.COL_MAJOR
            atom=warpgroup.make_smem_layout_atom(sm90h.get_smem_layout_atom(orient,BFloat16,self.bk if major else rows),BFloat16)
            return cute.tile_to_shape(atom,(rows,self.bk,self.stages),order=(0,1,2) if major else (1,0,2))
        lg,lf=layout(self.pm,True),layout(self.pm,False)
        lw,lv=layout(self.bn,True),layout(self.bn,False)
        ag,tg=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),g,cute.slice_(lg,(None,None,0)),(self.pm,self.bk))
        af,tf=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),f,cute.slice_(lf,(None,None,0)),(self.pm,self.bk))
        aw,tw=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),w,cute.slice_(lw,(None,None,0)),(self.bn,self.bk))
        av,tv=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),v,cute.slice_(lv,(None,None,0)),(self.bn,self.bk))
        mg=sm90h.make_trivial_tiled_mma(BFloat16,BFloat16,warpgroup.OperandMajorMode.K,warpgroup.OperandMajorMode.K,Float32,(self.warps//4,1,1),(64,self.bn))
        mf=sm90h.make_trivial_tiled_mma(BFloat16,BFloat16,warpgroup.OperandMajorMode.MN,warpgroup.OperandMajorMode.MN,Float32,(self.warps//4,1,1),(64,self.bn))
        sg_type=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lg)],1024]
        sf_type=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lf)],1024]
        sw_type=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lw)],1024]
        sv_type=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lv)],1024]
        @cute.struct
        class Storage:
            bar:cute.struct.MemRange[cutlass.Int64,1]
            sg:sg_type
            sf:sf_type
            sw:sw_type
            sv:sv_type
        self.shared_storage=Storage
        self.kernel(ag,tg,af,tf,aw,tw,av,tv,y,lg,lf,lw,lv,mg,mf).launch(
            grid=[cute.ceil_div(y.shape[0],self.logical_m)*cute.ceil_div(self.n,self.bn),1,1],
            block=[32*self.warps,1,1],stream=stream)


_COMPILE_CACHE={}
DEFAULT_CONFIG=dict(BLOCK_M1=64,BLOCK_N=128,BLOCK_K=64,GROUP_M=1,num_warps=4,num_stages=2)


def feasibility(config, smem_limit=232448):
    """Reject unsupported physical launches explicitly; never change tile axes."""
    bm,bn,bk=(config[k] for k in ('BLOCK_M1','BLOCK_N','BLOCK_K'))
    if bm < 64:
        return 'WGMMA has a minimum 64-row instruction tile'
    if config['num_warps'] != bm // 16:
        return 'implementation requires one 4-warp group per 64-row tile'
    sizes=[bm*bk*config['num_stages']*2,bn*bk*config['num_stages']*2]*2
    usage=1024+sum((s+1023)//1024*1024 for s in sizes)
    if usage > smem_limit:
        return f'shared memory {usage} exceeds device limit {smem_limit}'
    return None


def input_dual_bwd_sm90_impl(g,f,w,v,length,config=None):
    """Same API and matrix views as Triton input_dual_bwd; explicit config first."""
    m,kg=g.shape;kp=f.shape[1];n=w.shape[1]
    if f.shape[0]!=m or w.shape[0]!=kg or v.shape!=(kp,n):
        raise ValueError('dual dgrad shape mismatch')
    if any(t.dtype!=torch.bfloat16 or t.device!=g.device or not t.is_cuda for t in (g,f,w,v)):
        raise ValueError('requires BF16 CUDA operands on same device')
    if g.stride(1)!=1 or f.stride(0)!=1 or w.stride(0)!=1 or v.stride(1)!=1:
        raise ValueError('requires production row/column-major operand views')
    if any(t.data_ptr()%16 or any(s%8 for s in t.stride() if s!=1) for t in (g,f,w,v)):
        raise ValueError('TMA requires 16-byte alignment and aligned non-unit strides')
    if torch.cuda.get_device_capability(g.device)!=(9,0):
        raise ValueError('requires Hopper SM90')
    y=g.new_empty((m,n))
    tensors=(g,f,w.t(),v.t(),y)
    limit=torch.cuda.get_device_properties(g.device).shared_memory_per_block_optin
    def launch(c):
        reason=feasibility(c,limit)
        if reason is not None:
            raise ValueError(reason)
        args=[from_dlpack(t,assumed_align=16) for t in tensors]
        args.append(cuda.CUstream(torch.cuda.current_stream(g.device).cuda_stream))
        key=(str(g.device),tuple((tuple(t.shape),t.stride()) for t in tensors),tuple(sorted(c.items())))
        if key not in _COMPILE_CACHE:
            _COMPILE_CACHE[key]=cute.compile(DualBackwardSm90(n,kg,kp,**c),*args)
        _COMPILE_CACHE[key](*args)
    if config is None:
        from miniworld_engine.autotune.trimul_sm90_config import resolve
        config=resolve('trimul_parity_dual_bwd_sm90_cute',tensors[:-1],extra=(length,limit),
            feasibility=lambda c:feasibility(c,limit),run=launch)
    launch(config)
    return y


def _fake(g,f,w,v,length):
    return g.new_empty((g.shape[0],w.shape[1]))


from miniworld_engine.kernels._compile import opaque


@opaque(fake=_fake,name='trimul_parity_dual_bwd_sm90')
def input_dual_bwd_sm90(g: torch.Tensor,f: torch.Tensor,w: torch.Tensor,v: torch.Tensor,length: int) -> torch.Tensor:
    return input_dual_bwd_sm90_impl(g,f,w,v,length)
