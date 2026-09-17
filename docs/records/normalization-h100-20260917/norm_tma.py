"""Experimental SM90 CuTe TMA normalization; no production dispatch until qualified."""
import operator
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
from cutlass import BFloat16,Float32,Int32
from cutlass.cute.nvgpu import cpasync,warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from cuda.bindings import driver as cuda
from quack import copy_utils

class NormTma:
 def __init__(self,n,bm,warps,stages,grid,bwd):
  self.n,self.bm,self.warps,self.stages,self.grid,self.bwd=n,bm,warps,stages,grid,bwd
 @cute.jit
 def __call__(self,x,dy,w,b,mean,rstd,out,dw,db,eps:Float32,stream:cuda.CUstream):
  atom=warpgroup.make_smem_layout_atom(sm90h.get_smem_layout_atom(LayoutEnum.COL_MAJOR,BFloat16,self.bm),BFloat16)
  lx=cute.tile_to_shape(atom,(self.bm,self.n,self.stages),order=(1,0,2))
  lo=cute.tile_to_shape(atom,(self.bm,self.n),order=(1,0))
  ax,tx=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),x,cute.slice_(lx,(None,None,0)),(self.bm,self.n))
  if cutlass.const_expr(self.bwd):
   ay,ty=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),dy,cute.slice_(lx,(None,None,0)),(self.bm,self.n))
   ao,to=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(),out,lo,(self.bm,self.n))
  else:
   ay,ty,ao,to=ax,tx,ax,out
  st=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lx)],1024]
  ot=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lo)],1024]
  yt=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lx) if self.bwd else 1],1024]
  dt=cute.struct.MemRange[Float32,2*self.warps*self.n if self.bwd else 1]
  @cute.struct
  class Storage:
   barriers:cute.struct.MemRange[cutlass.Int64,self.stages]
   x:st
   y:yt
   o:ot
   partial:dt
  self.storage=Storage
  self.kernel(ax,tx,ay,ty,ao,to,w,b,mean,rstd,out,dw,db,lx,lo,eps).launch(grid=[self.grid,1,1],block=[self.warps*32,1,1],stream=stream)
 @cute.kernel
 def kernel(self,ax,tx,ay,ty,ao,to,w,b,mean,rstd,out,dw,db,lx,lo,eps):
  tid,_,_=cute.arch.thread_idx();pid,_,_=cute.arch.block_idx()
  warp=cute.arch.make_warp_uniform(cute.arch.warp_idx());lane=tid%32
  storage=cutlass.utils.SmemAllocator().allocate(self.storage)
  sx=storage.x.get_tensor(lx.outer,swizzle=lx.inner)
  so=storage.o.get_tensor(lo.outer,swizzle=lo.inner)
  if cutlass.const_expr(self.bwd):
   sy=storage.y.get_tensor(lx.outer,swizzle=lx.inner)
   partial=storage.partial.get_tensor(cute.make_layout((2,self.warps,self.n),stride=(self.warps*self.n,self.n,1)))
  bar=storage.barriers.data_ptr()
  if warp==0:
   with cute.arch.elect_one():
    cpasync.prefetch_descriptor(ax)
    if cutlass.const_expr(self.bwd):cpasync.prefetch_descriptor(ay)
    for j in cutlass.range_constexpr(self.stages):cute.arch.mbarrier_init(bar+j,1)
  cute.arch.mbarrier_init_fence();cute.arch.barrier()
  gx=cute.local_tile(tx,(self.bm,self.n),(None,0))
  loadx,_,_=copy_utils.tma_get_copy_fn(ax,0,cute.make_layout(1),gx,sx)
  if cutlass.const_expr(self.bwd):
   gy=cute.local_tile(ty,(self.bm,self.n),(None,0))
   loady,_,_=copy_utils.tma_get_copy_fn(ay,0,cute.make_layout(1),gy,sy)
  values=cute.make_rmem_tensor((self.n//32,),Float32)
  gamma=cute.make_rmem_tensor((self.n//32,),Float32)
  other=cute.make_rmem_tensor((self.n//32,),Float32)
  accumw=cute.make_rmem_tensor((self.n//32,),Float32);accumb=cute.make_rmem_tensor((self.n//32,),Float32)
  accumw.fill(0);accumb.fill(0)
  for c in cutlass.range_constexpr(self.n//32):gamma[c]=w[lane+c*32].to(Float32)
  tiles=cute.ceil_div(tx.shape[0],self.bm)
  if warp==0:
   for j in cutlass.range_constexpr(self.stages):
    if pid+j*self.grid<tiles:
     with cute.arch.elect_one():cute.arch.mbarrier_arrive_and_expect_tx(bar+j,self.bm*self.n*(4 if self.bwd else 2))
     loadx(src_idx=pid+j*self.grid,dst_idx=j,tma_bar_ptr=bar+j)
     if cutlass.const_expr(self.bwd):loady(src_idx=pid+j*self.grid,dst_idx=j,tma_bar_ptr=bar+j)
  it=Int32(0)
  for tile in cutlass.range(pid,tiles,self.grid):
   stage=it%self.stages
   cute.arch.mbarrier_wait(bar+stage,(it//self.stages)&1)
   if cutlass.const_expr(self.bwd):
    if warp==0:
     with cute.arch.elect_one():cute.arch.cp_async_bulk_wait_group(0,read=True)
   cute.arch.barrier()
   for ri in cutlass.range_constexpr(self.bm//self.warps):
    rowlocal=warp+ri*self.warps;row=tile*self.bm+rowlocal
    if row<tx.shape[0]:
     s=Float32(0);s2=Float32(0)
     if cutlass.const_expr(self.bwd):
      mu=mean[row].to(Float32);rs=rstd[row].to(Float32)
      for c in cutlass.range_constexpr(self.n//32):
       values[c]=(sx[rowlocal,lane+c*32,stage].to(Float32)-mu)*rs
       other[c]=sy[rowlocal,lane+c*32,stage].to(Float32)
       wd=other[c]*gamma[c];s=s+wd*values[c];s2=s2+wd
      c1=cute.arch.warp_reduction(s,operator.add)/self.n
      c2=cute.arch.warp_reduction(s2,operator.add)/self.n
      for c in cutlass.range_constexpr(self.n//32):
       dx=(other[c]*gamma[c]-(values[c]*c1+c2))*rs
       so[rowlocal,lane+c*32]=dx.to(BFloat16)
       accumw[c]=accumw[c]+other[c]*values[c];accumb[c]=accumb[c]+other[c]
     else:
      for c in cutlass.range_constexpr(self.n//32):
       values[c]=sx[rowlocal,lane+c*32,stage].to(Float32);s=s+values[c]
      mu=cute.arch.warp_reduction(s,operator.add)/self.n
      for c in cutlass.range_constexpr(self.n//32):
       values[c]=values[c]-mu;s2=s2+values[c]*values[c]
      var=cute.arch.warp_reduction(s2,operator.add)/self.n
      rs=cute.math.rsqrt(var+eps,fastmath=True)
      if lane==0:mean[row]=mu;rstd[row]=rs
      for c in cutlass.range_constexpr(self.n//32):out[row,lane+c*32]=(values[c]*rs*gamma[c]+b[lane+c*32].to(Float32)).to(BFloat16)
   cute.arch.barrier()
   if warp==0:
    if tile+self.stages*self.grid<tiles:
     with cute.arch.elect_one():cute.arch.mbarrier_arrive_and_expect_tx(bar+stage,self.bm*self.n*(4 if self.bwd else 2))
     loadx(src_idx=tile+self.stages*self.grid,dst_idx=stage,tma_bar_ptr=bar+stage)
     if cutlass.const_expr(self.bwd):loady(src_idx=tile+self.stages*self.grid,dst_idx=stage,tma_bar_ptr=bar+stage)
   if cutlass.const_expr(self.bwd):
    cute.arch.fence_view_async_shared();cute.arch.barrier()
    if warp==0:
     go=cute.local_tile(to,(self.bm,self.n),(tile,0))
     ss,gg=cpasync.tma_partition(ao,0,cute.make_layout(1),cute.group_modes(so,0,cute.rank(so)),cute.group_modes(go,0,cute.rank(go)))
     cute.copy(ao,ss,gg)
     with cute.arch.elect_one():cute.arch.cp_async_bulk_commit_group()
   it=it+1
  if cutlass.const_expr(self.bwd):
   for c in cutlass.range_constexpr(self.n//32):
    partial[0,warp,lane+c*32]=accumw[c];partial[1,warp,lane+c*32]=accumb[c]
   cute.arch.barrier()
   for c in cutlass.range_constexpr(cute.ceil_div(self.n,self.warps*32)):
    col=tid+c*self.warps*32
    if col<self.n:
     sw=Float32(0);sb=Float32(0)
     for wi in cutlass.range_constexpr(self.warps):sw=sw+partial[0,wi,col];sb=sb+partial[1,wi,col]
     dw[pid,col]=sw;db[pid,col]=sb
   if warp==0:
    with cute.arch.elect_one():cute.arch.cp_async_bulk_wait_group(0,read=True)
   cute.arch.barrier()

_cache={}
def prepare(x,dy,w,b,mean,rstd,out,dw,db,config,bwd):
 m,n=x.shape;bm=config['BLOCK_M1'];warps=config['num_warps'];stages=config['num_stages'];grid=dw.shape[0]
 if n%32 or bm%warps or bm not in(16,32,64,128) or config['BLOCK_K']!=n:raise ValueError('prototype needs BK=N feature tile, N%32=0 and BM divisible by warps')
 if x.stride(0)!=1 or m%8:raise ValueError('prototype TMA input requires m-major and M%8=0')
 shared=1024 + bm*n*stages*2*(2 if bwd else 1) + bm*n*2 + (2*warps*n*4 if bwd else 1028)
 if shared>torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin:raise ValueError(f'configuration requires {shared} shared-memory bytes')
 tensors=(x,dy,w,b,mean,rstd,out,dw,db)
 args=[from_dlpack(t.detach(),assumed_align=16) for t in tensors]+[Float32(1e-5),cuda.CUstream(torch.cuda.current_stream().cuda_stream)]
 key=(tuple((t.shape,t.stride(),t.dtype) for t in tensors),tuple(sorted(config.items())),bwd)
 if key not in _cache:_cache[key]=cute.compile(NormTma(n,bm,warps,stages,grid,bwd),*args)
 compiled=_cache[key]
 # Use the explicit runtime tensor and stream arguments captured for this invocation.
 return lambda:compiled(*args[:-1],cuda.CUstream(torch.cuda.current_stream().cuda_stream))
