"""Experimental single-parameter partial reduction; dw/db still use two launches."""
import torch,cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.nvgpu import cpasync,warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90h
from cuda.bindings import driver as cuda
class ReductionTiled:
 def __init__(self,g,n,bn,nw):self.g,self.n,self.bn,self.nw=g,n,bn,nw;self.bg=1<<(g-1).bit_length()
 @cute.jit
 def __call__(self,x,y,stream:cuda.CUstream):
  if cutlass.const_expr(self.bn<8):
   layout=cute.make_composed_layout(cute.make_swizzle(0,4,3),0,cute.make_layout((self.bg,self.bn),stride=(self.bn,1)))
  else:
   atom=warpgroup.make_smem_layout_atom(sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR,Float32,self.bn),Float32)
   layout=cute.tile_to_shape(atom,(self.bg,self.bn),order=(0,1))
  ax,tx=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),x,layout,(self.bg,self.bn))
  st=cute.struct.Align[cute.struct.MemRange[Float32,cute.cosize(layout)],1024]
  @cute.struct
  class Storage:
   barrier:cute.struct.MemRange[cutlass.Int64,1]
   data:st
   partial:cute.struct.MemRange[Float32,self.nw*self.bn]
  self.storage=Storage
  self.kernel(ax,tx,y,layout).launch(grid=[cute.ceil_div(self.n,self.bn),1,1],block=[self.nw*32,1,1],stream=stream)
 @cute.kernel
 def kernel(self,ax,tx,y,layout):
  tid,_,_=cute.arch.thread_idx();pid,_,_=cute.arch.block_idx();warp=cute.arch.make_warp_uniform(cute.arch.warp_idx());lane=tid%32
  store=cutlass.utils.SmemAllocator().allocate(self.storage)
  shared=store.data.get_tensor(layout.outer,swizzle=layout.inner);bar=store.barrier.data_ptr()
  part=store.partial.get_tensor(cute.make_layout((self.nw,self.bn),stride=(self.bn,1)))
  if warp==0:
   with cute.arch.elect_one():cute.arch.mbarrier_init(bar,1)
  cute.arch.mbarrier_init_fence();cute.arch.barrier()
  if warp==0:
   globaltile=cute.local_tile(tx,(self.bg,self.bn),(0,pid))
   ss,gg=cpasync.tma_partition(ax,0,cute.make_layout(1),cute.group_modes(shared,0,2),cute.group_modes(globaltile,0,2))
   with cute.arch.elect_one():cute.arch.mbarrier_arrive_and_expect_tx(bar,self.bg*self.bn*4)
   cute.copy(ax,gg,ss,tma_bar_ptr=bar)
  cute.arch.mbarrier_wait(bar,0)
  col=lane%self.bn; rowgroup=warp*(32//self.bn)+lane//self.bn
  acc=Float32(0)
  for r in cutlass.range(rowgroup,self.g,self.nw*(32//self.bn)):acc=acc+shared[r,col]
  for i in cutlass.range_constexpr((32//self.bn).bit_length()-1):
   acc=acc+cute.arch.shuffle_sync_bfly(acc,offset=self.bn*(1<<i))
  if lane<self.bn:part[warp,col]=acc
  cute.arch.barrier()
  if tid<self.bn and pid*self.bn+tid<self.n:
   acc=Float32(0)
   for r in cutlass.range_constexpr(self.nw):acc=acc+part[r,tid]
   y[pid*self.bn+tid]=acc

cache={}
def prepare(x,y,bn=4,nw=4):
 args=[from_dlpack(t.detach(),assumed_align=16) for t in (x,y)]
 key=(tuple(x.shape),bn,nw)
 stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
 if key not in cache:cache[key]=cute.compile(ReductionTiled(*x.shape,bn,nw),*args,stream)
 f=cache[key];return lambda:f(*args,cuda.CUstream(torch.cuda.current_stream().cuda_stream))
