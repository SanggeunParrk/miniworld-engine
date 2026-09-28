"""One TMA transfer per full input-LN operand, preserving its shared layout."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class WholeInput:
    def __init__(self,plan,l2='128B'):
        old=plan.dx.reduce_only;self.__dict__.update(old.__dict__)
        p=plan.p;d=p.D;rows=16
        body=old.source_text
        marker='extern "C" __global__ __launch_bounds__'
        helper='''TMN_DEVI void whole_load(void* dst,const CUtensorMap* map,uint64_t* bar,int tile){
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0,0,%3}],[%2];"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(tile):"memory");
}
'''
        body=body.replace(marker,helper+marker)
        start=body.index('   for(int c=0;c<D;c+=64){')
        end=body.index('\n  mbar_wait',start)
        body=body[:start]+'''   whole_load(sm,&p.x,bar,row/ROWS);
   whole_load(sm+SB,&p.dn,bar,row/ROWS);
   whole_load(sm+2*SB,&p.res,bar,row/ROWS);
  }
'''+body[end:]
        start=body.index('  if(tid==0){for(int c=0;c<D;c+=64){')
        end=body.index('\n  __syncthreads();',start)
        body=body[:start]+'''  if(tid==0){
   asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0,{0,0,0,%2}],[%1];"::"l"(&p.dx),"r"(smem_u32(sm)),"r"(row/ROWS):"memory");
   tma_store_commit();tma_store_wait_all();
  }
'''+body[end:]
        body=body.replace('mw_independent_input_ln','mw_whole_input_ln')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}','-DINPUT_ROWS=16','-DINPUT_THREADS=128','-DINPUT_MINBLOCKS=4']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_whole_input_ln').kernel('mw_whole_input_ln')
        self.kernel.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[64,rows,d//64,1],dims=[64,rows,d//64,p.M//rows],strides_bytes=[d*2,128,rows*d*2],swizzle='128B',l2=l2)
        self.params=L.Struct([*[tm(t) for t in (p.x,p.tensors[10],p.dy,p.dx)],*old.params.fields[4:]])
    def __call__(self):
        self.kernel.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
