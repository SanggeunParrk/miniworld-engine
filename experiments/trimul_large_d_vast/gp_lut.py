"""Exact BF16 sigmoid lookup in otherwise unchanged contraction/GP kernels."""
import re
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class LookupGP:
    def __init__(self,plan,kind='shared'):
        old=plan.contract_gp;self.__dict__.update(old.__dict__)
        assert plan.p.n==384 and kind in ('shared','global')
        offset=(self.smem+127)//128*128
        lo,hi=0x3880,0x4100;span=hi-lo;count=2*span
        self.lut=torch.empty(count,device=plan.p.x.device,dtype=torch.float32)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={plan.p.D}']
        helper=f'''
TMN_DEVI float exact_lookup(float x,const float* table){{
 unsigned raw=__float_as_uint(x)>>16,mag=raw&32767;
 if(mag>={lo} && mag<{hi})return table[(raw>>15)*{span}+mag-{lo}];
 return math::sigmoid(x);
}}
'''
        init='''#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
'''+helper+f'''
extern "C" __global__ void init_lookup(float* out){{
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<{count}){{unsigned raw=(i/{span})*32768+{lo}+i%{span};out[i]=math::sigmoid(__uint_as_float(raw<<16));}}
}}
extern "C" __global__ void verify_lookup(const float* table,float* actual,float* ref){{
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<65536){{float x=__uint_as_float(unsigned(i)<<16);actual[i]=exact_lookup(x,table);ref[i]=math::sigmoid(x);}}
}}
'''
        self.init_cubin=T.compile_text(init,flags)
        unit=T.load_unit(str(self.init_cubin),'lookup_init')
        unit.kernel('init_lookup').launch(((count+255)//256,1,1),(256,1,1),[self.lut],0)
        actual=torch.empty(65536,device=self.lut.device);ref=torch.empty_like(actual)
        unit.kernel('verify_lookup').launch((256,1,1),(256,1,1),[self.lut,actual,ref],0)
        self.all_patterns_exact=bool(torch.equal(actual.view(torch.int32),ref.view(torch.int32)))
        assert self.all_patterns_exact
        body=old.source_text
        marker='int N;};';assert body.count(marker)==1
        body=body.replace(marker,'int N;const float* lut;};')
        at=body.index('template<int MODE> TMN_DEVI void load_input(')
        body=body[:at]+helper+body[at:]
        lut_expr=f'reinterpret_cast<float*>(sm+{offset})' if kind=='shared' else 'p.lut'
        for word in ('bf16lo(gr)','bf16hi(gr)'):
            old_expr=f'math::sigmoid({word})';assert body.count(old_expr)==1
            body=body.replace(old_expr,f'exact_lookup({word},{lut_expr})')
        if kind=='shared':
            # All CTA threads participate before the existing initial join.
            marker=' if(threadIdx.x==0){for(int i=0;i<7;++i)mbar_init'
            assert body.count(marker)==1
            body=body.replace(marker,f' for(int q=threadIdx.x;q<{count};q+=blockDim.x)reinterpret_cast<float*>(sm+{offset})[q]=p.lut[q];\n'+marker)
            self.smem=offset+count*4
        name=re.search(r'void (mw_\w+)\(__grid_constant__ const Params p\)',body).group(1)
        body=body.replace(name,'mw_lookup_contract_gp')
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_lookup_contract_gp').kernel('mw_lookup_contract_gp')
        self.k.set_max_dynamic_smem(self.smem)
        self.params=T._launch_module().Struct([*old.params.fields,self.lut])
        self.threads=288 if plan.p.D==512 else 256
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):
        self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(self.threads,1,1),[self.params],self.smem)
