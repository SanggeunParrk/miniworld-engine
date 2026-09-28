"""CUDA 12.8 register-pressure scheduling on unchanged hot kernels."""
import copy
import re
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_saved_front_overlap import SavedOverlapFront
from saved_input_stats import SavedInputFront

def build(plan,component,level):
    assert level in (0,2,5,8,10)
    if component=='front':
        compile_text=T.compile_text
        def with_level(source,flags):
            return compile_text(source,[*flags,f'-Xptxas=--register-usage-level={level}'])
        T.compile_text=with_level
        try:
            if plan.p.D==256:
                base=copy.copy(plan.f.front)
                fields=base.params.fields.copy();fields[7]=None
                base.params=T._launch_module().Struct(fields)
                op=SavedInputFront(base,plan.p.input_stats)
            else:op=SavedOverlapFront(plan.baseline_front,plan.pre)
        finally:T.compile_text=compile_text
        kernel=op.k
    elif component=='contract':
        assert plan.p.D in (384,512)
        op=copy.copy(plan.contract_gp)
        name=re.search(r'void (mw_\w+)\(__grid_constant__ const Params p\)',op.source_text).group(1)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={plan.p.D}',f'-Xptxas=--register-usage-level={level}']
        op.cubin=T.compile_text(op.source_text,flags)
        op.k=T.load_unit(str(op.cubin),name).kernel(name);op.k.set_max_dynamic_smem(op.smem)
        kernel=op.k
    else:raise ValueError(component)
    drv=kernel.unit.drv;fn=drv.d.CUfunction(int(kernel.handle))
    query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
    op.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');op.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    return op
