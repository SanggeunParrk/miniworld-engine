import hashlib,json,os,pathlib
import torch
from torch.utils.cpp_extension import load
ROOT=pathlib.Path(__file__).resolve().parent
def extension():
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
    build=ROOT/'build';build.mkdir(exist_ok=True)
    module=load(name='triattn_d128_compact',sources=[str(ROOT/'bias_compact.cu')],
        build_directory=str(build),extra_include_paths=[str(ROOT),os.environ['CUTLASS_PATH']+'/include'],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--objdir-as-tempdir','-std=c++17',
        '--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v'],verbose=True)
    return module
if __name__=='__main__':
    ext=extension()
    manifest=dict(torch=torch.__version__,binary=ext.__file__,smem=ext.smem(),
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'bias_compact.cu',ROOT/'fa3_utils.h',pathlib.Path(ext.__file__)]})
    (ROOT/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
