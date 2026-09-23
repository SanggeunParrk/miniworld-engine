"""Use source-identical existing binaries for untouched kernels; build the two edited ones."""
import os,sys,importlib.util,hashlib,runpy
from pathlib import Path
R=Path(__file__).resolve().parent;repo=R.parents[1]
from miniworld_engine.integrations import opm_train,pwa_train
import miniworld_engine
src=Path(miniworld_engine.__file__).parent/'integrations/csrc'
oldsrc=Path('/home/psk6950/miniworld-engine-msa/src/miniworld_engine/integrations/csrc')
cache=Path('/home/psk6950/MiniWorld/runs/msa_bench_20260921/cuda/engine_jit')
for name,filename in [('miniworld_pwa_fwd2','pwa_fwd2.cu'),('miniworld_pwa_ctr','pwa_ctr.cu'),('miniworld_pwa_dgv_bwd','dgv_bwd.cu'),('miniworld_pwa_ln_vg','ln_vg.cu'),('miniworld_pwa_pair3','pair3.cu'),('miniworld_pwa_fwd3','pwa_fwd3.cu')]:
    assert (src/filename).read_bytes()==(oldsrc/filename).read_bytes(),filename
    spec=importlib.util.spec_from_file_location(name,str(cache/name/(name+'.so')))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    pwa_train._EXT[name]=mod
if os.environ.get('FUSION_BASELINE') == '1':
    for name,filename in [('miniworld_pwa_glue3','pwa_glue3.cu'),('miniworld_opm_epilogue','opm_epilogue.cu')]:
        assert (src/filename).read_bytes()==(oldsrc/filename).read_bytes(),filename
        directory='opm_epilogue' if name=='miniworld_opm_epilogue' else name
        spec=importlib.util.spec_from_file_location(name,str(cache/directory/(name+'.so')))
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        if name=='miniworld_opm_epilogue':opm_train._EXT['mod']=mod
        else:pwa_train._EXT[name]=mod
if sys.argv[1]=='test':
    # Independent compiler builds; the GPU tests themselves run serially.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(opm_train._ext),pool.submit(pwa_train._build,'miniworld_pwa_glue3','pwa_glue3.cu')]
        for future in futures:future.result()
    import pytest
    raise SystemExit(pytest.main(['-q','tests/integrations/test_opm_train_gpu.py','tests/integrations/test_pwa_train_gpu.py']))
else:
    sys.argv=sys.argv[1:]
    runpy.run_path(sys.argv[0],run_name='__main__')
