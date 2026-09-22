"""Rebuild the unchanged upstream T16 source with locally available CUDA headers.

The original CUDA-13 object cannot be loaded by this node's CUDA-12.9 driver.
Record a new build; never retain exact/audit claims tied to the original object.
"""
from pathlib import Path
import hashlib,json,os,shutil,sys

root=Path(os.environ['MINIWORLD_ANTHROPIC_ROOT'])/'common/opt_core/opt_core/kernels/transition/esm'
sys.path.insert(0,str(root))
import ef2_t16_nvjit as J
import ef2_t16_transition as T
J._STATE['incs']=['/home/psk6950/ext/cutlass/include','/usr/local/cuda-12.9/include/cccl','/usr/local/cuda-12.9/include']
J._STATE['header_versions']={'cutlass_path':'/home/psk6950/ext/cutlass','CUDA_headers':'12.9','note':'local rebuild; not upstream certified object'}
dst=root/'ef2_t16/sm_90a'
old=dst/'manifest.json'
backup=Path(__file__).parent/'t16-original-manifest.json'
if not backup.exists():shutil.copy2(old,backup)
man=json.loads(backup.read_text())
pairs=J.build_all(str(Path(__file__).parent/'t16_build'))
entries={}
for spec,cub in pairs:
    stem=J.stem_of(spec['name']);fn=stem+'.cubin';st=cub.ptxas
    shutil.copy2(cub.path,dst/fn)
    prev=man['cubins'][stem]
    entries[stem]=dict(file=fn,cubin_sha256=cub.sha256,bytes=len(cub.data),source_key=cub.skey,
        source_sha256=hashlib.sha256(spec['src'].encode()).hexdigest(),source_file=spec['name'],module=spec['module'],
        tag=spec.get('tag'),entry=prev['entry'],options=list(cub.options),macros={},nvrtc='.'.join(map(str,cub.nvrtc)),arch='sm_90a',
        compile_secs=cub.secs,smem=T.SMEM_BYTES,regs_ptxas=st['regs'],regs=st['regs'],spill_bytes=0,
        stack_bytes=st['stack_bytes'],spill_stores=st['spill_stores'],spill_loads=st['spill_loads'],
        c7512=bool(st.get('c7512')),diagnostics=st.get('diagnostics',[]),spill_free_required=True,
        ptxas_twin_sha256=st.get('twin_sha256'),ptxas_source=st.get('twin_note'),
        local_rebuild=True,loadcheck=None)
new={'arch':'sm_90a','cubins':entries,'build':{'nvrtc':list(J.nvrtc_version()),'headers':J.header_versions(),'source_revision':'f4f62fa6592ae4938d49b1757bea0cfeff9f468e'}}
old.write_text(json.dumps(new,indent=2)+'\n')
(dst/'SHA256SUMS').write_text(''.join(f"{e['cubin_sha256']}  {e['file']}\n" for e in entries.values()))
print(json.dumps(new,indent=2))
