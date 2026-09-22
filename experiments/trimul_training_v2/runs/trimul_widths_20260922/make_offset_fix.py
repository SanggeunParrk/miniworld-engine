from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;s=R.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/triton/backward_fused.py';src=s.read_text()
start=src.index('def prune_dual(');end=src.index('\ndef _input_dual_bwd_fake',start)
body=src[start:end];fixed=body.replace('rk=tl.arange(0,BLOCK_K);ag=', '# Promote before stride multiplication: KP*M may exceed 2^31.\n    rk=tl.arange(0,BLOCK_K).to(tl.int64);ag=');assert fixed!=body
imports='import triton\nimport triton.language as tl\nfrom miniworld_engine.kernels._tiles import tile_order\nfrom miniworld_engine.autotune.configs import configs_for\n'
(R/'dual_original.py').write_text(imports+body);(R/'dual_fixed.py').write_text(imports+fixed)
(R/'offset_fix.json').write_text(json.dumps(dict(source=str(s),source_sha256=hashlib.sha256(src.encode()).hexdigest(),change='cast rk to int64 before k*fs1',example=dict(M=589824,KP=4096,max_column_offset=4095*589824,int32_max=2**31-1)),indent=2))
