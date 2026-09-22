import json,torch
from pathlib import Path
from miniworld_engine.integrations import anthropic as A
T=A.provider('transition');E=T._flash_ext()
bits=torch.arange(65536,dtype=torch.int32,device='cuda').to(torch.int16).view(torch.bfloat16)
want=torch.nn.functional.silu(bits);got=torch.empty_like(want);E.silu_table(got)
nan2=want.isnan() & got.isnan();bad=int(((want.view(torch.int16)!=got.view(torch.int16)) & ~nan2).sum())
r=dict(loaded_from=E.__file__,abi_key=T.flash_abi_key(),silu_mismatch=bad,smem_bytes=int(E.smem_bytes()),hidden_chunk=int(E.hidden_chunk()))
print(r);Path(__file__).with_suffix('.json').write_text(json.dumps(r,indent=2));assert bad==0
