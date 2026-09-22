from pathlib import Path
import json
R=Path(__file__).resolve().parent
from miniworld_engine import settings
from miniworld_engine.autotune import trimul_sm90_config as sm90_cfg
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
manifest=R.parent/'trimul_sm90_parity_20260917/engine/docs/records/normalization-h100-20260917/measured-configs-L384.json'
configs=json.loads(manifest.read_text());configs['layernorm_bwd_split_sm90_cute']=dict(BLOCK_M1=32,BLOCK_K=256,num_warps=8,num_stages=2);original_resolve=sm90_cfg.resolve;calls={}
def resolve(op,tensors,**kw):
 if op not in configs:return original_resolve(op,tensors,**kw)
 cfg=dict(configs[op]);assert kw['feasibility'](cfg) is None
 calls[op]=calls.get(op,0)+1;return cfg
sm90_cfg.resolve=resolve
import triton
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _input_dual_bwd_kernel
tri_configs=json.loads((manifest.parent/'triton-configs-L384.json').read_text())
for name,kernel in [('front',B._bidir_front_kernel),('f567',_output_f567_kernel),('dual_bwd',_input_dual_bwd_kernel)]:
 cfg=dict(tri_configs[name]);nw=cfg.pop('num_warps');ns=cfg.pop('num_stages');kernel.configs=[triton.Config(cfg,num_warps=nw,num_stages=ns)];kernel.cache.clear();kernel.early_config_prune=None
