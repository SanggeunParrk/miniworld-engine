"""Measure a declared candidate subset; preserve honest partial coverage."""
import json
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import native,capture,trimul_sm90_config,cache
from miniworld_engine.kernels.layernorm.cute.tma_backward import OP,backward_impl
root=Path(__file__).parent
m=768**2;n=256
x=torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t();dy=torch.randn_like(x);w=torch.randn(n,device='cuda');mean=x.float().mean(1);rs=(x.float().var(1,unbiased=False)+1e-5).rsqrt()
# Candidate subset is selected by the preceding correctness/timing sweep, not
# installed into the production resolver. The cache records just these trials.
configs=[r['config'] for r in json.loads((root/'bench-L768.json').read_text())]
original=trimul_sm90_config.partition_configs
trimul_sm90_config.partition_configs=lambda op,feasibility,directory=None: ([c for c in configs if feasibility(c) is None],[])
settings.configure(run_autotune=True,bench_rep_ms=30,bench_clear_mb=64)
capture.reset();capture.set_incremental(True);capture.set_round_cache(str(root/'native-rounds'))
backward_impl(dy,x,w,mean,rs,x.stride());torch.cuda.synchronize()
shard=root/'b4-hot-native.json';capture.dump_shard(str(shard))
trimul_sm90_config.partition_configs=original
merged=capture.merge_shards([shard],gpu=cache.gpu_key())
settings.configure(run_autotune=False)
limit=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
bucket=native.tensor_key(x,dy,w,mean,rs,extra=(limit,))
configs_all=native.candidates_for(OP,bucket)
chosen=native.choose_config(OP,configs_all,dtype=str(x.dtype),bucket=bucket,device_index=0)
data=cache._load(OP,cache.gpu_key())
report={'measured_candidates':len(configs),'declared_candidates':len(trimul_sm90_config.declared_configs(OP)), 'feasible_candidates':len(configs_all),'chosen':chosen,'merged':merged,'remaining':native.pending_candidates(OP,data),'source_identity':native.source_identity(),'full_coverage':False}
(root/'b4-hot-cache-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
# The next call must select through the unmodified resolver/cache.
backward_impl(dy,x,w,mean,rs,x.stride());torch.cuda.synchronize()
