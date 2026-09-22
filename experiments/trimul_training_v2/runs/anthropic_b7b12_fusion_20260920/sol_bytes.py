import json
from pathlib import Path
rows=[]
for n in (384,768):
 m=n*n;traffic={'pre_read':m*1024*2,'left_right_grad_read':m*512*2,'xn_read':m*128*2,'gate_grad_read':m*128*2,'x_read':m*128*2,'residual_read':m*128*2,'dx_write':m*128*2,'pair_mask_read':m*2,'mean_rstd_read':m*8,'weight_read':4*128*256*2+128*128*2,'gamma_read':128*4,'weight_and_ln_grad_write':4*128*256*2+256*4}
 b=sum(traffic.values());flops=2*m*128*1024*2+2*m*128*128
 rows.append(dict(L=n,minimum_named_tensor_traffic_bytes=b,traffic=traffic,tensor_flops=flops,copy_ceiling_lower_bound_us=b/2.962e6,gemm_ceiling_lower_bound_us=flops/688e6,scope='optimistic compulsory tensor bytes, excluding parameter partials and cache misses; diagnostic ceilings, not proof of achievable fused-kernel SoL'))
Path(__file__).with_name('compulsory-traffic.json').write_text(json.dumps(rows,indent=2))
