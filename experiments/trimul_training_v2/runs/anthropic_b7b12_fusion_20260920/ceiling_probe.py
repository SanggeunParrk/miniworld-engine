from front_core import *
with torch.no_grad():
 a=torch.randn(128*1024*1024,device='cuda',dtype=torch.bfloat16);b=torch.empty_like(a)
 tcopy=paired({'copy':capture(lambda:b.copy_(a))})['copy'];copybytes=2*a.numel()*a.element_size()
 del a,b
 n=8192;a=torch.randn((n,n),device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a);c=torch.empty_like(a)
 tmm=paired({'mm':capture(lambda:torch.mm(a,b,out=c))})['mm'];flops=2*n*n*n
 r=dict(device=torch.cuda.get_device_name(),copy=dict(bytes_per_call=copybytes,time=tcopy,TB_s=copybytes/tcopy['median_us']/1e6),gemm=dict(N=n,flops=flops,time=tmm,TFLOP_s=flops/tmm['median_us']/1e6),note='Empirical copy/GEMM diagnostics; neither measurement alone proves a roofline for fused B7-B12')
 (R/'ceiling-probe.json').write_text(json.dumps(r,indent=2));print('CEILING',r['copy']['TB_s'],r['gemm']['TFLOP_s'],flush=True)
