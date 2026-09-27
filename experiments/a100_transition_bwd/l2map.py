"""SM -> L2 partition map of this card: every SM times L2 hits on 64 lines spread over a 4 MB buffer (pointer chase, lines warm in L2);
SMs whose latency pattern over the lines match are behind the same partition.  python l2map.py [out.json]"""
import json
import sys

import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
__global__ void k(const unsigned* buf, int nlines, int stride_words, int reps, unsigned long long* lat, int* smid_out) {
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x != 0) return;
  smid_out[blockIdx.x] = smid;
  unsigned sink = 0;
  for (int l = 0; l < nlines; ++l) {
    const unsigned* a = buf + (size_t)l * stride_words;
    unsigned v = 0;
    for (int w = 0; w < 4; ++w) { asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a + v)); sink += v; }  // warm
    long long t0 = clock64();
    for (int r = 0; r < reps; ++r) { asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a + v)); sink += v; }
    long long t1 = clock64();
    lat[(size_t)blockIdx.x * nlines + l] = (t1 - t0) / reps;
  }
  if (sink == 0xdeadbeef) smid_out[0] = -1;
}
void run(torch::Tensor buf, int nlines, int stride_words, int reps, torch::Tensor lat, torch::Tensor smid) {
  k<<<smid.size(0), 32>>>((const unsigned*)buf.data_ptr(), nlines, stride_words, reps, (unsigned long long*)lat.data_ptr(), smid.data_ptr<int>());
}
"""
m = load_inline("l2map", cpp_sources="void run(torch::Tensor, int, int, int, torch::Tensor, torch::Tensor);", cuda_sources=src,
                functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
nsm = torch.cuda.get_device_properties(0).multi_processor_count
nl, stride = 64, 16384                                         # 64 lines, 64 KB apart
buf = torch.zeros(nl * stride, dtype=torch.int32, device="cuda")
nb = nsm * 4                                                    # several blocks per SM so every SM appears
lat = torch.zeros(nb, nl, dtype=torch.int64, device="cuda")
smid = torch.zeros(nb, dtype=torch.int32, device="cuda")
m.run(buf, nl, stride, 64, lat, smid)
torch.cuda.synchronize()
m.run(buf, nl, stride, 64, lat, smid)
torch.cuda.synchronize()
lat, smid = lat.cpu().double(), smid.cpu()
per = {}
for b in range(nb):
    per.setdefault(int(smid[b]), []).append(lat[b])
sms = sorted(per)
M = torch.stack([torch.stack(per[s]).mean(0) for s in sms])     # [sm][line]
# 2-means on the per-SM latency vectors (lines near for one partition are far for the other)
Z = M - M.mean(0, keepdim=True)
u, sv, vt = torch.linalg.svd(Z, full_matrices=False)
proj = Z @ vt[0]
part = (proj > 0).int()
for _ in range(20):
    c0, c1 = M[part == 0].mean(0), M[part == 1].mean(0)
    part = ((M - c1).norm(dim=1) < (M - c0).norm(dim=1)).int()
c0, c1 = M[part == 0].mean(0), M[part == 1].mean(0)
near0 = c0 < c1                                                 # lines homed in partition 0
print("partition sizes:", int((part == 0).sum()), int((part == 1).sum()), " lines homed 0/1:", int(near0.sum()), int((~near0).sum()))
print("near / far latency:", float(torch.minimum(c0, c1).mean()), float(torch.maximum(c0, c1).mean()))
d = (M - torch.where(part[:, None].bool(), c1[None], c0[None])).abs().mean(1)
print("per-SM misfit (max):", float(d.max()), " separation:", float((c0 - c1).abs().mean()))
mp = {int(s): int(p) for s, p in zip(sms, part)}
print("smid -> partition:", "".join(str(mp[s]) for s in sms))
if len(sys.argv) > 1:
    json.dump(mp, open(sys.argv[1], "w"))
