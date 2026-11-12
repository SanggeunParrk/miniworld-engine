#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "dq_rows_candidate.cuh"
template <class G>
void launch_attn_bwd_dq(const triattn_dq80::DqParams& p, int z) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(triattn_dq80::attn_bwd_dq_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / G::BM, p.L / G::R, z * p.H);
  triattn_dq80::attn_bwd_dq_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_bwd_dq(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias, torch::Tensor lse, torch::Tensor delta,
                 torch::Tensor dq, torch::Tensor dbp, int64_t L, int64_t H, int64_t Z, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo, int64_t lddq,
                 double scl, double sm_scale, int64_t rows, int64_t nkv, int64_t bn, int64_t minb,int64_t bm,bool cached) {
  for (auto* t : {&q, &k, &v, &dov, &bias, &dq, &dbp}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_bwd_dq: bf16 CUDA tensors");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == Z * H * L * L, "attn_bwd_dq: bias [z, H, L, L] contiguous");
  for (auto* t : {&lse, &delta})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == Z * H * L * L, "attn_bwd_dq: lse / delta [z, H, L, L] fp32");
  TORCH_CHECK(L % a100::TA_BM == 0 && L >= a100::TA_BM && L % bn == 0 && L % rows == 0, "attn_bwd_dq: L must be a multiple of 128");
  const int64_t T = Z * L * L;
  for (auto [t, ld] : std::initializer_list<std::pair<const torch::Tensor*, int64_t>>{{&q, ldq}, {&k, ldk}, {&v, ldv}, {&dov, lddo}, {&dq, lddq}}) {
    TORCH_CHECK(ld >= H * 32 && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_bwd_dq: row strides / alignment");
    TORCH_CHECK((t->storage_offset() + (T - 1) * ld + H * 32) * t->element_size() <= (int64_t)t->storage().nbytes(), "attn_bwd_dq: tensor extents");
  }
  TORCH_CHECK(dbp.is_contiguous() && dbp.numel() == (L / rows) * Z * H * L * L, "attn_bwd_dq: dbp [L / rows, z H, L, L] contiguous");
  TORCH_CHECK(Z * H <= 65535 && L / rows <= 65535, "attn_bwd_dq: grid");
  triattn_dq80::DqParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.dov = reinterpret_cast<const __nv_bfloat16*>(dov.data_ptr());
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
  p.dq = reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()); p.dbp = reinterpret_cast<__nv_bfloat16*>(dbp.data_ptr());
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddq = lddq; p.L = (int)L; p.H = (int)H; p.ZH = (int)(Z * H);
  p.scl = (float)scl; p.sm_scale = (float)sm_scale;

#define DQ_CFG(R,NKV,BN,MINB,BM,CACHE) if(rows==R&&nkv==NKV&&bn==BN&&minb==MINB&&bm==BM&&cached==CACHE){launch_attn_bwd_dq<triattn_dq80::DqCfg<R,NKV,BN,MINB,BM,CACHE>>(p,(int)Z);return;}
 DQ_CFG(4,3,32,1,128,true)
 DQ_CFG(8,3,32,1,128,true)
 DQ_CFG(8,2,32,1,64,true)
 DQ_CFG(4,2,32,1,64,true)
 DQ_CFG(8,3,32,1,64,true)
 DQ_CFG(8,2,32,1,64,false)
 DQ_CFG(8,3,32,1,128,false)
#undef DQ_CFG
 TORCH_CHECK(false,"own DQ candidate geometry");
}
void reduce(torch::Tensor dbp,torch::Tensor db,int64_t groups){
 TORCH_CHECK(dbp.is_cuda()&&db.is_cuda()&&dbp.is_contiguous()&&db.is_contiguous()&&dbp.scalar_type()==at::kBFloat16&&db.scalar_type()==at::kBFloat16&&dbp.numel()==groups*db.numel()&&db.numel()%8==0,"db reduce shape");
 const long long plane8=db.numel()/8;triattn_dq80::db_reduce_kernel<<<(plane8+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)dbp.data_ptr(),(__nv_bfloat16*)db.data_ptr(),groups,plane8);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&attn_bwd_dq);m.def("reduce",&reduce);}
