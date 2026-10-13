#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "atom_core_qresident_dkv.cuh"
namespace {
void check_bf16(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, name, ": a bf16 CUDA tensor");
}

// a token-major operand [T][ld] viewed over `t`: row stride ld (>= H hd), 16-byte aligned rows, and the whole [T][H hd] window inside the storage
void check_tokens(const torch::Tensor& t, const char* name, int64_t T, int64_t ld, int64_t cols) {
  TORCH_CHECK(ld >= cols && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": row stride / alignment");
  TORCH_CHECK((t.storage_offset() + (T - 1) * ld + cols) * t.element_size() <= (int64_t)t.storage().nbytes(), name, ": tensor extents");
}

void check_f32(const torch::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous() && t.numel() == n, name, ": a contiguous fp32 CUDA tensor of ", n, " elements");
}

// a gradient buffer [T][cols]: fp32 or bf16, contiguous
void check_grad(const torch::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(t.is_cuda() && (t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16) && t.is_contiguous() && t.numel() == n, name,
              ": a contiguous fp32 or bf16 CUDA tensor of ", n, " elements");
}

// a gradient operand [T][ld] (a view of a wider buffer is fine: the gradient of q / k / v written straight into the columns of the projection's gradient):
// fp32 or bf16, unit column stride, row stride ld >= cols (a multiple of 2 elements), the whole [T][cols] window inside the storage
void check_grad_view(const torch::Tensor& t, const char* name, int64_t T, int64_t ld, int64_t cols) {
  TORCH_CHECK(t.is_cuda() && (t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16) && t.dim() == 2 && t.stride(1) == 1 && t.stride(0) == ld && ld >= cols &&
              ld % 2 == 0, name, ": a fp32 / bf16 CUDA view [T][ld]");
  TORCH_CHECK((t.storage_offset() + (T - 1) * ld + cols) * t.element_size() <= (int64_t)t.storage().nbytes() &&
              reinterpret_cast<uintptr_t>(t.data_ptr()) % (2 * t.element_size()) == 0, name, ": extents / alignment");
}

template<class G> void launch_dq(const atom_core_qresident248::DqParams& p,int A) {
 static bool set[64]={}; const int dev=at::cuda::current_device()&63;
 if(!set[dev]){C10_CUDA_CHECK(cudaFuncSetAttribute(atom_core_qresident248::attn_bwd_dq_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));set[dev]=true;}
 dim3 grid(p.L/aa80::BM,A/G::R,p.H);
 atom_core_qresident248::attn_bwd_dq_kernel<G><<<grid,256,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<class G> void launch_dkv(const atom_core_qresident248::DkvParams& p,int A) {
 static bool set[64]={}; const int dev=at::cuda::current_device()&63;
 if(!set[dev]){C10_CUDA_CHECK(cudaFuncSetAttribute(atom_core_qresident248::attn_bwd_dkv_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));set[dev]=true;}
 dim3 grid(p.L/aa80::BM,A,p.H);
 atom_core_qresident248::attn_bwd_dkv_kernel<G><<<grid,256,G::SMEM,at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<class G, bool DQ> std::vector<int64_t> resource_values(){
 cudaFuncAttributes a; int blocks=0;
 if constexpr(DQ){C10_CUDA_CHECK(cudaFuncGetAttributes(&a,atom_core_qresident248::attn_bwd_dq_kernel<G>));C10_CUDA_CHECK(cudaFuncSetAttribute(atom_core_qresident248::attn_bwd_dq_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,atom_core_qresident248::attn_bwd_dq_kernel<G>,256,G::SMEM));}
 else{C10_CUDA_CHECK(cudaFuncGetAttributes(&a,atom_core_qresident248::attn_bwd_dkv_kernel<G>));C10_CUDA_CHECK(cudaFuncSetAttribute(atom_core_qresident248::attn_bwd_dkv_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,atom_core_qresident248::attn_bwd_dkv_kernel<G>,256,G::SMEM));}
 cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,at::cuda::current_device()));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,G::SMEM,256,blocks,blocks*8,prop.maxThreadsPerMultiProcessor,prop.multiProcessorCount,a.binaryVersion,a.ptxVersion};
}
} // anonymous
void attn_bwd_dq(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias, torch::Tensor lse, torch::Tensor delta,
                 torch::Tensor dq, torch::Tensor dbp, int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo,
                 int64_t lddq, double scl, double bscl, double sm_scale, int64_t rows, int64_t nkv, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps, int64_t order) {
  for (auto* t : {&q, &k, &v, &dov, &bias, &dbp}) check_bf16(*t, "attn_bwd_dq");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == H * L * L, "attn_bwd_dq: bias [H, L, L] contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A % rows == 0 && A / rows <= 65535 && H <= 65535, "attn_bwd_dq: L a multiple of 128, A a multiple of rows");
  if (ss <= 0) ss = L;
  TORCH_CHECK(ss >= L, "attn_bwd_dq: ss >= L");
  const int64_t T = (A - 1) * ss + L, cols = H * hd;
  check_tokens(q, "attn_bwd_dq q", T, ldq, cols); check_tokens(k, "attn_bwd_dq k", T, ldk, cols); check_tokens(v, "attn_bwd_dq v", T, ldv, cols);
  check_tokens(dov, "attn_bwd_dq dov", T, lddo, cols);
  check_f32(lse, "attn_bwd_dq lse", A * H * L); check_f32(delta, "attn_bwd_dq delta", A * H * L);
  check_grad_view(dq, "attn_bwd_dq dq", T, lddq, cols);
  TORCH_CHECK(dbp.is_contiguous() && dbp.numel() == (A / rows) * H * L * L, "attn_bwd_dq: dbp [A / rows, H, L, L] contiguous");
  atom_core_qresident248::DqParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.dov = reinterpret_cast<const __nv_bfloat16*>(dov.data_ptr());
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
  p.dq = dq.data_ptr(); p.dbp = reinterpret_cast<__nv_bfloat16*>(dbp.data_ptr());
  const int64_t km = kpen.numel() > 0;
  p.kpen = nullptr; p.kps = 0;
  if (km) {
    TORCH_CHECK(kpen.is_cuda() && kpen.scalar_type() == at::kFloat && kpen.is_contiguous() && kps >= L && (A - 1) * kps + L <= kpen.numel() &&
                reinterpret_cast<uintptr_t>(kpen.data_ptr()) % 16 == 0 && kps % 4 == 0, "attn_bwd_dq: kpen fp32 [A][L] (16-byte aligned rows)");
    p.kpen = kpen.data_ptr<float>(); p.kps = kps;
  }
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddq = lddq; p.ss = ss; p.L = (int)L; p.H = (int)H;
  p.scl = (float)scl; p.bscl = (float)bscl; p.sm_scale = (float)sm_scale;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t ob = dq.scalar_type() == at::kBFloat16;
  TORCH_CHECK(hd==32 && rows==4 && nkv==3 && minb==1 && ob==1, "E237 original R4/HD32/NKV3/minb1/BF gradients only");
  TORCH_CHECK(order==0 || order==1 || order==2 || order==3 || order==4 || order==6,"E237 supported order");
  TORCH_CHECK(order<=1 || (A/4)%order==0,"E237 chunk divides physical groups");
  if(km==0){
    switch(order){
    case 0: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,0,0>>(p,(int)A); return;
    case 1: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,0,1>>(p,(int)A); return;
    case 2: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,0,2>>(p,(int)A); return;
    case 3: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,0,3>>(p,(int)A); return;
    case 4: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,0,4>>(p,(int)A); return;
    case 6: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,0,6>>(p,(int)A); return;
    }
  }
  if(km==1){
    switch(order){
    case 0: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,1,0>>(p,(int)A); return;
    case 1: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,1,1>>(p,(int)A); return;
    case 2: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,1,2>>(p,(int)A); return;
    case 3: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,1,3>>(p,(int)A); return;
    case 4: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,1,4>>(p,(int)A); return;
    case 6: launch_dq<atom_core_qresident248::DqCfg<32,4,3,1,1,1,6>>(p,(int)A); return;
    }
  }
  TORCH_CHECK(false,"E237 unsupported order");
}

void attn_bwd_dkv(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias_t, torch::Tensor lse, torch::Tensor delta,
                  torch::Tensor dk, torch::Tensor dv, int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo,
                  int64_t lddk, int64_t lddv, double scl, double bscl, double sm_scale, int64_t nst, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps, int64_t order) {
  for (auto* t : {&q, &k, &v, &dov, &bias_t}) check_bf16(*t, "attn_bwd_dkv");
  TORCH_CHECK(bias_t.is_contiguous() && bias_t.numel() == H * L * L, "attn_bwd_dkv: bias_t [H, L, L] contiguous");
  TORCH_CHECK(L % aa80::BM == 0 && L >= aa80::BM && A <= 65535 && H <= 65535, "attn_bwd_dkv: L a multiple of 128");
  if (ss <= 0) ss = L;
  TORCH_CHECK(ss >= L, "attn_bwd_dkv: ss >= L");
  const int64_t T = (A - 1) * ss + L, cols = H * hd;
  check_tokens(q, "attn_bwd_dkv q", T, ldq, cols); check_tokens(k, "attn_bwd_dkv k", T, ldk, cols); check_tokens(v, "attn_bwd_dkv v", T, ldv, cols);
  check_tokens(dov, "attn_bwd_dkv dov", T, lddo, cols);
  check_f32(lse, "attn_bwd_dkv lse", A * H * L); check_f32(delta, "attn_bwd_dkv delta", A * H * L);
  check_grad_view(dk, "attn_bwd_dkv dk", T, lddk, cols); check_grad_view(dv, "attn_bwd_dkv dv", T, lddv, cols);
  TORCH_CHECK(dk.scalar_type() == dv.scalar_type(), "attn_bwd_dkv: dk and dv in one dtype");
  atom_core_qresident248::DkvParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.dov = reinterpret_cast<const __nv_bfloat16*>(dov.data_ptr());
  p.bias_t = reinterpret_cast<const __nv_bfloat16*>(bias_t.data_ptr()); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
  p.dk = dk.data_ptr(); p.dv = dv.data_ptr();
  const int64_t km = kpen.numel() > 0;
  p.kpen = nullptr; p.kps = 0;
  if (km) {
    TORCH_CHECK(kpen.is_cuda() && kpen.scalar_type() == at::kFloat && kpen.is_contiguous() && kps >= L && (A - 1) * kps + L <= kpen.numel(), "attn_bwd_dkv: kpen fp32 [A][L]");
    p.kpen = kpen.data_ptr<float>(); p.kps = kps;
  }
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddk = lddk; p.lddv = lddv; p.ss = ss; p.L = (int)L; p.H = (int)H;
  p.scl = (float)scl; p.bscl = (float)bscl; p.sm_scale = (float)sm_scale;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t ob = dk.scalar_type() == at::kBFloat16;
  TORCH_CHECK(hd==32 && nst==4 && minb==2 && ob==1, "E237 original HD32/NST4/minb2/BF gradients only");
  TORCH_CHECK(order==0 || order==1 || order==2 || order==3 || order==4 || order==6,"E237 supported order");
  TORCH_CHECK(order<=1 || (A)%order==0,"E237 chunk divides physical groups");
  if(km==0){
    switch(order){
    case 0: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,0,0>>(p,(int)A); return;
    case 1: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,0,1>>(p,(int)A); return;
    case 2: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,0,2>>(p,(int)A); return;
    case 3: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,0,3>>(p,(int)A); return;
    case 4: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,0,4>>(p,(int)A); return;
    case 6: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,0,6>>(p,(int)A); return;
    }
  }
  if(km==1){
    switch(order){
    case 0: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,1,0>>(p,(int)A); return;
    case 1: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,1,1>>(p,(int)A); return;
    case 2: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,1,2>>(p,(int)A); return;
    case 3: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,1,3>>(p,(int)A); return;
    case 4: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,1,4>>(p,(int)A); return;
    case 6: launch_dkv<atom_core_qresident248::DkvCfg<32,4,2,1,1,6>>(p,(int)A); return;
    }
  }
  TORCH_CHECK(false,"E237 unsupported order");
}
std::vector<int64_t> resources(bool dq,bool km,int64_t order){
 if(dq==true && km==false){switch(order){
 case 0:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,0,0>,true>();
 case 1:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,0,1>,true>();
 case 2:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,0,2>,true>();
 case 3:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,0,3>,true>();
 case 4:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,0,4>,true>();
 case 6:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,0,6>,true>();
 }}
 if(dq==true && km==true){switch(order){
 case 0:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,1,0>,true>();
 case 1:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,1,1>,true>();
 case 2:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,1,2>,true>();
 case 3:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,1,3>,true>();
 case 4:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,1,4>,true>();
 case 6:return resource_values<atom_core_qresident248::DqCfg<32,4,3,1,1,1,6>,true>();
 }}
 if(dq==false && km==false){switch(order){
 case 0:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,0,0>,false>();
 case 1:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,0,1>,false>();
 case 2:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,0,2>,false>();
 case 3:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,0,3>,false>();
 case 4:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,0,4>,false>();
 case 6:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,0,6>,false>();
 }}
 if(dq==false && km==true){switch(order){
 case 0:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,1,0>,false>();
 case 1:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,1,1>,false>();
 case 2:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,1,2>,false>();
 case 3:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,1,3>,false>();
 case 4:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,1,4>,false>();
 case 6:return resource_values<atom_core_qresident248::DkvCfg<32,4,2,1,1,6>,false>();
 }}
TORCH_CHECK(false,"E237 resource order");}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("attn_bwd_dq",&attn_bwd_dq);m.def("attn_bwd_dkv",&attn_bwd_dkv);m.def("resources",&resources);}
