#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "atom_fwd_qresident.cuh"
namespace {
void check_bf16(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, name, ": a bf16 CUDA tensor");
}

// a token-major operand [T][ld] viewed over `t`: row stride ld (>= H hd), 16-byte aligned rows, and the whole [T][H hd] window inside the storage
void check_tokens(const torch::Tensor& t, const char* name, int64_t T, int64_t ld, int64_t cols) {
  TORCH_CHECK(ld >= cols && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": row stride / alignment");
  TORCH_CHECK((t.storage_offset() + (T - 1) * ld + cols) * t.element_size() <= (int64_t)t.storage().nbytes(), name, ": tensor extents");
}

template <class G>
void launch_fwd(const atom_fwd_qresident257::FwdParams& p, int A) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(atom_fwd_qresident257::attn_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / atom_fwd_qresident257::BM, A / G::R, p.H);
  atom_fwd_qresident257::attn_fwd_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_f32(const torch::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous() && t.numel() == n, name, ": a contiguous fp32 CUDA tensor of ", n, " elements");
}

template<class G> std::vector<int64_t> resource_values(){
 cudaFuncAttributes a; int blocks=0;
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,atom_fwd_qresident257::attn_fwd_kernel<G>));
 C10_CUDA_CHECK(cudaFuncSetAttribute(atom_fwd_qresident257::attn_fwd_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,atom_fwd_qresident257::attn_fwd_kernel<G>,256,G::SMEM));
 cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,at::cuda::current_device()));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,G::SMEM,256,blocks,blocks*8,prop.maxThreadsPerMultiProcessor,prop.multiProcessorCount,a.binaryVersion,a.ptxVersion};
}
} // anonymous
void attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor gate, torch::Tensor out, torch::Tensor lse,
              int64_t A, int64_t L, int64_t H, int64_t hd, int64_t ldq, int64_t ldk, int64_t ldv, int64_t ldg, int64_t ldo,
              double scl, double bscl, int64_t mode, int64_t rows, int64_t nkv, int64_t minb, int64_t ss, torch::Tensor kpen, int64_t kps) {
  for (auto* t : {&q, &k, &v, &bias}) check_bf16(*t, "attn_fwd");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == H * L * L, "attn_fwd: bias [H, L, L] contiguous");
  TORCH_CHECK(L % atom_fwd_qresident257::BM == 0 && L >= atom_fwd_qresident257::BM && A % rows == 0 && A / rows <= 65535 && H <= 65535, "attn_fwd: L a multiple of 128, A a multiple of rows");
  if (ss <= 0) ss = L;
  TORCH_CHECK(ss >= L, "attn_fwd: ss >= L");
  const int64_t T = (A - 1) * ss + L, cols = H * hd;       // tokens spanned by the operands
  check_tokens(q, "attn_fwd q", T, ldq, cols); check_tokens(k, "attn_fwd k", T, ldk, cols); check_tokens(v, "attn_fwd v", T, ldv, cols);
  atom_fwd_qresident257::FwdParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.gate = nullptr; p.lse = nullptr; p.out = nullptr;
  if (mode == atom_fwd_qresident257::MODE_GATE || mode == atom_fwd_qresident257::MODE_PLAIN || mode == atom_fwd_qresident257::MODE_PGATE) {
    check_bf16(out, "attn_fwd out");
    check_tokens(out, "attn_fwd out", T, ldo, cols);
    p.out = out.data_ptr();
    if (mode == atom_fwd_qresident257::MODE_GATE || mode == atom_fwd_qresident257::MODE_PGATE) {
      check_bf16(gate, "attn_fwd gate");
      check_tokens(gate, "attn_fwd gate", T, ldg, cols);
      p.gate = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
    } else if (lse.numel() > 0) {
      check_f32(lse, "attn_fwd lse", A * H * L);
      p.lse = lse.data_ptr<float>();
    }
  } else {
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.numel() == A * L * cols && ldo == cols && ss == L,
                "attn_fwd: out float [A L, H hd]");
    check_f32(lse, "attn_fwd lse", A * H * L);
    p.out = out.data_ptr(); p.lse = lse.data_ptr<float>();
  }
  const int64_t km = kpen.numel() > 0;
  p.kpen = nullptr; p.kps = 0;
  if (km) {
    TORCH_CHECK(kpen.is_cuda() && kpen.scalar_type() == at::kFloat && kpen.is_contiguous() && kps >= L && (A - 1) * kps + L <= kpen.numel() &&
                reinterpret_cast<uintptr_t>(kpen.data_ptr()) % 16 == 0 && kps % 4 == 0, "attn_fwd: kpen fp32 [A][L] (16-byte aligned rows)");
    p.kpen = kpen.data_ptr<float>(); p.kps = kps;
  }
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.ldg = ldg; p.ldo = ldo; p.ss = ss; p.L = (int)L; p.H = (int)H; p.scl = (float)scl; p.bscl = (float)bscl;
  const at::cuda::CUDAGuard guard(q.device());
  const int64_t cfg = ((((hd * 10 + rows) * 10 + nkv) * 10 + minb) * 10 + mode) * 10 + km;
#define FWD_CFG(HD, R, NKV, MINB, MODE, KM) \
  if (cfg == ((((int64_t(HD) * 10 + (R)) * 10 + (NKV)) * 10 + (MINB)) * 10 + (MODE)) * 10 + (KM)) { \
    launch_fwd<atom_fwd_qresident257::FwdCfg<HD, R, NKV, MINB, MODE, KM>>(p, (int)A); return; }
  FWD_CFG(32, 1, 4, 2, 2, 0)
  FWD_CFG(32, 1, 4, 2, 2, 1)
  FWD_CFG(32, 1, 4, 2, 3, 0)
  FWD_CFG(32, 1, 4, 2, 3, 1)
  FWD_CFG(32, 2, 3, 2, 2, 0)
  FWD_CFG(32, 2, 3, 2, 2, 1)
  FWD_CFG(32, 2, 3, 2, 3, 0)
  FWD_CFG(32, 2, 3, 2, 3, 1)
#undef FWD_CFG
 TORCH_CHECK(false,"E257 original HD32 R1/R2 MODE_PLAIN/PGATE only");
}
std::vector<int64_t> resources(int64_t rows,bool gate,bool km){
 if(rows==1 && gate==false && km==false) return resource_values<atom_fwd_qresident257::FwdCfg<32,1,4,2,2,0>>();
 if(rows==1 && gate==false && km==true) return resource_values<atom_fwd_qresident257::FwdCfg<32,1,4,2,2,1>>();
 if(rows==1 && gate==true && km==false) return resource_values<atom_fwd_qresident257::FwdCfg<32,1,4,2,3,0>>();
 if(rows==1 && gate==true && km==true) return resource_values<atom_fwd_qresident257::FwdCfg<32,1,4,2,3,1>>();
 if(rows==2 && gate==false && km==false) return resource_values<atom_fwd_qresident257::FwdCfg<32,2,3,2,2,0>>();
 if(rows==2 && gate==false && km==true) return resource_values<atom_fwd_qresident257::FwdCfg<32,2,3,2,2,1>>();
 if(rows==2 && gate==true && km==false) return resource_values<atom_fwd_qresident257::FwdCfg<32,2,3,2,3,0>>();
 if(rows==2 && gate==true && km==true) return resource_values<atom_fwd_qresident257::FwdCfg<32,2,3,2,3,1>>();
 TORCH_CHECK(false,"E257 supported original R1/R2 MODE_PLAIN/PGATE, HD32 both KM");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("attn_fwd",&attn_fwd);m.def("resources",&resources);}
