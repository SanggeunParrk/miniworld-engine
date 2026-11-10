// transition_fwd_sm80.cu -- torch bindings of the fused A100 (sm_80) Transition FORWARD for the widths D = 64 / 128 and the hidden widths H = 128 .. 512 (tr_fwd_sm80.cuh, the
// kernel of fused_sm80 generalised over (D, H)), on the current CUDA stream:
//   fwd   x [T, D] bf16 -> (out [T, D], xn [T, D] | [0], stats [T, 2] f32 | [0]);  any T (rows past the end are predicated)
//   pack  the weight / affine layouts of the kernel in one launch (gather tables built by the Python side)
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "tr_fwd_sm80.cuh"

namespace {
int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}
template <class K>
void set_smem(K kernel, int bytes) {
  static std::vector<const void*> done;
  const void* k = reinterpret_cast<const void*>(kernel);
  if (std::find(done.begin(), done.end(), k) == done.end()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    done.push_back(k);
  }
}
template <class T> const T* cptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <class T> T* mptr(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

template <int D, int H, bool LN>
std::vector<torch::Tensor> fwd_t(torch::Tensor x, torch::Tensor w, torch::Tensor gb, double eps, bool save) {
  using G = a100::TrCfgT<D, H>;
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) == D && x.scalar_type() == at::kBFloat16, "x: [T, D] bf16");
  TORCH_CHECK(w.numel() == (int64_t)G::NCHUNK * G::SLOT / 2 && (!LN || gb.numel() == 2 * D), "packed weights");
  const int64_t T = x.size(0);
  auto out = torch::empty_like(x);
  auto xn = save ? torch::empty_like(x) : torch::empty({0}, x.options());
  auto stats = torch::empty({save ? T : 0, 2}, x.options().dtype(at::kFloat));
  a100::TrParams p;
  p.x = cptr<__nv_bfloat16>(x); p.w = cptr<__nv_bfloat16>(w); p.gb = LN ? cptr<float4>(gb) : nullptr; p.out = mptr<__nv_bfloat16>(out);
  p.stats = save ? mptr<float2>(stats) : nullptr;
  p.xn = save ? mptr<__nv_bfloat16>(xn) : nullptr;
  p.trace = nullptr;
  p.T = (int)T; p.num_tiles = (p.T + G::BM - 1) / G::BM; p.eps = (float)eps;
  if (T == 0) return {out, xn, stats};
  const int g = std::min(p.num_tiles, num_sms() * G::MINB);
  TORCH_CHECK(p.num_tiles / g + 2 <= G::MAX_ITEMS, "fwd: too many work items per CTA");
  set_smem(a100::tr_fwd_kernel<G, LN>, G::SMEM);
  a100::tr_fwd_kernel<G, LN><<<g, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, xn, stats};
}
}  // namespace

// ln: the LayerNorm + residual Transition; else the bare SwiGLU FFN (no LayerNorm, no residual: xn / stats are never written, gb is not read)
std::vector<torch::Tensor> fwd(int64_t D, int64_t H, bool ln, torch::Tensor x, torch::Tensor w, torch::Tensor gb, double eps, bool save) {
#define FWD_CASE(DV, HV)                                                                                                                              \
  if (D == DV && H == HV) return ln ? fwd_t<DV, HV, true>(x, w, gb, eps, save) : fwd_t<DV, HV, false>(x, w, gb, eps, false);
  FWD_CASE(64, 128) FWD_CASE(64, 256) FWD_CASE(128, 256) FWD_CASE(128, 512)
#undef FWD_CASE
  TORCH_CHECK(false, "fused forward: no build for D = ", D, ", H = ", H);
  return {};
}

// weight packing in one launch: out16[j] = cvt(scale * src[idx16[j]]), out32[j] = src32[idx32[j]].  idx16 entry: bits 0-25 element,
// 26-27 source (Wa | Wb | Ws, bf16), 28: x 0.5, 29: f16 (else bf16); idx32: bits 0-25 element, 26: beta (else gamma), f32 sources
__global__ void pack_kernel(const __nv_bfloat16* __restrict__ wa, const __nv_bfloat16* __restrict__ wb, const __nv_bfloat16* __restrict__ ws,
                            const float* __restrict__ gamma, const float* __restrict__ beta, const int* __restrict__ idx16, int n16,
                            uint16_t* __restrict__ out16, const int* __restrict__ idx32, int n32, float* __restrict__ out32) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n16) {
    const int e = idx16[j], k = e & 0x3ffffff, src = (e >> 26) & 3;
    float v = __bfloat162float((src == 0 ? wa : src == 1 ? wb : ws)[k]);
    if (e & (1 << 28)) v *= 0.5f;
    out16[j] = (e & (1 << 29)) ? __half_as_ushort(__float2half_rn(v)) : __bfloat16_as_ushort(__float2bfloat16_rn(v));
  } else if (j - n16 < n32) {
    const int e = idx32[j - n16];
    out32[j - n16] = ((e >> 26) & 1 ? beta : gamma)[e & 0x3ffffff];
  }
}

void pack(torch::Tensor wa, torch::Tensor wb, torch::Tensor ws, torch::Tensor gamma, torch::Tensor beta, torch::Tensor idx16, torch::Tensor idx32,
          torch::Tensor out16, torch::Tensor out32) {
  TORCH_CHECK(wa.scalar_type() == at::kBFloat16 && wb.scalar_type() == at::kBFloat16 && ws.scalar_type() == at::kBFloat16, "bf16 weights");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat, "f32 affine");
  TORCH_CHECK(wa.is_contiguous() && wb.is_contiguous() && ws.is_contiguous(), "contiguous weights");
  const int n16 = (int)idx16.numel(), n32 = (int)idx32.numel();
  TORCH_CHECK(out16.numel() == n16 && out32.numel() == n32, "pack sizes");
  pack_kernel<<<(n16 + n32 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      cptr<__nv_bfloat16>(wa), cptr<__nv_bfloat16>(wb), cptr<__nv_bfloat16>(ws), cptr<float>(gamma), cptr<float>(beta), cptr<int>(idx16), n16,
      mptr<uint16_t>(out16), cptr<int>(idx32), n32, mptr<float>(out32));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd);
  m.def("pack", &pack);
}
