// transition_sm80.cu -- torch bindings of the A100 (sm_80) Transition kernels, on the current CUDA stream.
//   fwd:  one kernel (LayerNorm, [a | b] GEMM, SwiGLU, squeeze + residual); training also writes xn and (mean, rstd)
//   bwd:  two kernels -- PW (per 64-unit hidden slice: a, b, dh, the SwiGLU backward, dWa / dWb / dWs^T partials, dA | dB blocks)
//         then X (d_xn = [dA | dB] [Wa; Wb], LayerNorm backward + residual -> dx, dgamma / dbeta partials) -- and the partial sums.
// Sources and records: experiments/a100_transition_fwd, experiments/a100_transition_bwd.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "tr_bwd_pw_sm80.cuh"

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
}  // namespace

// x [T, 128] bf16 -> (out [T, 128], xn [T, 128] | [0], stats [T, 2] f32 | [0])
std::vector<torch::Tensor> fwd(torch::Tensor x, torch::Tensor w, torch::Tensor gb, double eps, bool save) {
  using G = a100::TrCfg;
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) == 128 && x.scalar_type() == at::kBFloat16, "x: [T,128] bf16");
  TORCH_CHECK(w.numel() == (int64_t)G::NCHUNK * G::SLOT / 2 && gb.numel() == 256, "packed weights");
  const int64_t T = x.size(0);
  auto out = torch::empty_like(x);
  auto xn = save ? torch::empty_like(x) : torch::empty({0}, x.options());
  auto stats = torch::empty({save ? T : 0, 2}, x.options().dtype(at::kFloat));
  a100::TrParams p;
  p.x = cptr<__nv_bfloat16>(x); p.w = cptr<__nv_bfloat16>(w); p.gb = cptr<float4>(gb); p.out = mptr<__nv_bfloat16>(out);
  p.stats = save ? mptr<float2>(stats) : nullptr;
  p.xn = save ? mptr<__nv_bfloat16>(xn) : nullptr;
  p.trace = nullptr;
  p.T = (int)T; p.num_tiles = (p.T + G::BM - 1) / G::BM; p.eps = (float)eps;
  if (T == 0) return {out, xn, stats};
  const int g = std::min(p.num_tiles, num_sms() * G::MINB);
  TORCH_CHECK(p.num_tiles / g + 2 <= G::MAX_ITEMS, "fwd: too many work items per CTA");
  set_smem(a100::tr_fwd_kernel<G>, G::SMEM);
  a100::tr_fwd_kernel<G><<<g, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, xn, stats};
}

// the partial sums in one launch, in a fixed order, straight into the parameters' dtypes: part [nrep][dWa (2x) | dWb | dWs^T][512][128],
// dgb [ng][dgamma | dbeta][128]
__device__ __forceinline__ void cvt_to(float& o, float v) { o = v; }
__device__ __forceinline__ void cvt_to(__nv_bfloat16& o, float v) { o = __float2bfloat16_rn(v); }
template <class TW, class TA>
__global__ void finalize_kernel(const float* __restrict__ part, int nrep, const float* __restrict__ dgb, int ng, float sa, TW* __restrict__ dwa,
                                TW* __restrict__ dwb, TW* __restrict__ dws, TA* __restrict__ dgam, TA* __restrict__ dbeta) {
  constexpr int N = 512 * 128;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < 3 * N) {
    float v = 0.f;
    for (int r = 0; r < nrep; ++r) v += part[(size_t)r * 3 * N + j];
    const int m = j / N, e = j - m * N;
    if (m == 0) cvt_to(dwa[e], v * sa);
    else if (m == 1) cvt_to(dwb[e], v);
    else cvt_to(dws[(e & 127) * 512 + (e >> 7)], v);    // dWs^T [h][d] -> dWs [d][h]
  } else if (j < 3 * N + 256) {
    const int k = j - 3 * N;
    float v = 0.f;
    for (int g = 0; g < ng; ++g) v += dgb[g * 256 + k];
    if (k < 128) cvt_to(dgam[k], v);
    else cvt_to(dbeta[k - 128], v);
  }
}

// dy, x, xn [T, 128] bf16, stats [T, 2] -> (dx, dgamma, dbeta in adt, dWa, dWb, dWs in wdt)
std::vector<torch::Tensor> bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor xn, torch::Tensor stats, torch::Tensor wdw, torch::Tensor wx,
                               torch::Tensor gamma, torch::Tensor beta, double eps, int64_t nrep_in, at::ScalarType adt, at::ScalarType wdt) {
  TORCH_CHECK(dy.is_contiguous() && x.is_contiguous() && xn.is_contiguous() && stats.is_contiguous(), "contiguous inputs");
  const int64_t T = x.size(0);
  TORCH_CHECK(T % 256 == 0 && dy.size(0) == T && xn.size(0) == T && stats.size(0) == T, "T % 256 and matching rows");
  const int nsm = num_sms(), nrep = nrep_in > 0 ? (int)nrep_in : std::max(1, 2 * nsm / 8);   // PW grid: 8 slices x nrep = two waves (27: 1.4 % faster than 13 at L768)
  auto f32 = x.options().dtype(at::kFloat);
  auto ab = torch::empty({3 * T * 512}, x.options());                // [T / 16][32 K][dA | dB | (unused)][32 lanes] 16 B
  auto part = torch::empty({nrep, 3, 512, 128}, f32);
  auto dx = torch::empty_like(x);
  const int gx = (int)std::min<int64_t>(T / 256, nsm);
  auto dgb = torch::empty({gx, 2, 128}, f32);
  auto st = at::cuda::getCurrentCUDAStream();

  a100::PWParams q;
  q.x = cptr<__nv_bfloat16>(xn); q.dy = cptr<__nv_bfloat16>(dy); q.wdw = cptr<__nv_bfloat16>(wdw);
  q.gamma = cptr<float>(gamma); q.beta = cptr<float>(beta); q.stats = mptr<float2>(stats); q.ab = mptr<uint4>(ab);
  q.part = mptr<float>(part); q.T = (int)T; q.eps = (float)eps;
  set_smem(a100::tr_bwd_pw_kernel, a100::CfgDW::SMEM);
  a100::tr_bwd_pw_kernel<<<8 * nrep, 256, a100::CfgDW::SMEM, st>>>(q);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  a100::BwdParams p{};
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.wx = cptr<__nv_bfloat16>(wx); p.gamma = cptr<float>(gamma);
  p.stats = mptr<float2>(stats); p.ab = mptr<uint4>(ab); p.dx = mptr<__nv_bfloat16>(dx); p.dgb = mptr<float>(dgb);
  p.T = (int)T; p.num_tiles = (int)(T / 256); p.eps = (float)eps;
  set_smem(a100::tr_bwd_x_kernel, a100::X_SMEM);
  a100::tr_bwd_x_kernel<<<gx, 256, a100::X_SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  TORCH_CHECK((adt == at::kFloat || adt == at::kBFloat16) && (wdt == at::kFloat || wdt == at::kBFloat16), "f32 / bf16 gradients");
  auto dgam = torch::empty({128}, x.options().dtype(adt)), dbeta = torch::empty({128}, x.options().dtype(adt));
  auto dwa = torch::empty({512, 128}, x.options().dtype(wdt)), dwb = torch::empty({512, 128}, x.options().dtype(wdt));
  auto dws = torch::empty({128, 512}, x.options().dtype(wdt));
  const int nfin = 3 * 512 * 128 + 256;
  const float sa = PW_2DA ? 0.5f : 1.f;
#define FIN(TW, TA) finalize_kernel<TW, TA><<<(nfin + 255) / 256, 256, 0, st>>>(mptr<float>(part), nrep, mptr<float>(dgb), gx, sa, \
      mptr<TW>(dwa), mptr<TW>(dwb), mptr<TW>(dws), mptr<TA>(dgam), mptr<TA>(dbeta))
  if (wdt == at::kFloat && adt == at::kFloat) FIN(float, float);
  else if (wdt == at::kFloat) FIN(float, __nv_bfloat16);
  else if (adt == at::kFloat) FIN(__nv_bfloat16, float);
  else FIN(__nv_bfloat16, __nv_bfloat16);
#undef FIN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dgam, dbeta, dwa, dwb, dws};
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
  m.def("bwd", &bwd);
}
