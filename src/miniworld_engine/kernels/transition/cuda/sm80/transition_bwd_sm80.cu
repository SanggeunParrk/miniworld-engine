// transition_bwd_sm80.cu -- torch bindings of the fused A100 (sm_80) Transition BACKWARD for D = 64 / 128 and the hidden widths H = 128 .. 512 (the PW and X roles of fused_sm80,
// tr_bwd_g_sm80.cuh / tr_bwd_sm80.cuh generalised over (D, H)), on the current CUDA stream:
//   bwd  PW (per hidden slice x row replica: a, b, dh, the SwiGLU backward, dWa / dWb / dWs^T partials, dA | dB blocks), X (d_xn = [dA | dB] [Wa; Wb], the LayerNorm backward
//        + residual -> dx, dgamma / dbeta partials; the bare FFN: dx = d_xn), then the partial sums in a fixed order -> the gradients in the parameters' dtypes
//   pack the weight layouts of the two roles in one launch (gather tables built by the Python side)
// Rows must be a whole number of 256-row tiles.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "tr_bwd_g_sm80.cuh"

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

__device__ __forceinline__ void cvt_to(float& o, float v) { o = v; }
__device__ __forceinline__ void cvt_to(__nv_bfloat16& o, float v) { o = __float2bfloat16_rn(v); }

// the partial sums in one launch, in a fixed order, straight into the parameters' dtypes: part [nrep][dWa (2x) | dWb | dWs^T][H][D], dgb [ng][dgamma | dbeta][D]
template <bool LN, class TW, class TA>
__global__ void finalize_g_kernel(const float* __restrict__ part, int nrep, const float* __restrict__ dgb, int ng, float sa, int D, int H, TW* __restrict__ dwa,
                                  TW* __restrict__ dwb, TW* __restrict__ dws, TA* __restrict__ dgam, TA* __restrict__ dbeta) {
  const int N = H * D;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < 3 * N) {
    float v = 0.f;
    for (int r = 0; r < nrep; ++r) v += part[(size_t)r * 3 * N + j];
    const int m = j / N, e = j - m * N;
    if (m == 0) cvt_to(dwa[e], v * sa);
    else if (m == 1) cvt_to(dwb[e], v);
    else cvt_to(dws[(e % D) * H + e / D], v);                                 // dWs^T [h][d] -> dWs [d][h]
  } else if (LN && j < 3 * N + 2 * D) {
    const int k = j - 3 * N;
    float v = 0.f;
    for (int g = 0; g < ng; ++g) v += dgb[g * 2 * D + k];
    if (k < D) cvt_to(dgam[k], v);
    else cvt_to(dbeta[k - D], v);
  }
}

template <int D, int H, bool LN>
std::vector<torch::Tensor> bwd_t(torch::Tensor dy, torch::Tensor x, torch::Tensor a_in, torch::Tensor stats, torch::Tensor wdw, torch::Tensor wx, torch::Tensor gamma,
                                 int64_t nrep_in, at::ScalarType adt, at::ScalarType wdt) {
  using PG = a100::PwCfgT<D, H>;
  using XG = a100::BwdCfg<128 * D, 8, 2, D, H>;
  const int64_t T = x.size(0);
  TORCH_CHECK(dy.is_contiguous() && x.is_contiguous() && a_in.is_contiguous() && x.size(1) == D, "contiguous [T, D] inputs");
  TORCH_CHECK(T % 256 == 0 && dy.size(0) == T && a_in.size(0) == T, "T % 256 and matching rows");
  TORCH_CHECK(wdw.numel() == (int64_t)PG::NSL * PG::WRES / 2 && wx.numel() == (int64_t)XG::NCHUNK * XG::SLOT / 2, "packed weights");
  TORCH_CHECK((adt == at::kFloat || adt == at::kBFloat16) && (wdt == at::kFloat || wdt == at::kBFloat16), "f32 / bf16 gradients");
  // PW grid: NSL slices x nrep = two waves, with at least 4 row stages (64 rows) per replica: a small T would otherwise write (and sum) mostly empty partial sums
  const int nsm = num_sms(), nrep = nrep_in > 0 ? (int)nrep_in : std::max(1, std::min(2 * nsm / PG::NSL, (int)(T / PG::RS / 4)));
  auto f32 = x.options().dtype(at::kFloat);
  auto ab = torch::empty({3 * T * H}, x.options());                           // [T / 16][H / 16 K][dA | dB | (unused)][32 lanes] 16 B
  auto part = torch::empty({nrep, 3, H, D}, f32);
  auto dx = torch::empty_like(x);
  const int gx = (int)std::min<int64_t>(2 * (T / 256), nsm);               // fewer tiles than half the SMs: half tiles (the schedule deals them first), so twice the CTAs share the rows
  auto dgb = torch::empty({LN ? gx : 0, 2, D}, f32);
  auto st = at::cuda::getCurrentCUDAStream();

  a100::PWParams q{};
  q.x = cptr<__nv_bfloat16>(a_in); q.dy = cptr<__nv_bfloat16>(dy); q.wdw = cptr<__nv_bfloat16>(wdw); q.ab = mptr<uint4>(ab); q.part = mptr<float>(part);
  q.T = (int)T;
  set_smem(a100::tr_bwd_pwg_kernel<PG>, PG::SMEM);
  a100::tr_bwd_pwg_kernel<PG><<<PG::NSL * nrep, 256, PG::SMEM, st>>>(q);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  a100::BwdParams p{};
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.wx = cptr<__nv_bfloat16>(wx); p.gamma = LN ? cptr<float>(gamma) : nullptr;
  p.stats = LN ? mptr<float2>(stats) : nullptr; p.ab = mptr<uint4>(ab); p.dx = mptr<__nv_bfloat16>(dx); p.dgb = LN ? mptr<float>(dgb) : nullptr;
  p.T = (int)T; p.num_tiles = (int)(T / 256);
  constexpr int xsmem = XG::SMEM + a100::XA_BYTES;
  set_smem(a100::tr_bwd_xg_kernel<XG, LN>, xsmem);
  a100::tr_bwd_xg_kernel<XG, LN><<<gx, 256, xsmem, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto dgam = torch::empty({LN ? D : 0}, x.options().dtype(adt)), dbeta = torch::empty({LN ? D : 0}, x.options().dtype(adt));
  auto dwab = torch::empty({2 * H, D}, x.options().dtype(wdt));              // [dWa; dWb] in one buffer (the autograd function splits it)
  auto dws = torch::empty({D, H}, x.options().dtype(wdt));
  const int nfin = 3 * H * D + (LN ? 2 * D : 0);
#define FIN(TW, TA) finalize_g_kernel<LN, TW, TA><<<(nfin + 255) / 256, 256, 0, st>>>(mptr<float>(part), nrep, LN ? mptr<float>(dgb) : nullptr, gx, 0.5f, D, H,  \
      mptr<TW>(dwab), mptr<TW>(dwab) + (size_t)H * D, mptr<TW>(dws), LN ? mptr<TA>(dgam) : nullptr, LN ? mptr<TA>(dbeta) : nullptr)
  if (wdt == at::kFloat && adt == at::kFloat) FIN(float, float);
  else if (wdt == at::kFloat) FIN(float, __nv_bfloat16);
  else if (adt == at::kFloat) FIN(__nv_bfloat16, float);
  else FIN(__nv_bfloat16, __nv_bfloat16);
#undef FIN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dgam, dbeta, dwab, dws};
}

template <int D, int H>
std::vector<torch::Tensor> bwd_ln(bool ln, torch::Tensor dy, torch::Tensor x, torch::Tensor a_in, torch::Tensor stats, torch::Tensor wdw, torch::Tensor wx,
                                  torch::Tensor gamma, int64_t nrep, at::ScalarType adt, at::ScalarType wdt) {
  return ln ? bwd_t<D, H, true>(dy, x, a_in, stats, wdw, wx, gamma, nrep, adt, wdt) : bwd_t<D, H, false>(dy, x, a_in, stats, wdw, wx, gamma, nrep, adt, wdt);
}
}  // namespace

// dy, x [T, D] bf16; a_in = xn (LayerNorm) or x (FFN); stats [T, 2] f32 (LN); wdw | wx: the packed weights; gamma f32 [D] (LN)
// -> (dx, dgamma, dbeta in adt (empty without LN), [dWa; dWb] [2 H, D], dWs [D, H] in wdt)
std::vector<torch::Tensor> bwd(int64_t D, int64_t H, bool ln, torch::Tensor dy, torch::Tensor x, torch::Tensor a_in, torch::Tensor stats, torch::Tensor wdw,
                               torch::Tensor wx, torch::Tensor gamma, int64_t nrep, at::ScalarType adt, at::ScalarType wdt) {
  if (D == 64 && H == 128) return bwd_ln<64, 128>(ln, dy, x, a_in, stats, wdw, wx, gamma, nrep, adt, wdt);
  if (D == 64 && H == 256) return bwd_ln<64, 256>(ln, dy, x, a_in, stats, wdw, wx, gamma, nrep, adt, wdt);
  if (D == 128 && H == 256) return bwd_ln<128, 256>(ln, dy, x, a_in, stats, wdw, wx, gamma, nrep, adt, wdt);
  if (D == 128 && H == 512) return bwd_ln<128, 512>(ln, dy, x, a_in, stats, wdw, wx, gamma, nrep, adt, wdt);
  TORCH_CHECK(false, "fused backward: no build for D = ", D, ", H = ", H);
  return {};
}

// weight packing in one launch (see transition_fwd_sm80.cu): out16[j] = cvt(scale * src[idx16[j]])
__global__ void pack_kernel(const __nv_bfloat16* __restrict__ wa, const __nv_bfloat16* __restrict__ wb, const __nv_bfloat16* __restrict__ ws, const int* __restrict__ idx16,
                            int n16, uint16_t* __restrict__ out16) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n16) {
    const int e = idx16[j], k = e & 0x3ffffff, src = (e >> 26) & 3;
    float v = __bfloat162float((src == 0 ? wa : src == 1 ? wb : ws)[k]);
    if (e & (1 << 28)) v *= 0.5f;
    out16[j] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
  }
}

void pack(torch::Tensor wa, torch::Tensor wb, torch::Tensor ws, torch::Tensor idx16, torch::Tensor out16) {
  TORCH_CHECK(wa.scalar_type() == at::kBFloat16 && wb.scalar_type() == at::kBFloat16 && ws.scalar_type() == at::kBFloat16, "bf16 weights");
  TORCH_CHECK(wa.is_contiguous() && wb.is_contiguous() && ws.is_contiguous(), "contiguous weights");
  const int n16 = (int)idx16.numel();
  TORCH_CHECK(out16.numel() == n16, "pack sizes");
  pack_kernel<<<(n16 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(cptr<__nv_bfloat16>(wa), cptr<__nv_bfloat16>(wb), cptr<__nv_bfloat16>(ws), cptr<int>(idx16), n16,
                                                                             mptr<uint16_t>(out16));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bwd", &bwd);
  m.def("pack", &pack);
}
