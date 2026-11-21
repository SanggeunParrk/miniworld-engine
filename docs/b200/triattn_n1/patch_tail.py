import sys
F = sys.argv[1]; s = open(F).read()
kern = r'''
// ---- the gate + out-projection + residual tail (K = 128 / 256 gate channels; the shared `tail_k`, h100_sources/triattn_bidir_sm90.cu, untouched) with the
// pipeline as template parameters: channels a stage (KCH) and stages (NS). Same arithmetic in the same order (u rounded to bf16, k16 steps in order, the
// product rounded to bf16, then x added): bitwise equal.
template <int K, bool SU, int KCH, int NS>
__global__ void __launch_bounds__(256, 2) tail_b200_k(const bf16* __restrict__ G, const bf16* __restrict__ O, const bf16* __restrict__ Wo, const bf16* __restrict__ X,
                                                      bf16* __restrict__ out, bf16* __restrict__ U, long LL, long ldg, long ldo) {
  using namespace gb;
  constexpr int TM = 128, SP = 136, WS = K + 8, NKC = K / KCH, STG = 2 * KCH * SP;
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sWo = reinterpret_cast<bf16*>(smraw);
  bf16* sT = sWo + 128 * WS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, ur = tid >> 4, uc = tid & 15;
  for (int i = tid; i < 128 * (K / 8); i += 256) {
    const int n = i / (K / 8), c = i % (K / 8);
    cp16(sWo + n * WS + c * 8, Wo + (long)n * K + c * 8);
  }
  cp_commit();
  const long ntiles = LL / TM;
  for (long tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const long tok0 = tile * TM;
    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    auto issue = [&](int kc) {
      if (kc < NKC) {
        bf16* g = sT + (kc % NS) * STG;
#pragma unroll
        for (int r = 0; r < KCH / 16; ++r) {
          cp16(g + (r * 16 + ur) * SP + uc * 8, G + (long)(kc * KCH + r * 16 + ur) * ldg + tok0 + uc * 8);
          cp16(g + KCH * SP + (r * 16 + ur) * SP + uc * 8, O + (long)(kc * KCH + r * 16 + ur) * ldo + tok0 + uc * 8);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int i = 0; i < NS - 1; ++i) issue(i);
    for (int kc = 0; kc < NKC; ++kc) {
      cp_wait<NS - 2>();
      bf16* g = sT + (kc % NS) * STG;
      bf16* o = g + KCH * SP;
#pragma unroll
      for (int r = 0; r < KCH / 16; ++r) {  // u = bf16(sigmoid(g) o), in place of o, for this thread's own units
        float gv[8], ov[8], rv[8];
        unpack8(*reinterpret_cast<const uint4*>(g + (r * 16 + ur) * SP + uc * 8), gv);
        unpack8(*reinterpret_cast<const uint4*>(o + (r * 16 + ur) * SP + uc * 8), ov);
#pragma unroll
        for (int e = 0; e < 8; ++e) rv[e] = (1.f / (1.f + __expf(-gv[e]))) * ov[e];
        const uint4 pu = pack8(rv);
        *reinterpret_cast<uint4*>(o + (r * 16 + ur) * SP + uc * 8) = pu;
        if (SU) *reinterpret_cast<uint4*>(U + (long)(kc * KCH + r * 16 + ur) * LL + tok0 + uc * 8) = pu;
      }
      __syncthreads();
      issue(kc + NS - 1);
#pragma unroll
      for (int kk = 0; kk < KCH / 16; ++kk) {
        unsigned a[4];
        {
          const int mi = lane >> 3;
          ldsm4t(a, o + (kk * 16 + (mi >> 1) * 8 + (lane & 7)) * SP + w * 16 + (mi & 1) * 8);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          unsigned b[4];
          const int mi = lane >> 3;
          ldsm4(b, sWo + (j * 16 + (mi >> 1) * 8 + (lane & 7)) * WS + kc * KCH + kk * 16 + (mi & 1) * 8);
          mma16816(acc[2 * j], a, b[0], b[1]);
          mma16816(acc[2 * j + 1], a, b[2], b[3]);
        }
      }
    }
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int c = j * 8 + 2 * (lane & 3), r0 = w * 16 + (lane >> 2);
      *reinterpret_cast<__nv_bfloat162*>(sT + r0 * SP + c) = __floats2bfloat162_rn(acc[j][0], acc[j][1]);
      *reinterpret_cast<__nv_bfloat162*>(sT + (r0 + 8) * SP + c) = __floats2bfloat162_rn(acc[j][2], acc[j][3]);
    }
    __syncthreads();
    for (int i = tid; i < TM * 16; i += 256) {
      const int r = i >> 4, ch = i & 15;
      float p[8], xf[8], q[8];
      unpack8(*reinterpret_cast<const uint4*>(sT + r * SP + ch * 8), p);
      unpack8(*reinterpret_cast<const uint4*>(X + (tok0 + r) * 128 + ch * 8), xf);
#pragma unroll
      for (int e = 0; e < 8; ++e) q[e] = p[e] + xf[e];
      *reinterpret_cast<uint4*>(out + (tok0 + r) * 128 + ch * 8) = pack8(q);
    }
    __syncthreads();
  }
}
'''
# place after the gate_bwd128 kernel (end of its closing brace before "}  // namespace\n\n// -> (dx bf16")
s = s.replace("}  // namespace\n\n// -> (dx bf16", kern + "}  // namespace\n\n// -> (dx bf16", 1)
host = r'''
// g: [K, LL] gate planes (row stride ldg), o: [K, LL], wo [128, K], x / out [LL, 128]; u: optional [K, LL] (training)
torch::Tensor tail_b200(torch::Tensor g, torch::Tensor o, torch::Tensor wo, torch::Tensor x, torch::Tensor out, c10::optional<torch::Tensor> u, int64_t kch, int64_t ns, int64_t gm) {
  const long K = wo.size(1), LL = g.size(1);
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == torch::kBFloat16 && o.scalar_type() == torch::kBFloat16 && wo.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kBFloat16 &&
              out.scalar_type() == torch::kBFloat16 && (K == 128 || K == 256) && wo.size(0) == 128 && wo.is_contiguous() && g.size(0) == K && o.size(0) == K && o.size(1) == LL &&
              g.stride(1) == 1 && o.stride(1) == 1 && g.stride(0) % 8 == 0 && o.stride(0) % 8 == 0 && LL % 128 == 0 && x.is_contiguous() && out.is_contiguous() && x.size(0) == LL && x.size(1) == 128 &&
              out.sizes() == x.sizes() && reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0, "tail_b200: bf16 planes [K, LL], wo [128, K]");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long ntiles = LL / 128;
  const int grid = (int)std::min<long>(ntiles, (long)sms * gm);
  bf16* up = nullptr;
  if (u.has_value()) {
    TORCH_CHECK(u->is_cuda() && u->scalar_type() == torch::kBFloat16 && u->dim() == 2 && u->size(0) == K && u->size(1) == LL && u->is_contiguous() && reinterpret_cast<uintptr_t>(u->data_ptr()) % 16 == 0, "tail_b200: u [K, LL]");
    up = reinterpret_cast<bf16*>(u->data_ptr<at::BFloat16>());
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, size_t smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kern<<<grid, 256, smem, stream>>>(reinterpret_cast<const bf16*>(g.data_ptr<at::BFloat16>()), reinterpret_cast<const bf16*>(o.data_ptr<at::BFloat16>()), reinterpret_cast<const bf16*>(wo.data_ptr<at::BFloat16>()),
                                      reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()), reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()), up, LL, g.stride(0), o.stride(0));
  };
  const size_t ws = (size_t)128 * (K + 8) * 2;
#define TB_CASE(KK, KCHV, NSV) if (K == KK && kch == KCHV && ns == NSV) { const size_t sm = ws + (size_t)NSV * 2 * KCHV * 136 * 2; if (up) go(tail_b200_k<KK, true, KCHV, NSV>, sm); else go(tail_b200_k<KK, false, KCHV, NSV>, sm); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
  TB_CASE(256, 16, 4) TB_CASE(256, 16, 5) TB_CASE(256, 16, 3) TB_CASE(256, 32, 2) TB_CASE(256, 32, 3)
  TB_CASE(128, 16, 4) TB_CASE(128, 16, 5) TB_CASE(128, 16, 6) TB_CASE(128, 16, 8) TB_CASE(128, 32, 3) TB_CASE(128, 32, 4) TB_CASE(128, 32, 5)
#undef TB_CASE
  TORCH_CHECK(false, "tail_b200: unsupported (K, kch, ns)");
  return out;
}
'''
s = s.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n", host + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"tail_b200\", &tail_b200, \"B200 gate + out-projection + residual tail\");\n", 1)
open(F, "w").write(s); print("ok")
