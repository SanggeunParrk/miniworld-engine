import sys
F = sys.argv[1]; s = open(F).read()
kernel = r'''
// ---- gate backward with the du GEMM (K = 128): the shared `gate_bwd_fused_k<128>` (h100_sources/triattn_bidir_sm90.cu, untouched), bitwise the same
// arithmetic, with the gate / o planes' loads of a tile issued BEFORE its mma.sync GEMM (the shared kernel loads them after it: one exposed HBM round
// trip a tile; 2 CTAs an SM do not cover it).
namespace gb {
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
}  // namespace gb

__global__ void __launch_bounds__(256, 2) gate_bwd128_k(const bf16* __restrict__ DY, const bf16* __restrict__ Wo, const bf16* __restrict__ G, const bf16* __restrict__ O,
                                                        bf16* __restrict__ dout, bf16* __restrict__ dg, long LL, long ldg, long ldo, long lddg) {
  using namespace gb;
  constexpr int K = 128, TMB = 64, SD = 136, WS = K + 8, WR = K / 8, BUF = 9216;
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sWo = reinterpret_cast<bf16*>(smraw);
  bf16* sDy = sWo + 128 * WS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, q4 = lane & 3, g8 = lane >> 2, mi = lane >> 3;
  for (int i = tid; i < 128 * (K / 8); i += 256) {
    const int n = i / (K / 8), c = i % (K / 8);
    cp16(sWo + n * WS + c * 8, Wo + (long)n * K + c * 8);
  }
  cp_commit();
  const long ntiles = LL / TMB;
  auto issue = [&](long tile, int s) {
    for (int i = tid; i < TMB * 16; i += 256) {
      const int r = i >> 4, ch = i & 15;
      cp16(sDy + s * BUF + r * SD + ch * 8, DY + (tile * TMB + r) * 128 + ch * 8);
    }
    cp_commit();
  };
  if ((long)blockIdx.x < ntiles) issue(blockIdx.x, 0);
  else cp_commit();
  int it = 0;
  for (long tile = blockIdx.x; tile < ntiles; tile += gridDim.x, ++it) {
    if (tile + gridDim.x < ntiles) issue(tile + gridDim.x, (it + 1) & 1);
    else cp_commit();
    const long tok0 = tile * TMB;
    uint4 gg[4], oo[4];                                  // this warp's 16 channels of the gate / o planes: in flight during the GEMM
    {
      const long tk = tok0 + (lane & 7) * 8;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const long row = w * WR + i * 4 + (lane >> 3);
        gg[i] = __ldcs(reinterpret_cast<const uint4*>(G + row * ldg + tk));
        oo[i] = __ldcs(reinterpret_cast<const uint4*>(O + row * ldo + tk));
      }
    }
    cp_wait<1>();
    __syncthreads();
    const bf16* dy = sDy + (it & 1) * BUF;
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      unsigned a[4];
      ldsm4t(a, sWo + (ks * 16 + (mi >> 1) * 8 + (lane & 7)) * WS + w * WR + (mi & 1) * 8);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned b[4];
        ldsm4(b, dy + (j * 16 + (mi >> 1) * 8 + (lane & 7)) * SD + ks * 16 + (mi & 1) * 8);
        mma16816(acc[2 * j], a, b[0], b[1]);
        mma16816(acc[2 * j + 1], a, b[2], b[3]);
      }
    }
    __syncthreads();
    bf16* stg = sDy + (it & 1) * BUF + w * 16 * 72;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = j * 8 + 2 * q4;
      *reinterpret_cast<__nv_bfloat162*>(stg + g8 * 72 + c) = __floats2bfloat162_rn(acc[j][0], acc[j][1]);
      *reinterpret_cast<__nv_bfloat162*>(stg + (g8 + 8) * 72 + c) = __floats2bfloat162_rn(acc[j][2], acc[j][3]);
    }
    __syncwarp();
    uint4 du[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) du[i] = *reinterpret_cast<const uint4*>(stg + (i * 4 + (lane >> 3)) * 72 + (lane & 7) * 8);
    __syncwarp();
    const long tk = tok0 + (lane & 7) * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float a[8], gv[8], ovv[8], r[8], q[8];
      unpack8(du[i], a);
      unpack8(gg[i], gv);
      unpack8(oo[i], ovv);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float s = 1.f / (1.f + __expf(-gv[e]));
        r[e] = a[e] * s;
        q[e] = a[e] * ovv[e] * s * (1.f - s);
      }
      const long row = w * WR + i * 4 + (lane >> 3);
      *reinterpret_cast<uint4*>(dout + row * LL + tk) = pack8(r);
      *reinterpret_cast<uint4*>(dg + row * lddg + tk) = pack8(q);
    }
    __syncthreads();
  }
}
'''
s = s.replace("}  // namespace\n\n// -> (dx bf16", kernel + "}  // namespace\n\n// -> (dx bf16", 1)
host = '''
// dy [LL, 128], wo [128, 128], g / o [128, LL] gate planes -> do [128, LL]; dg (a [128, LL] view) is written
torch::Tensor gate_bwd128(torch::Tensor dy, torch::Tensor wo, torch::Tensor g, torch::Tensor o, torch::Tensor dg, int64_t gm) {
  const long LL = dy.size(0);
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == torch::kBFloat16 && dy.is_contiguous() && dy.dim() == 2 && dy.size(1) == 128 && wo.scalar_type() == torch::kBFloat16 &&
              wo.is_contiguous() && wo.dim() == 2 && wo.size(0) == 128 && wo.size(1) == 128 && LL % 64 == 0 && g.size(0) == 128 && g.size(1) == LL && o.size(0) == 128 &&
              o.size(1) == LL && dg.size(0) == 128 && dg.size(1) == LL && g.stride(1) == 1 && o.stride(1) == 1 && dg.stride(1) == 1 && g.stride(0) % 8 == 0 &&
              o.stride(0) % 8 == 0 && dg.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dg.data_ptr()) % 16 == 0, "gate_bwd128: dy [LL, 128], wo [128, 128], planes [128, LL]");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto dout = torch::empty({128, LL}, dy.options());
  const int grid = (int)std::min<long>(LL / 64, (long)sms * gm);
  const size_t smem = (size_t)(128 * 136 + 2 * 9216) * 2;
  C10_CUDA_CHECK(cudaFuncSetAttribute(gate_bwd128_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  gate_bwd128_k<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(dy.data_ptr<at::BFloat16>()), reinterpret_cast<const bf16*>(wo.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(g.data_ptr<at::BFloat16>()), reinterpret_cast<const bf16*>(o.data_ptr<at::BFloat16>()), reinterpret_cast<bf16*>(dout.data_ptr<at::BFloat16>()),
      reinterpret_cast<bf16*>(dg.data_ptr<at::BFloat16>()), LL, g.stride(0), o.stride(0), dg.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dout;
}
'''
s = s.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n", host + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"gate_bwd128\", &gate_bwd128, \"B200 fused du GEMM + gate backward, K = 128\");\n", 1)
open(F, "w").write(s); print("ok")
