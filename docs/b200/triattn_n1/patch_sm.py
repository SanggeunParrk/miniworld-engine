import sys
F = sys.argv[1]; s = open(F).read()
kern = r'''
// ---- logits + softmax + bf16 cast of the bias planes in ONE kernel (the shared chain: `logits_k`, torch's `softmax_warp_forward` and a copy kernel: three
// launches of 1-2 us each at L = 128 .. 256). Planes r < nstart as they are (softmax over the row), the others transposed ([h, i, k] = bias[h, k, i]);
// masked keys -> -inf. The arithmetic is torch's: a thread's elements lane + 32 it, exp(x - max) summed in it order, the xor butterfly 16 .. 1, then
// x / sum (accurate exp and division), rounded to bf16: bitwise equal to torch.softmax(logits, -1).to(bf16).
extern "C" __device__ float __nv_expf(float);
__device__ __forceinline__ void softmax_row(float (&v)[32], int iters, int L, int lane, bf16* __restrict__ dst) {
  float mx = -INFINITY;
#pragma unroll
  for (int it = 0; it < 32; ++it) if (it < iters) mx = fmaxf(mx, v[it]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.f;
#pragma unroll
  for (int it = 0; it < 32; ++it) if (it < iters) { v[it] = __nv_expf(v[it] - mx); sum += v[it]; }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
#pragma unroll
  for (int it = 0; it < 32; ++it) {
    const int k = lane + 32 * it;
    if (it < iters && k < L) dst[k] = __float2bfloat16_rn(__fdiv_rn(v[it], sum));
  }
}
__global__ void __launch_bounds__(256) softmax_planes_k(const bf16* __restrict__ b, const bool* __restrict__ keep, bf16* __restrict__ p, int nstart, int L, long ld) {
  extern __shared__ float tile[];   // ending planes: [L][33]
  const int r = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5, iters = (L + 31) >> 5;
  const bf16* src = b + (long)r * ld;
  bf16* dst = p + (long)r * L * L;
  float v[32];
  if (r < nstart) {
    const int row = blockIdx.x * 8 + w;
    if (row >= L) return;
#pragma unroll
    for (int it = 0; it < 32; ++it) {
      const int k = lane + 32 * it;
      v[it] = (it < iters && k < L && (keep == nullptr || keep[k])) ? __bfloat162float(src[(long)row * L + k]) : -INFINITY;
    }
    softmax_row(v, iters, L, lane, dst + (long)row * L);
  } else {
    const int i0 = blockIdx.x * 32;
    if (i0 >= L) return;
    for (int k = w; k < L; k += 8) tile[k * 33 + lane] = (i0 + lane < L) ? __bfloat162float(src[(long)k * L + i0 + lane]) : 0.f;
    __syncthreads();
    for (int ri = w; ri < 32; ri += 8) {
      const int row = i0 + ri;
      if (row >= L) break;
#pragma unroll
      for (int it = 0; it < 32; ++it) {
        const int k = lane + 32 * it;
        v[it] = (it < iters && k < L && (keep == nullptr || keep[k])) ? tile[k * 33 + ri] : -INFINITY;
      }
      softmax_row(v, iters, L, lane, dst + (long)row * L);
    }
  }
}
'''
s = s.replace("}  // namespace\n\n// -> (dx bf16", kern + "}  // namespace\n\n// -> (dx bf16", 1)
host = '''
// bias [R, L L] plane rows (bf16, unit column stride): the first nstart planes starting, the rest ending -> p [R, L, L] bf16 (softmax over the last axis); keep: optional bool [L]
torch::Tensor softmax_planes(torch::Tensor bias, c10::optional<torch::Tensor> keep, int64_t nstart, int64_t L) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kBFloat16 && bias.dim() == 2 && bias.size(1) == L * L && nstart >= 0 && nstart <= bias.size(0) && bias.stride(1) == 1 && L <= 1024,
              "softmax_planes: bf16 [R, L L] planes, L <= 1024");
  const bool* kp = nullptr;
  if (keep.has_value()) {
    TORCH_CHECK(keep->is_cuda() && keep->scalar_type() == torch::kBool && keep->is_contiguous() && keep->numel() == L, "softmax_planes: bool [L] mask");
    kp = keep->data_ptr<bool>();
  }
  const long R = bias.size(0);
  auto out = torch::empty({R, L, L}, bias.options());
  const size_t smem = nstart < R ? (size_t)L * 33 * 4 : 0;
  if (smem > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(softmax_planes_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  const dim3 grid((unsigned)((L + 7) / 8), (unsigned)R);
  softmax_planes_k<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(bias.data_ptr<at::BFloat16>()), kp, reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
      (int)nstart, (int)L, bias.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
'''
s = s.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n", host + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"softmax_planes\", &softmax_planes, \"B200 logits + softmax + bf16 cast of the bias planes\");\n", 1)
open(F, "w").write(s); print("ok")
