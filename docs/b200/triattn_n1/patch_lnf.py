import sys
F = sys.argv[1]; s = open(F).read()
kernel = '''
// y = LN(x) * w + b, C = 128, starting layout (the shared glue's ln_fwd_k<16>, same arithmetic order: y / mean / rstd bitwise equal): the next row
// step's load is issued before this one's reductions, w / b are registers, a persistent grid, streaming stores.
__global__ void __launch_bounds__(256) ln_fwd128_k(const bf16* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bs, bf16* __restrict__ y,
                                                   float* __restrict__ mean_o, float* __restrict__ rstd_o, long M, float eps) {
  const int lane = threadIdx.x & 31, sub = lane % LPR, grp = lane / LPR;
  float ww[8], bb[8];
  {
    const float4 w0 = *reinterpret_cast<const float4*>(w + sub * 8), w1 = *reinterpret_cast<const float4*>(w + sub * 8 + 4);
    const float4 b0 = *reinterpret_cast<const float4*>(bs + sub * 8), b1 = *reinterpret_cast<const float4*>(bs + sub * 8 + 4);
    ww[0] = w0.x; ww[1] = w0.y; ww[2] = w0.z; ww[3] = w0.w; ww[4] = w1.x; ww[5] = w1.y; ww[6] = w1.z; ww[7] = w1.w;
    bb[0] = b0.x; bb[1] = b0.y; bb[2] = b0.z; bb[3] = b0.w; bb[4] = b1.x; bb[5] = b1.y; bb[6] = b1.z; bb[7] = b1.w;
  }
  const long stride = (long)gridDim.x * 8 * RPW;
  long row0 = (((long)blockIdx.x * 256 + threadIdx.x) >> 5) * RPW;
  uint4 cur = make_uint4(0, 0, 0, 0), nxt;
  if (row0 + grp < M) cur = ldcs(x + (row0 + grp) * C + sub * 8);
  for (; row0 < M; row0 += stride) {
    nxt = make_uint4(0, 0, 0, 0);
    if (row0 + stride + grp < M) nxt = ldcs(x + (row0 + stride + grp) * C + sub * 8);
    const long row = row0 + grp;
    const bool ok = row < M;
    float v[8], s = 0.f;
    if (ok) {
      unpack8(cur, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) s += v[e];
    }
    const float mean = grp_sum(s) / C;
    float q = 0.f;
    if (ok) {
#pragma unroll
      for (int e = 0; e < 8; ++e) { const float d = v[e] - mean; q = fmaf(d, d, q); }
    }
    const float rstd = rsqrtf(grp_sum(q) / C + eps);
    if (ok) {
      float o[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = fmaf((v[e] - mean) * rstd, ww[e], bb[e]);
      __stcs(reinterpret_cast<uint4*>(y + row * C + sub * 8), pack8(o));
    }
    if (sub == 0 && ok) { mean_o[row] = mean; rstd_o[row] = rstd; }
    cur = nxt;
  }
}
'''
s = s.replace("}  // namespace\n\n// -> (dx bf16", kernel + "}  // namespace\n\n// -> (dx bf16", 1)
host = '''
// -> (y bf16 [M, 128], mean fp32 [M], rstd fp32 [M])
std::vector<torch::Tensor> ln_fwd128(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, int64_t gm) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == C && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "ln_fwd128: x [M, 128] bf16 contiguous");
  const long M = x.size(0);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = (int)std::min<long>((M + 8 * RPW - 1) / (8 * RPW), (long)sms * gm);
  auto y = torch::empty_like(x);
  auto mean = torch::empty({M}, x.options().dtype(torch::kFloat32)), rstd = torch::empty({M}, x.options().dtype(torch::kFloat32));
  ln_fwd128_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()), w.data_ptr<float>(), b.data_ptr<float>(),
      reinterpret_cast<bf16*>(y.data_ptr<at::BFloat16>()), mean.data_ptr<float>(), rstd.data_ptr<float>(), M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}
'''
s = s.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"ln_bwd128\", &ln_bwd128, \"B200 LayerNorm backward, C = 128\"); }",
              host + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"ln_bwd128\", &ln_bwd128, \"B200 LayerNorm backward, C = 128\");\n  m.def(\"ln_fwd128\", &ln_fwd128, \"B200 LayerNorm forward, C = 128\");\n}")
assert "ln_fwd128_k" in s and "m.def(\"ln_fwd128\"" in s
open(F, "w").write(s); print("ok")
