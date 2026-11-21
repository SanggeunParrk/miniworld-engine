import sys
F = sys.argv[1]; s = open(F).read()
kern = '''
// d gamma | d beta = the partial rows [n, 2, 128] summed in row order, ONE launch (the shared `colsum` of bidir_sm100.cu is two): 8 CTAs of 32 columns,
// each thread one column over every 8th row, the 8 row groups added in a fixed order.
__global__ void __launch_bounds__(256) colsum256_k(const float* __restrict__ part, float* __restrict__ out, int n) {
  __shared__ float red[8][32];
  const int c = blockIdx.x * 32 + (threadIdx.x & 31), rg = threadIdx.x >> 5;
  float s = 0.f;
  for (int r = rg; r < n; r += 8) s += __ldcg(part + (long)r * 256 + c);
  red[rg][threadIdx.x & 31] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = 0.f;
#pragma unroll
    for (int g = 0; g < 8; ++g) t += red[g][threadIdx.x];
    out[blockIdx.x * 32 + threadIdx.x] = t;
  }
}
'''
s = s.replace("}  // namespace\n\n// -> (dx bf16", kern + "}  // namespace\n\n// -> (dx bf16", 1)
host = '''
// part [n, 2, 128] fp32 -> [2, 128] fp32, the rows summed in order
torch::Tensor colsum256(torch::Tensor part) {
  TORCH_CHECK(part.is_cuda() && part.scalar_type() == torch::kFloat32 && part.is_contiguous() && part.dim() == 3 && part.size(1) == 2 && part.size(2) == C, "colsum256: fp32 [n, 2, 128]");
  auto out = torch::empty({2, C}, part.options());
  colsum256_k<<<8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(), out.data_ptr<float>(), (int)part.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
'''
s = s.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n", host + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"colsum256\", &colsum256, \"B200 column sum of the LayerNorm backward's partial rows\");\n", 1)
open(F, "w").write(s); print("ok")
