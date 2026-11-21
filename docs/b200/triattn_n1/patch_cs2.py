import sys
F = sys.argv[1]; s = open(F).read()
# 1. kernel: ticket + final column sum
s = s.replace("template <typename TD> struct Raw { uint4 x, y, d0, d1; float mean, rstd; };",
              "template <typename TD> struct Raw { uint4 x, y, d0, d1; float mean, rstd; };\n__device__ unsigned g_ln_ticket;   // arrivals of the last-CTA column sum (self-resetting; zero at load)")
s = s.replace("const float* __restrict__ w, const float* __restrict__ mean_i, const float* __restrict__ rstd_i,\n                                                   bf16* __restrict__ dx, float* __restrict__ part, long M, int L, int tr) {\n  __shared__ float red[8 * RPW * 2 * C];",
              "const float* __restrict__ w, const float* __restrict__ mean_i, const float* __restrict__ rstd_i,\n                                                   bf16* __restrict__ dx, float* __restrict__ part, float* __restrict__ dgb, long M, int L, int tr) {\n  __shared__ float red[8 * RPW * 2 * C];\n  __shared__ unsigned last;")
old_tail = "    part[((long)blockIdx.x * 2 + k) * C + cc] = s;\n  }\n}\n"
new_tail = """    part[((long)blockIdx.x * 2 + k) * C + cc] = s;
  }
  // the last CTA to arrive sums the partial rows in a fixed order (row 0 .. grid - 1): d gamma / d beta without a second launch
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned t = atomicAdd(&g_ln_ticket, 1u);
    last = (t == gridDim.x - 1) ? 1u : 0u;
    if (last) g_ln_ticket = 0u;
  }
  __syncthreads();
  if (last) {
    __threadfence();
    for (int c = threadIdx.x; c < 2 * C; c += 256) {
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      int r = 0;
      for (; r + 4 <= (int)gridDim.x; r += 4) {
        s0 += __ldcg(part + ((long)(r) * 2) * C + c); s1 += __ldcg(part + ((long)(r + 1) * 2) * C + c);
        s2 += __ldcg(part + ((long)(r + 2) * 2) * C + c); s3 += __ldcg(part + ((long)(r + 3) * 2) * C + c);
      }
      for (; r < (int)gridDim.x; ++r) s0 += __ldcg(part + ((long)r * 2) * C + c);
      dgb[c] = (s0 + s1) + (s2 + s3);
    }
  }
}
}\n"""
assert old_tail in s
s = s.replace(old_tail, new_tail, 1)
# 2. launch: pass dgb; return {dx, dgb}
s = s.replace("rstd.data_ptr<float>(), dxb, part.data_ptr<float>(), M, (int)L, (int)trans);\n  };", "rstd.data_ptr<float>(), dxb, part.data_ptr<float>(), dgb.data_ptr<float>(), M, (int)L, (int)trans);\n  };", 1)
s = s.replace("  auto part = torch::empty({grid, 2, C}, x.options().dtype(torch::kFloat32));\n  auto xb = reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()); auto dyb",
              "  auto part = torch::empty({grid, 2, C}, x.options().dtype(torch::kFloat32));\n  auto dgb = torch::empty({2, C}, x.options().dtype(torch::kFloat32));\n  auto xb = reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()); auto dyb", 1)
s = s.replace("// -> (dx bf16 [M, 128], partial sums fp32 [grid, 2, 128]); gm CTAs an SM", "// -> (dx bf16 [M, 128], d gamma | d beta fp32 [2, 128] (the partial rows summed in order by the last CTA)); gm CTAs an SM")
i = s.index("ln_bwd128(torch::Tensor dyln")
j = s.index("return {dx, part};", i)
s = s[:j] + "return {dx, dgb};" + s[j + len("return {dx, part};"):]
open(F, "w").write(s); print("ok")
