import sys
F = sys.argv[1]; s = open(F).read()
# template on TR: iterate over PHYSICAL rows for the ending orientation (x, dy, dx coalesced; dyln, mean, rstd gathered)
s = s.replace("template <typename TD>\n__device__ __forceinline__ void fetch(Raw<TD>& R,", "template <typename TD, bool TR>\n__device__ __forceinline__ void fetch(Raw<TD>& R,")
s = s.replace("""  const long row = row0 + grp;
  if (row < M) {
    R.mean = mean_i[row]; R.rstd = rstd_i[row];
    const long pr = phys_row(row, L, tr);""", """  const long pr = row0 + grp;                      // physical row (x, dy, dx)
  if (pr < M) {
    const long row = TR ? phys_row(pr, L, 1) : pr;  // the logical row (dyln, mean, rstd); the swap is its own inverse
    R.mean = mean_i[row]; R.rstd = rstd_i[row];""")
s = s.replace("template <typename TD>\n__global__ void __launch_bounds__(256) ln_bwd128_k(", "template <typename TD, bool TR>\n__global__ void __launch_bounds__(256) ln_bwd128_k(")
s = s.replace("fetch<TD>(cur, row0, grp, sub, dyln, x, dy, mean_i, rstd_i, M, L, tr);", "fetch<TD, TR>(cur, row0, grp, sub, dyln, x, dy, mean_i, rstd_i, M, L, tr);")
s = s.replace("fetch<TD>(nxt, row0 + stride, grp, sub, dyln, x, dy, mean_i, rstd_i, M, L, tr);", "fetch<TD, TR>(nxt, row0 + stride, grp, sub, dyln, x, dy, mean_i, rstd_i, M, L, tr);")
s = s.replace("      __stcs(reinterpret_cast<uint4*>(dx + phys_row(row, L, tr) * C + sub * 8), pack8(r));", "      __stcs(reinterpret_cast<uint4*>(dx + row * C + sub * 8), pack8(r));")
# launch: dispatch on trans
old_launch = s[s.index("  if (dyln.scalar_type() == torch::kFloat32)\n    ln_bwd128_k<float>"):s.index("  C10_CUDA_KERNEL_LAUNCH_CHECK();")]
new_launch = """  auto go = [&](auto tdp, auto trc) {
    using TD = std::remove_cv_t<std::remove_pointer_t<decltype(tdp)>>;
    constexpr bool TR = decltype(trc)::value;
    ln_bwd128_k<TD, TR><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const TD*>(dyln.data_ptr()), xb, dyb, w.data_ptr<float>(), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), dxb, part.data_ptr<float>(), M, (int)L, (int)trans);
  };
  if (dyln.scalar_type() == torch::kFloat32) { if (trans) go((float*)nullptr, std::true_type{}); else go((float*)nullptr, std::false_type{}); }
  else { if (trans) go((bf16*)nullptr, std::true_type{}); else go((bf16*)nullptr, std::false_type{}); }
"""
s = s.replace(old_launch, new_launch)
open(F, "w").write(s); print("ok")
