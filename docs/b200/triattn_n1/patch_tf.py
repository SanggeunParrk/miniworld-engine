import sys
F = sys.argv[1]; s = open(F).read()
s = s.replace("template <typename TD, bool TR>\n__global__ void __launch_bounds__(256) ln_bwd128_k(", "template <typename TD, bool TR, bool FOLD>\n__global__ void __launch_bounds__(256) ln_bwd128_k(")
s = s.replace("bf16* __restrict__ dx, float* __restrict__ part, float* __restrict__ dgb, long M, int L, int tr, int fold) {", "bf16* __restrict__ dx, float* __restrict__ part, float* __restrict__ dgb, long M, int L, int tr) {")
s = s.replace("  __shared__ float red[8 * RPW * 2 * C];\n  __shared__ unsigned last;", "  __shared__ float red[8 * RPW * 2 * C];")
a = s.index("  if (!fold) return;")
b = s.index("}\n", s.index("dgb[c] = (s0 + s1) + (s2 + s3);"))   # end of for loop braces ... find kernel end below
# locate the end of the kernel: the text after the fold block is "    }\n  }\n}\n"
tail_old = s[a:]
k = tail_old.index("      dgb[c] = (s0 + s1) + (s2 + s3);\n    }\n  }\n}\n") + len("      dgb[c] = (s0 + s1) + (s2 + s3);\n    }\n  }\n}\n")
block = tail_old[:k]
new_block = block.replace("  if (!fold) return;   // (the caller sums the partial rows: bo_colsum -- measured 2-4 us faster than this fold at L >= 256)\n", "  if constexpr (FOLD) {   // (off by default: the caller's bo_colsum is 2-4 us faster than this fold at L >= 256)\n  __shared__ unsigned last;\n")
new_block = new_block[:-2] + "  }\n}\n"
s = s[:a] + new_block + tail_old[k:]
s = s.replace("ln_bwd128_k<TD, TR><<<", "ln_bwd128_k<TD, TR, FOLD><<<")
s = s.replace("(int)L, (int)trans, (int)fold);\n  };", "(int)L, (int)trans);\n  };")
s = s.replace("  auto go = [&](auto tdp, auto trc) {\n    using TD = std::remove_cv_t<std::remove_pointer_t<decltype(tdp)>>;\n    constexpr bool TR = decltype(trc)::value;",
              "  auto go = [&](auto tdp, auto trc) {\n    using TD = std::remove_cv_t<std::remove_pointer_t<decltype(tdp)>>;\n    constexpr bool TR = decltype(trc)::value;\n    constexpr bool FOLD = false;   // (the fold variant is built by changing this: measured slower)")
open(F, "w").write(s); print("ok")
