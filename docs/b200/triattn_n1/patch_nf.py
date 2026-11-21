import sys
F = sys.argv[1]; s = open(F).read()
s = s.replace("bf16* __restrict__ dx, float* __restrict__ part, float* __restrict__ dgb, long M, int L, int tr) {", "bf16* __restrict__ dx, float* __restrict__ part, float* __restrict__ dgb, long M, int L, int tr, int fold) {")
s = s.replace("  // the last CTA to arrive sums the partial rows in a fixed order (row 0 .. grid - 1): d gamma / d beta without a second launch\n  __threadfence();",
              "  if (!fold) return;   // (the caller sums the partial rows: bo_colsum -- measured 2-4 us faster than this fold at L >= 256)\n  // fold: the last CTA to arrive sums the partial rows in a fixed order (row 0 .. grid - 1): d gamma / d beta without a second launch\n  __threadfence();")
s = s.replace("(int)L, (int)trans);\n  };\n  if (dyln.scalar_type() == torch::kFloat32) { if (trans)", "(int)L, (int)trans, (int)fold);\n  };\n  if (dyln.scalar_type() == torch::kFloat32) { if (trans)")
s = s.replace("bool trans, int64_t gm) {\n  const long M = x.size(0);\n  TORCH_CHECK(x.size(1) == C && x.is_contiguous() && dy.is_contiguous()", "bool trans, int64_t gm, bool fold) {\n  const long M = x.size(0);\n  TORCH_CHECK(x.size(1) == C && x.is_contiguous() && dy.is_contiguous()")
s = s.replace("  return {dx, dgb};\n}\n\n// -> (y bf16", "  if (fold) return {dx, dgb};\n  return {dx, part};\n}\n\n// -> (y bf16")
s = s.replace("// -> (dx bf16 [M, 128], d gamma | d beta fp32 [2, 128] (the partial rows summed in order by the last CTA)); gm CTAs an SM", "// -> (dx bf16 [M, 128], partial sums fp32 [grid, 2, 128]) or, fold, (dx, d gamma | d beta fp32 [2, 128] summed by the last CTA); gm CTAs an SM")
assert "int fold) {" in s and "bool fold) {" in s and "if (fold) return" in s and "(int)fold);" in s and "if (!fold) return;" in s
open(F, "w").write(s); print("ok")
