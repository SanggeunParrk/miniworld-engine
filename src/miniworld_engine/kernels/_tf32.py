"""Round-to-nearest TF32 operands for the portable (Triton) fp32 kernels.

``tl.dot(a, b, input_precision="tf32")`` hands the raw fp32 bits to the tensor core, which drops the low 13 mantissa bits: it TRUNCATES
(rounds toward zero). cuBLAS under ``allow_tf32`` rounds each operand to nearest. Truncation is biased (every operand shrinks by 2^-11 on
average), so a kernel built on it measured 3-6x the error of the PyTorch TF32 statements against an fp64 run (Transition, ConditionedTransition,
AdaLN, the attention cores), where one rounded to nearest first measures 1.0-1.1x.

``tf32_round`` is the rounding (add half an ulp of the TF32 mantissa to the bit pattern, clear the low 13 bits: round half away from zero,
exactly what ``cvt.rna.tf32.f32`` does, in two full-rate integer instructions instead of a quarter-rate convert; +-inf and NaN keep their
class). ``tf32_dot`` is ``tl.dot`` with that applied to fp32 operands when the precision is ``"tf32"``; bf16 / fp16 operands and every other
precision compile to the plain ``tl.dot`` (nothing is added to those kernels). It is the ONE place the kernels spell the rounding.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def tf32_round(x):
    """``x`` (fp32) rounded to the nearest TF32 value (10 mantissa bits), as an fp32 tensor; other dtypes are returned unchanged."""
    if x.dtype == tl.float32:
        x = ((x.to(tl.int32, bitcast=True) + 0x1000) & -0x2000).to(tl.float32, bitcast=True)
    return x


@triton.jit
def tf32_dot(a, b, acc=None, input_precision: tl.constexpr = "tf32", out_dtype: tl.constexpr = tl.float32):
    """``tl.dot(a, b, acc)`` whose fp32 operands are rounded to the TF32 grid first when ``input_precision`` is ``"tf32"``."""
    if input_precision == "tf32":
        a = tf32_round(a)
        b = tf32_round(b)
    return tl.dot(a, b, acc, input_precision=input_precision, out_dtype=out_dtype)
