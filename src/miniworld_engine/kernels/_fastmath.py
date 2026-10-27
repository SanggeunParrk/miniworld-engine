"""Cheap sigmoid for the SwiGLU gates of the bf16 Triton kernels.

``tl.sigmoid`` is ``1 / (1 + exp(-x))``: two multi-function-unit operations per element (``ex2`` and ``rcp``). The gate epilogue of the
Transition expand / backward kernels runs on the CUDA cores between the tensor-core loops of a tile, so those operations are not hidden
behind the MMAs: on H100 (bf16, D768) the sigmoid is ~0.3 ms of a 3.0 ms expand kernel. ``0.5 + 0.5 * tanh(x / 2)`` is the same function
with ONE operation (``tanh.approx.f32``, a PTX instruction every card from sm_75 on has, which is why it is written as inline PTX rather
than ``libdevice.tanh``: the libdevice routine is the accurate software version and is slower than ``tl.sigmoid``). Its error is about
2.5e-4 absolute, below the half-ulp of a bf16 result (2e-3 relative to values near 0.5), so it is used only where the
gate output is rounded to bf16; the fp32 (and fp16) paths keep ``tl.sigmoid``.
"""
import triton
import triton.language as tl


@triton.jit
def _tanh_approx(x):
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def sigmoid_fast(x):
    """``sigmoid(x)`` through one ``tanh.approx.f32``; ``x`` fp32."""
    return 0.5 + 0.5 * _tanh_approx(0.5 * x)
