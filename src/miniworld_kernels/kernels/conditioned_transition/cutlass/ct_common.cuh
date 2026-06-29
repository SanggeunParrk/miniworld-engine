// Shared CUTLASS 3.x Hopper TF32 WGMMA GEMM helpers for the ConditionedTransition tail.
// All GEMMs are of the form D = A @ B^T with row-major fp32 buffers, tfloat32 operands,
// fp32 accumulate. B is stored (N,K) row-major; reading it ColumnMajor (K,N) yields B^T.
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace ct {
using namespace cute;

using ElementA   = float;
using ElementB   = float;
using ElementC   = float;
using ElementAcc = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;     // A:(M,K)
using LayoutB = cutlass::layout::ColumnMajor;  // B stored (N,K) row-major -> read as B^T (K,N)
using LayoutC = cutlass::layout::RowMajor;     // D:(M,N)

constexpr int Align = 128 / cutlass::sizeof_bits<ElementA>::value;  // 4 for fp32 (16B)

using ArchTag = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;
using TileShape    = Shape<_128, _128, _32>;
using ClusterShape = Shape<_1, _2, _1>;

} // namespace ct
