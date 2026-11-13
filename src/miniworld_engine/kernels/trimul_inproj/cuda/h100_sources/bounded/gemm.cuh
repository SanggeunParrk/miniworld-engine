// SPDX-License-Identifier: Apache-2.0
// The TriMul contraction on H100 bounded by the live extents of the pair mask, read on the device: a CUTLASS 3 sm_90 TMA warp-specialized
// persistent batched GEMM (one plane per batch index) whose tile scheduler (bounded_scheduler.hpp) gives each output tile a k-tile count
// from the live extents of its plane's sample: one k tile for a tile past the live rows / columns (its operand rows are zero: K1 masks
// a / b by m[i] & m[k], so it writes the zeros the full product would), ceil(live k / TK) otherwise (the k tiles past it hold zero
// products). The output is the full-K product of the same kernel bit for bit; the launch is shape-static (CUDA graphs: the extents are
// read by every replay).
//   mode 0 (outgoing): t[i, j] = sum_k a[i, k] b[j, k];  mode 1 (incoming): t[i, j] = sum_k a[k, i] b[k, j]
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "bounded_scheduler.hpp"
#include "bounded_pingpong.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

namespace mwe_bounded {
template <class S>
S plane_stride(int64_t ld, int64_t plane) {
  S s{};
  if constexpr (cute::is_static<cute::remove_cvref_t<decltype(cute::get<0>(s))>>::value)
    cute::get<1>(s) = ld;
  else
    cute::get<0>(s) = ld;
  cute::get<2>(s) = plane;
  return s;
}

template <int MODE, class Tile, class Cluster, bool PINGPONG>
struct BGemm {
  using LA = std::conditional_t<MODE == 0, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using LB = std::conditional_t<MODE == 0, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using LD = cutlass::layout::RowMajor;
  using bf = cutlass::bfloat16_t;
  using KSched = std::conditional_t<PINGPONG, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  using ESched = std::conditional_t<PINGPONG, cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, LD, 8, bf, LD, 8, ESched>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, bf, LA, 8, bf, LB, 8, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>, KSched>::CollectiveOp;
  using Shape4 = cute::Shape<int, int, int, int>;
  // cooperative: the stock kernel (its loops take the scheduler's per-tile counts); pingpong: the fork (its stock loops use one count)
  using Kernel = std::conditional_t<PINGPONG, cutlass::gemm::kernel::MweGemmPingpongBounded<Shape4, Main, Epi, BoundedScheduler>,
                                    cutlass::gemm::kernel::GemmUniversal<Shape4, Main, Epi, BoundedScheduler>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static void run(const void* a, const void* b, void* d, int n, int planes, const int* bounds, int nsamples, int swizzle, int raster,
                  int device, cudaStream_t stream) {
    using SA = typename Kernel::StrideA;
    using SB = typename Kernel::StrideB;
    using SD = typename Kernel::StrideD;
    const int64_t ps = (int64_t)n * n;
    cutlass::KernelHardwareInfo hw;
    hw.device_id = device;
    static int sms[64] = {0};
    if (!sms[device]) sms[device] = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    hw.sm_count = sms[device];
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {n, n, n, planes},
                                  {(const bf*)a, plane_stride<SA>(n, ps), (const bf*)b, plane_stride<SB>(n, ps)},
                                  {{1.f, 0.f}, nullptr, plane_stride<SD>(n, ps), (bf*)d, plane_stride<SD>(n, ps)},
                                  hw};
    args.scheduler.bounds = bounds;
    args.scheduler.nsamples = nsamples;
    args.scheduler.max_swizzle_size = 1;  // the problem tiles unrounded; the unit order's own strips instead
    args.scheduler.group = swizzle;
    using RO = typename Kernel::TileScheduler::RasterOrderOptions;
    args.scheduler.raster_order = raster == 0 ? RO::Heuristic : raster == 1 ? RO::AlongM : RO::AlongN;
    Gemm g;
    TORCH_CHECK(g.can_implement(args) == cutlass::Status::kSuccess, "bounded contraction: CUTLASS cannot implement n ", n);
    TORCH_CHECK(Gemm::get_workspace_size(args) == 0, "bounded contraction: a workspace was asked for");
    TORCH_CHECK(g.initialize(args, nullptr, stream) == cutlass::Status::kSuccess, "bounded contraction: initialize");
    TORCH_CHECK(g.run(stream) == cutlass::Status::kSuccess, "bounded contraction: launch");
  }
};
}  // namespace mwe_bounded

#define MWE_BOUNDED_CONFIG(NAME, TILE, CLUSTER, PINGPONG)                                                                               \
  void NAME(const void* a, const void* b, void* d, int n, int planes, int mode, const int* bounds, int nsamples, int swizzle, int raster, \
            int device, cudaStream_t stream) {                                                                                         \
    using namespace cute;                                                                                                               \
    if (mode == 0)                                                                                                                      \
      mwe_bounded::BGemm<0, TILE, CLUSTER, PINGPONG>::run(a, b, d, n, planes, bounds, nsamples, swizzle, raster, device, stream);       \
    else                                                                                                                                \
      mwe_bounded::BGemm<1, TILE, CLUSTER, PINGPONG>::run(a, b, d, n, planes, bounds, nsamples, swizzle, raster, device, stream);       \
  }
