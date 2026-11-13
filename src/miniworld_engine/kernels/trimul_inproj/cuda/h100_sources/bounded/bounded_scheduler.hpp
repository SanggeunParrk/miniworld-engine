// SPDX-License-Identifier: BSD-3-Clause (derived from CUTLASS 4.2, Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES)
// A persistent sm_90 tile scheduler whose work units carry a k-tile count bounded by the live extents of their batch's sample, read on
// the device (forked from CUTLASS 4.2 static_tile_scheduler.hpp / sm90_tile_scheduler.hpp; changes marked "bounded").
//
// bounds: int32 [nsamples, 3] (live rows, live columns, live k) on the device; batch l belongs to sample l % nsamples. A cluster tile past
// the live rows or columns runs one k tile (its operand rows there are zero by the caller's contract, so the product is the dense one's
// zeros); every other tile runs ceil(live k / TK) k tiles; the live tiles are scheduled first (BoundedPersistentSm90's unit order). The k tiles past live k hold zero products, so the output
// is the full-K product of the same kernel, bit for bit. Every CTA of a cluster gets the same count (TMA multicast).
#pragma once
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"

namespace mwe_bounded {
struct BoundedScheduler {};

using namespace cute;
using cutlass::FastDivmodU64; using cutlass::FastDivmodU64Pow2; using cutlass::KernelHardwareInfo; using cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params;
using cutlass::CudaHostAdapter; using cutlass::Status;
using cutlass::gemm::to_gemm_coord;

struct BoundedParams : PersistentTileSchedulerSm90Params {
  const int* bounds = nullptr;   // bounded
  int nsamples = 1, tile_m = 1, tile_n = 1, tile_k = 1, k_full = 0, group = 1;
};

struct BoundedArguments {
  int max_swizzle_size = 1;
  PersistentTileSchedulerSm90Params::RasterOrderOptions raster_order = PersistentTileSchedulerSm90Params::RasterOrderOptions::Heuristic;
  const int* bounds = nullptr;  // bounded
  int nsamples = 1;
  int group = 1;  // bounded: live units in strips of this many along the raster's minor mode (CUTLASS's swizzle; its own stays 1)
};

class BoundedPersistentSm90 {
 private:
  uint64_t current_work_linear_idx_;
  uint64_t total_grid_size_;

 public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;
    int32_t k_tiles = 0;  // bounded
    CUTLASS_HOST_DEVICE bool is_valid() const { return is_valid_tile; }
    CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() { return {-1, -1, -1, false, 0}; }
    CUTLASS_HOST_DEVICE bool is_final_split(uint32_t) const { return true; }
    CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const { return -1; }
  };

  using Params = BoundedParams;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  static constexpr bool IsDynamicPersistent = false;
  using Pipeline = cutlass::PipelineEmpty;
  using PipelineStorage = typename Pipeline::SharedStorage;
  using ThrottlePipeline = cutlass::PipelineEmpty;
  using ThrottlePipelineStorage = typename ThrottlePipeline::SharedStorage;
  struct CLCResponse {};
  class SharedStorage {
   public:
    CUTLASS_DEVICE PipelineStorage pipeline() { return PipelineStorage{}; }
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() { return ThrottlePipelineStorage{}; }
    CUTLASS_DEVICE CLCResponse* data() { return nullptr; }
  };

  using Arguments = BoundedArguments;

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params to_underlying_arguments(ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape, ClusterShape cluster_shape,
                                        [[maybe_unused]] KernelHardwareInfo const& hw_info, Arguments const& arguments,
                                        [[maybe_unused]] void* workspace = nullptr, [[maybe_unused]] const uint32_t epilogue_subtile = 1,
                                        [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    Params params;
    params.initialize(problem_blocks, to_gemm_coord(cluster_shape), hw_info, arguments.max_swizzle_size, arguments.raster_order);
    params.bounds = arguments.bounds;  // bounded
    params.nsamples = arguments.nsamples;
    params.group = arguments.group < 1 ? 1 : arguments.group;
    params.tile_m = size<0>(tile_shape);
    params.tile_n = size<1>(tile_shape);
    params.tile_k = size<2>(tile_shape);
    params.k_full = static_cast<int>(get<2>(problem_shape_mnkl));
    return params;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const& args) { return args.max_swizzle_size >= 0 && args.nsamples >= 1; }
  CUTLASS_HOST_DEVICE BoundedPersistentSm90() {}
  // bounded: the work is enumerated in cluster units (one unit = one cluster's tiles), live units first. Unit order: for each sample its
  // live units (the clusters inside its live rows x columns, plane by plane), then for each sample its dead units. A cluster takes units
  // cluster_id, cluster_id + clusters, ... -- the live tiles spread evenly over the clusters whatever the live extents (a fixed tile order
  // with dead tiles in place left some clusters with more live tiles than others). Without bounds every unit is live (plane-major).
  CUTLASS_DEVICE explicit BoundedPersistentSm90(Params const& params_) : scheduler_params(params_) {
#if defined(__CUDA_ARCH__)
    const dim3 cid = cute::cluster_id_in_grid(), cgrid = cute::cluster_grid_dims();
    current_work_linear_idx_ = uint64_t(cid.x) + uint64_t(cgrid.x) * (uint64_t(cid.y) + uint64_t(cgrid.y) * uint64_t(cid.z));
    total_grid_size_ = uint64_t(cgrid.x) * uint64_t(cgrid.y) * uint64_t(cgrid.z);
#endif
  }

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) { return get_current_work(); }
  CUTLASS_DEVICE WorkTileInfo get_current_work() const { return get_current_work_for_linear_idx(current_work_linear_idx_); }

  // bounded: (live cluster rows, live cluster columns, k tiles) of sample s
  CUTLASS_DEVICE void live_units(int s, int& lm, int& ln, int& kt) const {
    const Params& p = scheduler_params;
    const int cm = int(p.problem_tiles_m_), cn = int(p.problem_tiles_n_);
    kt = (p.k_full + p.tile_k - 1) / p.tile_k;
    if (p.bounds == nullptr) {
      lm = cm;
      ln = cn;
      return;
    }
    const int* b = p.bounds + 3 * s;
    const int live_m = __ldg(b), live_n = __ldg(b + 1), live_k = __ldg(b + 2);
    const int um = p.tile_m * int(p.cluster_shape_m_), un = p.tile_n * int(p.cluster_shape_n_);
    lm = min(cm, (max(live_m, 0) + um - 1) / um);
    ln = min(cn, (max(live_n, 0) + un - 1) / un);
    const int k = (min(max(live_k, 0), p.k_full) + p.tile_k - 1) / p.tile_k;
    kt = k < 1 ? 1 : k;
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work_for_linear_idx(uint64_t unit) const {
    const Params& p = scheduler_params;
    const int cm = int(p.problem_tiles_m_), cn = int(p.problem_tiles_n_), planes = int(p.problem_tiles_l_), ns = p.nsamples;
    const int per_sample = planes / ns;  // planes l = c * ns + s: per_sample planes of each sample
    if (unit >= uint64_t(planes) * uint64_t(cm) * uint64_t(cn)) {
      return WorkTileInfo::invalid_work_tile();
    }
    int64_t q = int64_t(unit);
    int um = 0, un = 0, l = 0, kt = 1;
    bool found = false;
    for (int s = 0; s < ns && !found; ++s) {  // live units
      int lm, ln, k;
      live_units(s, lm, ln, k);
      const int64_t count = int64_t(per_sample) * lm * ln;
      if (q < count) {
        const int64_t per_plane = int64_t(lm) * ln;
        const int c = int(q / per_plane), r = int(q % per_plane);
        l = c * ns + s;
        // strips of `group` units across the minor mode, the major mode fastest inside a strip (AlongN: n major, m minor)
        const bool along_n = p.raster_order_ == RasterOrder::AlongN;
        const int major = along_n ? ln : lm, minor = along_n ? lm : ln, g = min(p.group, minor);
        const int full = (minor / g) * g * major;
        int a, bmin;
        if (r < full) {
          const int strip = r / (g * major), w = r % (g * major);
          a = w / g;
          bmin = strip * g + w % g;
        } else {
          const int rem = minor - (minor / g) * g, w = r - full;
          a = w / rem;
          bmin = (minor / g) * g + w % rem;
        }
        um = along_n ? bmin : a;
        un = along_n ? a : bmin;
        kt = k;
        found = true;
      } else {
        q -= count;
      }
    }
    for (int s = 0; s < ns && !found; ++s) {  // dead units: one k tile each (zeros)
      int lm, ln, k;
      live_units(s, lm, ln, k);
      const int64_t per_plane = int64_t(cm) * cn - int64_t(lm) * ln;
      const int64_t count = int64_t(per_sample) * per_plane;
      if (q < count) {
        const int c = int(q / per_plane);
        const int64_t r = q % per_plane;
        l = c * ns + s;
        const int64_t below = int64_t(cn - ln) * cm;  // the cluster columns past the live ones, every row
        if (r < below) {
          un = ln + int(r / cm);
          um = int(r % cm);
        } else {
          const int64_t r2 = r - below;
          un = int(r2 / (cm - lm));
          um = lm + int(r2 % (cm - lm));
        }
        kt = 1;
        found = true;
      } else {
        q -= count;
      }
    }
    if (!found) {
      return WorkTileInfo::invalid_work_tile();
    }
    auto [cta_m, cta_n, cta_l] = cute::block_id_in_cluster();
    const int32_t m = um * int(p.cluster_shape_m_) + int(cta_m), n = un * int(p.cluster_shape_n_) + int(cta_n);
    return {m, n, l, true, kt};
  }

  // bounded: the k-tile count of the work unit advance_count steps ahead (0 past the end)
  CUTLASS_DEVICE int peek_k_tiles(uint32_t advance_count) const {
    auto w = get_current_work_for_linear_idx(current_work_linear_idx_ + total_grid_size_ * uint64_t(advance_count));
    return w.is_valid() ? w.k_tiles : 0;
  }

  CUTLASS_DEVICE void advance_to_next_work(uint32_t advance_count = 1) { current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count); }

  CUTLASS_DEVICE bool is_last_tile(WorkTileInfo& work_tile_info, uint32_t advance_count = 1) const {
    return not get_current_work_for_linear_idx(current_work_linear_idx_ + (total_grid_size_ * uint64_t(advance_count))).is_valid();
  }
  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {
    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();
    return get_work_idx_m_and_n(
      blk_per_grid_dim,
      divmod_cluster_shape_major,
      divmod_cluster_shape_minor,
      divmod_cluster_blk_major,
      log_swizzle_size,
      raster_order,
      cta_m_in_cluster,
      cta_n_in_cluster
    );
  }

  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order,
      uint64_t cta_m_in_cluster,
      uint64_t cta_n_in_cluster) {

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor +
                                               cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor +
                                               cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    }
    else {
      return {major_work_idx, minor_work_idx};
    }

  }

  // Computes the linear index within a batch given M and N tile offsets within the batch.
  // This essentially inverts the mapping performed in get_work_idx_m_and_n
  static CUTLASS_DEVICE
  uint64_t
  get_linear_idx_from_m_and_n(
    int32_t tile_m,
    int32_t tile_n,
    FastDivmodU64Pow2 const& divmod_cluster_shape_major,
    FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
    FastDivmodU64 const& divmod_cluster_blk_major,
    int32_t log_swizzle_size,
    RasterOrder raster_order) {

    uint64_t minor_work_idx, major_work_idx, cluster_minor_offset;
    if (raster_order == RasterOrder::AlongN) {
      minor_work_idx = static_cast<uint64_t>(tile_m);
      major_work_idx = static_cast<uint64_t>(tile_n);
      uint64_t cluster_m = divmod_cluster_shape_minor.divide(tile_m) * divmod_cluster_shape_minor.divisor;
      cluster_minor_offset = tile_m - cluster_m;
    }
    else {
      major_work_idx = static_cast<uint64_t>(tile_m);
      minor_work_idx = static_cast<uint64_t>(tile_n);
      uint64_t cluster_n = divmod_cluster_shape_minor.divide(tile_n) * divmod_cluster_shape_minor.divisor;
      cluster_minor_offset = tile_n - cluster_n;
    }

    uint64_t cluster_idx_minor, cluster_idx_major, cluster_major_offset;
    cluster_idx_minor = divmod_cluster_shape_minor.divide(minor_work_idx - cluster_minor_offset);
    divmod_cluster_shape_major(cluster_idx_major, cluster_major_offset, major_work_idx);

    uint64_t cluster_idx_minor_div_swizzle = cluster_idx_minor >> log_swizzle_size;
    uint64_t offset = cluster_idx_minor & ((1 << log_swizzle_size) - 1);

    uint64_t extra = cluster_idx_minor_div_swizzle * divmod_cluster_blk_major.divisor + cluster_idx_major;

    uint64_t cluster_id = (extra << log_swizzle_size) | offset;
    return (cluster_id * divmod_cluster_shape_major.divisor + cluster_major_offset) * divmod_cluster_shape_minor.divisor + cluster_minor_offset;
  }

  // Given the inputs, computes the total number of output blocks over which this problem will compute. 
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape cta_shape, ClusterShape cluster_shape) {
    auto cta_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shape_mnkl), cute::shape<0>(cta_shape)));
    auto cta_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shape_mnkl), cute::shape<1>(cta_shape)));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      to_gemm_coord(cluster_shape),
      cta_m, cta_n
    );
  }

  // Reloaded interface that receives WorkTileInfo to deduce next work.
  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto
  fetch_next_work(WorkTileInfo work_tile_info) {
    if (continue_current_work(work_tile_info)) {
      return cute::make_tuple(work_tile_info, true);
    }

    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }
  
  // Given the inputs, computes the total number of output blocks over which this problem will compute.
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl,
                          TileShape tile_shape_mnk,
                          AtomThrShape atom_thr_shape_mnk,
                          ClusterShape cluster_shape_mnk) {
    auto [tiles_m, tiles_n, tiles_l] = product_each(ceil_div(select<0,1,3>(problem_shape_mnkl), take<0,2>(tile_shape_mnk)));
    auto cta_m = round_nearest(tiles_m * size<0>(atom_thr_shape_mnk), size<0>(cluster_shape_mnk));
    auto cta_n = round_nearest(tiles_n * size<1>(atom_thr_shape_mnk), size<1>(cluster_shape_mnk));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      to_gemm_coord(cluster_shape_mnk),
      cta_m, cta_n
    );
  }

  // Kernel helper function to get next work tile
  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE
  auto
  fetch_next_work(
      WorkTileInfo work_tile_info,
      TileSchedulerPipeline& scheduler_pipeline,
      TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    return fetch_next_work(work_tile_info);
  }

  CUTLASS_DEVICE
  static auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
    // Get every cta coord in three dimensions of the cluster
    auto [cta_m_in_cluster, cta_n_in_cluster, cta_l_in_cluster] = cute::block_id_in_cluster();
    return make_coord(
      work_tile_info.M_idx + static_cast<int32_t>(cta_m_in_cluster),
      work_tile_info.N_idx + static_cast<int32_t>(cta_n_in_cluster),
      _,
      work_tile_info.L_idx + static_cast<int32_t>(cta_l_in_cluster)
    );
  }

  CUTLASS_DEVICE
  static auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info, dim3 block_id_in_cluster) {
    // Get every cta coord in three dimensions of the cluster
    auto [cta_m_in_cluster, cta_n_in_cluster, cta_l_in_cluster] = block_id_in_cluster;
    return make_coord(
      work_tile_info.M_idx + static_cast<int32_t>(cta_m_in_cluster),
      work_tile_info.N_idx + static_cast<int32_t>(cta_n_in_cluster),
      _,
      work_tile_info.L_idx + static_cast<int32_t>(cta_l_in_cluster)
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
      [[maybe_unused]] Params const& params,
      ProblemShapeMNKL problem_shape_mnk,
      BlockShape cta_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo hw_info,
      Arguments arguments = Arguments{},
      bool truncate_by_problem_size=true) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape_mnk, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  static dim3
  get_grid_shape(
      Params const& params,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo hw_info) {

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cluster_shape_mnk);
    Arguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.log_swizzle_size_;
    }
    args.raster_order = params.raster_order_ == RasterOrder::AlongN ? RasterOrderOptions::AlongN : RasterOrderOptions::AlongM;

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape_mnk),
      hw_info,
      args.max_swizzle_size,
      args.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Convert CTA-level work tile info to cluster-level tile coord
  CUTLASS_DEVICE
  auto
  work_tile_to_cluster_coord_mnkl(WorkTileInfo work_tile_info) const {
    // TileScheduler works at CTA-level, kernel works at cluster-level
    int m_coord = idx2crd(work_tile_info.M_idx / scheduler_params.cluster_shape_m_,
                          scheduler_params.problem_tiles_m_);
    int n_coord = idx2crd(work_tile_info.N_idx / scheduler_params.cluster_shape_n_,
                          scheduler_params.problem_tiles_n_);
    int l_coord = idx2crd(work_tile_info.L_idx,
                          scheduler_params.problem_tiles_l_);
    return make_coord(m_coord, n_coord, _, l_coord);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&) {
    return true;
  }

  // Performs the reduction across splits for a given output tile. Since this scheduler does
  // not split output tiles, no reduction is needed.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(Params const&, WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) {}

  // Performs the reduction across splits for a given output tile. No fixup is required for
  // work units returned by this scheduler.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  fixup(WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) const { }

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo&) {
    return false;
  }

  template <class ProblemShapeMNKL, class TileShape, class Shape>
  CUTLASS_DEVICE
  auto
  get_k_tile_iterator(WorkTileInfo const& work_tile_info, ProblemShapeMNKL problem_shape_MNKL, TileShape tile_shape, Shape) {
    auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape));
    return cute::make_coord_iterator(k_tiles);
  }

  template <class ProblemShape, class TileShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape problem_shape, TileShape tile_shape) {
    return work_tile_info.k_tiles;  // bounded
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const&) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_DEVICE
  static bool
  need_separate_reduction(Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  bool
  is_work_tile_for_reduction(WorkTileInfo const& work_tile_info, Params const& params) {
    return false;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  separate_reduction(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
  }

  // Shares the accumulator set with peers in the global workspace
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  share(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool
  requires_separate_reduction(Params const& params) {
    return false;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
    return 0;
  }
  template <class ProblemShape, class ElementAccumulator>
  static Status initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape, KernelHardwareInfo const&, uint32_t,
                                     const uint32_t = 1, uint32_t = 1, CudaHostAdapter* = nullptr) {
    return Status::kSuccess;
  }

 public:
  Params scheduler_params;
};
}  // namespace mwe_bounded

namespace cutlass::gemm::kernel::detail {
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<mwe_bounded::BoundedScheduler, arch::Sm90, TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = mwe_bounded::BoundedPersistentSm90;
};
}  // namespace cutlass::gemm::kernel::detail
