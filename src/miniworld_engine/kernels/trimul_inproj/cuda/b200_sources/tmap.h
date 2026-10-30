// tmap.h -- host-side CUtensorMap encoding through the runtime's driver entry point (no -lcuda link needed).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace tmap {

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeFn encoder() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p,
                "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeFn>(p);
  }
  return fn;
}

// dims / box innermost-first; strides_bytes for dims 1.. (innermost stride = element size).
inline CUtensorMap make(void* base, CUtensorMapDataType dt, std::vector<uint64_t> dims, std::vector<uint64_t> strides_bytes,
                        std::vector<uint32_t> box, CUtensorMapSwizzle swz,
                        CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_256B) {
  CUtensorMap m;
  std::vector<uint32_t> es(dims.size(), 1);
  auto encode = [&] {
    return encoder()(&m, dt, (cuuint32_t)dims.size(), base, dims.data(), strides_bytes.data(), box.data(), es.data(),
                     CU_TENSOR_MAP_INTERLEAVE_NONE, swz, l2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  CUresult r = encode();
  if (r == CUDA_ERROR_INVALID_CONTEXT) {
    // a thread with no current context (autograd's device thread, when the backward's first CUDA call is a launcher that encodes
    // its maps before any runtime call): bind the device's primary context and retry
    cudaFree(nullptr);
    r = encode();
  }
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

// A channel-major activation (TriMul planes / contraction output / their gradients) as a 4-D map over (token in sample q, channel in
// its 64-channel chunk, sample, chunk): dims {n2, 64, nb, ch / 64}, box {box_tok, 64, 1, box_ch / 64}. The box's shared-memory image is
// that of the 2-D [ch][tok] box, so a kernel loads / stores the same tiles at coordinates (tok % n2, 0, tok / n2, ch / 64). Two layouts:
//   [ch, nb, n2]           (``[P, B, L, L]``, the channel stride is the launch's token count nb n2)
//   [ch / 64, nb, 64, n2]  (sample-major chunks, a 5-D ``[P / 64, B, 64, L, L]`` tensor: the channel stride is one sample's n2 tokens,
//                          as for one sample, whatever B; the D64 training path)
// t: a 5-D tensor (sample-major), or [ch, ...] with ch * n2 * nb elements (n2 = L L, or the token count M with nb = 1).
inline CUtensorMap make_chmajor(const torch::Tensor& t, int64_t n2, uint32_t box_tok, uint32_t box_ch, CUtensorMapSwizzle swz) {
  const bool sm = t.dim() == 5;
  const int64_t ch = sm ? t.size(0) * 64 : t.size(0);
  const int64_t nb = t.numel() / (ch * n2);
  TORCH_CHECK(t.is_contiguous() && ch % 64 == 0 && nb * ch * n2 == t.numel() && box_ch % 64 == 0 && box_ch <= ch,
              "make_chmajor: a contiguous channel-major tensor with whole 64-channel chunks");
  TORCH_CHECK(!sm || (t.size(1) == nb && t.size(2) == 64 && t.size(3) * t.size(4) == n2), "make_chmajor: 5-D tensors are [P / 64, B, 64, L, L]");
  const uint64_t e = 2;   // bf16
  std::vector<uint64_t> strides = sm ? std::vector<uint64_t>{(uint64_t)n2 * e, (uint64_t)(64 * n2) * e, (uint64_t)(nb * 64 * n2) * e}
                                     : std::vector<uint64_t>{(uint64_t)(nb * n2) * e, (uint64_t)n2 * e, (uint64_t)(64 * nb * n2) * e};
  return make(t.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, {(uint64_t)n2, 64, (uint64_t)nb, (uint64_t)(ch / 64)}, strides,
              {box_tok, 64, 1, box_ch / 64}, swz);
}

// the channel count of such a tensor
inline int64_t chmajor_channels(const torch::Tensor& t) { return t.dim() == 5 ? t.size(0) * 64 : t.size(0); }

}  // namespace tmap
