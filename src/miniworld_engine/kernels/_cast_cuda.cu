// fp32 / bf16 casts of up to twelve tensors in ONE launch: the one CUDA implementation behind the H100 launchers' weight casts (the
// fp32 master -> bf16 copies of a training forward, the fp32 gradients handed back as the parameters' dtype). Round to nearest even.
// A source is a contiguous tensor or a strided 2-D matrix (a column-major parameter); every destination is contiguous.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kMax = 12;

struct Job {
  const void* src;
  void* dst;
  long rows, cols, s0, s1;  // the source element (r, c) is at r * s0 + c * s1; the destination is dense rows x cols
  int src_f32, dst_f32;
};
struct Jobs {
  Job j[kMax];
  int count;
};

__global__ void cast_k(Jobs p) {
  for (int k = blockIdx.y; k < p.count; k += gridDim.y) {
    const Job a = p.j[k];
    const long n = a.rows * a.cols;
    const bool dense = a.s1 == 1 && a.s0 == a.cols;
    if (dense && a.src_f32 && !a.dst_f32 && n % 4 == 0 && reinterpret_cast<uintptr_t>(a.src) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(a.dst) % 8 == 0) {   // the common fp32 master -> bf16 copy: four elements a thread
      for (long q = (long)blockIdx.x * blockDim.x + threadIdx.x; q < n / 4; q += (long)gridDim.x * blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(a.src)[q];
        const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&lo);
        u.y = *reinterpret_cast<const unsigned*>(&hi);
        reinterpret_cast<uint2*>(a.dst)[q] = u;
      }
      continue;
    }
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
      long o = i;
      if (!dense) {
        const long r = i / a.cols;
        o = r * a.s0 + (i - r * a.cols) * a.s1;
      }
      const float v = a.src_f32 ? static_cast<const float*>(a.src)[o] : __bfloat162float(static_cast<const __nv_bfloat16*>(a.src)[o]);
      if (a.dst_f32) static_cast<float*>(a.dst)[i] = v;
      else static_cast<__nv_bfloat16*>(a.dst)[i] = __float2bfloat16_rn(v);
    }
  }
}

bool supported_dtype(const at::Tensor& t) { return t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16; }

Job job(const at::Tensor& src, const at::Tensor& dst) {
  TORCH_CHECK(src.is_cuda() && supported_dtype(src) && supported_dtype(dst), "cast: fp32 / bf16 CUDA tensors");
  TORCH_CHECK(dst.is_contiguous() && dst.numel() == src.numel() && dst.device() == src.device(), "cast: a contiguous destination of the same size");
  Job j{};
  j.src = src.data_ptr();
  j.dst = dst.data_ptr();
  if (src.is_contiguous()) {
    j.rows = 1;
    j.cols = src.numel();
    j.s0 = j.cols;
    j.s1 = 1;
  } else {
    TORCH_CHECK(src.dim() == 2, "cast: a non-contiguous source must be a 2-D matrix");
    j.rows = src.size(0);
    j.cols = src.size(1);
    j.s0 = src.stride(0);
    j.s1 = src.stride(1);
  }
  j.src_f32 = src.scalar_type() == at::kFloat;
  j.dst_f32 = dst.scalar_type() == at::kFloat;
  return j;
}

void launch(const std::vector<at::Tensor>& srcs, const std::vector<at::Tensor>& dsts) {
  Jobs p{};
  long largest = 0;
  for (size_t k = 0; k < srcs.size(); ++k) {
    if (srcs[k].numel() == 0) continue;
    p.j[p.count++] = job(srcs[k], dsts[k]);
    largest = std::max(largest, srcs[k].numel());
  }
  if (!p.count) return;
  const at::cuda::CUDAGuard guard(srcs[0].device());
  const long blocks = std::min<long>((largest + 255) / 256, 1024);
  cast_k<<<dim3((unsigned)blocks, (unsigned)p.count), 256, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// types[k]: 0 bf16, 1 fp32. A source already of its type (and contiguous) is returned as it is unless `force`; the others are fresh
// contiguous tensors.
std::vector<at::Tensor> cast(std::vector<at::Tensor> srcs, std::vector<int64_t> types, bool force) {
  TORCH_CHECK(!srcs.empty() && srcs.size() <= kMax && srcs.size() == types.size(), "cast: one to twelve tensors with one type each");
  std::vector<at::Tensor> out, from, to;
  for (size_t k = 0; k < srcs.size(); ++k) {
    TORCH_CHECK(types[k] == 0 || types[k] == 1, "cast: type 0 (bf16) or 1 (fp32)");
    const auto dtype = types[k] ? at::kFloat : at::kBFloat16;
    if (!force && srcs[k].scalar_type() == dtype && srcs[k].is_contiguous()) {
      out.push_back(srcs[k]);
      continue;
    }
    out.push_back(at::empty(srcs[k].sizes(), srcs[k].options().dtype(dtype).memory_format(at::MemoryFormat::Contiguous)));
    from.push_back(srcs[k]);
    to.push_back(out.back());
  }
  if (!from.empty()) launch(from, to);
  return out;
}

// Casts each source into its (contiguous, preallocated) destination: row blocks of one buffer, a packed layout.
void cast_into(std::vector<at::Tensor> srcs, std::vector<at::Tensor> dsts) {
  TORCH_CHECK(!srcs.empty() && srcs.size() <= kMax && srcs.size() == dsts.size(), "cast_into: one to twelve source / destination pairs");
  launch(srcs, dsts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cast", &cast, "fp32 / bf16 casts of up to twelve tensors in one launch", py::arg("srcs"), py::arg("types"), py::arg("force") = false);
  m.def("cast_into", &cast_into, "fp32 / bf16 casts of up to twelve tensors into preallocated destinations", py::arg("srcs"),
        py::arg("dsts"));
}
