// ops.cu -- the torch extension of the A100 (sm_80) MPNN node-message kernels (mpnn_node_message/cuda/sm80.py)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "node_fwd_sm80.cuh"
#include "node_bwd_sm80.cuh"

#define CHECK_CUDA_BF16(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kBFloat16, #x " must be a CUDA bf16 tensor")
#define CHECK_CUDA_F32(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kFloat32, #x " must be a CUDA fp32 tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_ALIGN16(x) TORCH_CHECK((reinterpret_cast<uintptr_t>((x).data_ptr()) & 15) == 0, #x " must be 16-byte aligned")

namespace {
using bf = __nv_bfloat16;
inline const bf* bptr(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
inline int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }
}  // namespace

// E [G * K, 128] bf16, q [G, 128] bf16, nb [NN, 128] bf16, idx [G * K] int64, W1e / W2 [128, 128] bf16, b2 [128] bf16, mask [G * K] fp32 or bf16 -> reduced [G, 128] fp32
torch::Tensor node_fwd(torch::Tensor edge, torch::Tensor query, torch::Tensor nbt, torch::Tensor idx, torch::Tensor w1, torch::Tensor w2, torch::Tensor b2, torch::Tensor mask, int64_t K, double scale) {
  CHECK_CUDA_BF16(edge); CHECK_CUDA_BF16(query); CHECK_CUDA_BF16(nbt); CHECK_CUDA_BF16(w1); CHECK_CUDA_BF16(w2); CHECK_CUDA_BF16(b2);
  TORCH_CHECK(idx.is_cuda() && idx.scalar_type() == torch::kLong && mask.is_cuda(), "idx must be a CUDA int64 tensor");
  const bool mbf = mask.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(mbf || mask.scalar_type() == torch::kFloat32, "mask must be fp32 or bf16");
  CHECK_CONTIG(edge); CHECK_CONTIG(query); CHECK_CONTIG(nbt); CHECK_CONTIG(idx); CHECK_CONTIG(w1); CHECK_CONTIG(w2); CHECK_CONTIG(b2); CHECK_CONTIG(mask);
  CHECK_ALIGN16(edge); CHECK_ALIGN16(nbt); CHECK_ALIGN16(w1); CHECK_ALIGN16(w2); CHECK_ALIGN16(query);
  TORCH_CHECK(K >= 1 && K <= 128, "K must be in [1, 128]");
  const int64_t groups = query.numel() / 128;
  TORCH_CHECK(groups > 0 && query.numel() == groups * 128 && edge.numel() == groups * K * 128 && idx.numel() == groups * K && mask.numel() == groups * K, "bad operand sizes");
  TORCH_CHECK(w1.numel() == 128 * 128 && w2.numel() == 128 * 128 && b2.numel() == 128 && nbt.numel() % 128 == 0, "bad weight sizes");
  c10::cuda::CUDAGuard guard(edge.device());
  auto out = torch::empty({groups, 128}, edge.options().dtype(torch::kFloat32));
  mp80::NodeFwdParams prm{bptr(edge), bptr(query), bptr(nbt), idx.data_ptr<int64_t>(), bptr(w1), bptr(w2), bptr(b2), mask.data_ptr(), out.data_ptr<float>(), groups, (int)K, (float)(1.0 / scale)};
  using C = mp80::NodeFwdCfg;
  const unsigned ctas = (unsigned)std::min<int64_t>((int64_t)num_sms() * C::MINB, (groups + C::NW - 1) / C::NW);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (mbf) {
    cudaFuncSetAttribute(mp80::node_fwd_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    mp80::node_fwd_kernel<true><<<ctas, C::NTHR, C::SMEM, stream>>>(prm);
  } else {
    cudaFuncSetAttribute(mp80::node_fwd_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    mp80::node_fwd_kernel<false><<<ctas, C::NTHR, C::SMEM, stream>>>(prm);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// the backward but for the two weight-gradient GEMMs and the neighbour scatter (see node_bwd_sm80.cuh): the forward's inputs and gred = d reduced [G, 128] fp32 -> dedge / dpre / dh / act [rows, 128] bf16,
// dquery [G, 128] bf16 and db2_part [ctas, 128] fp32 (one partial row per CTA).  The outputs are the caller's (views of larger buffers when the rows are chunked); returns the CTA count.
int64_t node_bwd(torch::Tensor edge, torch::Tensor query, torch::Tensor nbt, torch::Tensor idx, torch::Tensor w1, torch::Tensor w2, torch::Tensor b2, torch::Tensor mask, torch::Tensor gred,
                 torch::Tensor dedge, torch::Tensor dpre, torch::Tensor dh, torch::Tensor act, torch::Tensor dquery, torch::Tensor db2_part, int64_t K, double scale) {
  CHECK_CUDA_BF16(edge); CHECK_CUDA_BF16(query); CHECK_CUDA_BF16(nbt); CHECK_CUDA_BF16(w1); CHECK_CUDA_BF16(w2); CHECK_CUDA_BF16(b2); CHECK_CUDA_F32(gred);
  CHECK_CUDA_BF16(dedge); CHECK_CUDA_BF16(dpre); CHECK_CUDA_BF16(dh); CHECK_CUDA_BF16(act); CHECK_CUDA_BF16(dquery); CHECK_CUDA_F32(db2_part);
  TORCH_CHECK(idx.is_cuda() && idx.scalar_type() == torch::kLong && mask.is_cuda(), "idx must be a CUDA int64 tensor");
  const bool mbf = mask.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(mbf || mask.scalar_type() == torch::kFloat32, "mask must be fp32 or bf16");
  CHECK_CONTIG(edge); CHECK_CONTIG(query); CHECK_CONTIG(nbt); CHECK_CONTIG(idx); CHECK_CONTIG(w1); CHECK_CONTIG(w2); CHECK_CONTIG(b2); CHECK_CONTIG(mask); CHECK_CONTIG(gred);
  CHECK_CONTIG(dedge); CHECK_CONTIG(dpre); CHECK_CONTIG(dh); CHECK_CONTIG(act); CHECK_CONTIG(dquery); CHECK_CONTIG(db2_part);
  CHECK_ALIGN16(edge); CHECK_ALIGN16(nbt); CHECK_ALIGN16(w1); CHECK_ALIGN16(w2); CHECK_ALIGN16(query); CHECK_ALIGN16(dedge); CHECK_ALIGN16(dpre); CHECK_ALIGN16(dh); CHECK_ALIGN16(act);
  TORCH_CHECK(K >= 1 && K <= 128, "K must be in [1, 128]");
  const int64_t groups = query.numel() / 128;
  TORCH_CHECK(groups > 0 && edge.numel() == groups * K * 128 && idx.numel() == groups * K && mask.numel() == groups * K && gred.numel() == groups * 128, "bad operand sizes");
  TORCH_CHECK(dedge.numel() == edge.numel() && dpre.numel() == edge.numel() && dh.numel() == edge.numel() && act.numel() == edge.numel() && dquery.numel() == query.numel(), "bad output sizes");
  using C = mp80::NodeBwdCfg;
  const int64_t ctas = std::min<int64_t>((int64_t)num_sms() * C::MINB, (groups + C::NW - 1) / C::NW);
  TORCH_CHECK(db2_part.numel() >= ctas * 128, "db2_part is too small");
  c10::cuda::CUDAGuard guard(edge.device());
  mp80::NodeBwdParams prm{bptr(edge), bptr(query), bptr(nbt), idx.data_ptr<int64_t>(), bptr(w1), bptr(w2), bptr(b2), mask.data_ptr(), gred.data_ptr<float>(),
                          reinterpret_cast<bf*>(dedge.data_ptr()), reinterpret_cast<bf*>(dpre.data_ptr()), reinterpret_cast<bf*>(dh.data_ptr()), reinterpret_cast<bf*>(act.data_ptr()),
                          reinterpret_cast<bf*>(dquery.data_ptr()), db2_part.data_ptr<float>(), groups, (int)K, (float)(1.0 / scale)};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (mbf) {
    cudaFuncSetAttribute(mp80::node_bwd_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    mp80::node_bwd_kernel<true><<<(unsigned)ctas, C::NTHR, C::SMEM, stream>>>(prm);
  } else {
    cudaFuncSetAttribute(mp80::node_bwd_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    mp80::node_bwd_kernel<false><<<(unsigned)ctas, C::NTHR, C::SMEM, stream>>>(prm);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ctas;
}

// nb_grad [NN, 128] bf16 = the segmented sum of the dpre rows [rows, 128] bf16 selected by perm [rows] int64, segment j = perm[off[j] .. off[j + 1]) (off [NN + 1] int64)
torch::Tensor nb_reduce(torch::Tensor dpre, torch::Tensor perm, torch::Tensor off, int64_t nn) {
  CHECK_CUDA_BF16(dpre); CHECK_CONTIG(dpre); CHECK_ALIGN16(dpre);
  TORCH_CHECK(perm.is_cuda() && perm.scalar_type() == torch::kLong && perm.is_contiguous() && off.is_cuda() && off.scalar_type() == torch::kLong && off.is_contiguous() && off.numel() == nn + 1, "bad CSR");
  c10::cuda::CUDAGuard guard(dpre.device());
  auto out = torch::empty({nn, 128}, dpre.options());
  mp80::nb_reduce_kernel<<<(unsigned)((nn + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(bptr(dpre), perm.data_ptr<int64_t>(), off.data_ptr<int64_t>(), reinterpret_cast<bf*>(out.data_ptr()), nn);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("node_fwd", &node_fwd);
  m.def("node_bwd", &node_bwd);
  m.def("nb_reduce", &nb_reduce);
}
