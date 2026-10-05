// ops.cu -- torch bindings of the A100 triangle-attention kernels, on the current CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <climits>

#include "attn_fwd_sm80.cuh"
#include "attn_bwd_dq_sm80.cuh"
#include "attn_bwd_dkv_sm80.cuh"
#include "f1_sm80.cuh"
#include "f3_sm80.cuh"
#include "b3_sm80.cuh"
#include "b1_sm80.cuh"
#include "wgrad_sm80.cuh"

namespace {

template <class G>
void launch_attn_fwd(const a100::AttnParams& p, int z) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::attn_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / a100::TA_BM, p.L / G::R, z * p.H);
  a100::attn_fwd_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The front: one persistent CTA of G::NW warps per SM (each warp walks the 32-token tiles of the grid-stride), W' resident in shared memory.
template <class G>
void launch_front(const a100::F1Params& p, int nsm) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::f1_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  const unsigned grid = std::min<unsigned>((unsigned)nsm, (p.ntile + G::NW - 1) / G::NW);
  a100::f1_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The back: one persistent CTA of G::NW warps per SM (each warp walks the 16-token tiles of the grid-stride), Wo resident in shared memory.
template <class G>
void launch_back(const a100::F3Params& p, int nsm) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::f3_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  const unsigned grid = std::min<unsigned>((unsigned)nsm, (p.ntile + G::NW - 1) / G::NW);
  a100::f3_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// One warp per batch constructs a stable key order. No host count or dynamic allocation.
__global__ void compact_mask_kernel(const bool* mask, int* indices, int* counts, int L) {
  const int z = blockIdx.x, lane = threadIdx.x;
  int n = 0;
  for (int j = lane; j < L; j += 32) {
    const bool keep = mask[z * L + j];
    const unsigned bits = __ballot_sync(0xffffffffu, keep);
    if (keep) indices[z * L + n + __popc(bits & ((1u << lane) - 1))] = j;
    n += __popc(bits);
  }
  if (lane == 0) counts[z] = n;
  for (int j = n + lane; j < L; j += 32) indices[z * L + j] = -1;
}

__global__ void compact_bias_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, const int* indices, int L, int H) {
  const int j = blockIdx.x * 256 + threadIdx.x, row = blockIdx.y, zh = blockIdx.z, z = zh / H;
  if (j < L) {
    const long long t = ((long long)zh * L + row) * L + j;
    const int source = indices[z * L + j];
    out[t] = source < 0 ? __float2bfloat16_rn(-3.3895313892515355e38f) : in[t - j + source];
  }
}

// The module has already applied this global key mask to bias. Only the small bias
// planes are gathered; K/V retain their original layout and are indexed by the core.
void compact_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor mask,
                       torch::Tensor out, torch::Tensor indices, torch::Tensor counts, torch::Tensor compact_bias, double scl) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t Z = q.size(0), L = q.size(1), H = q.size(3) / 32;
  TORCH_CHECK(L >= 768 && L % 384 == 0 && q.size(2) == L && q.size(3) == H * 32 && H > 0, "compact_attention: square L % 384 == 0, head dim 32");
  TORCH_CHECK(Z * H <= 65535 && L <= 65535, "compact_attention: grid");
  for (const auto& t : {q, k, v, bias, out, compact_bias})
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.scalar_type() == at::kBFloat16, "compact_attention: bf16 tensors on one CUDA device");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() && out.sizes() == q.sizes() && out.is_contiguous(), "compact_attention: output and Q/K/V shapes");
  for (const auto& t : {q, k, v})
    TORCH_CHECK(t.stride(3) == 1 && t.stride(2) >= H * 32 && t.stride(2) % 8 == 0 && t.stride(1) == L * t.stride(2)
                && (Z == 1 || t.stride(0) == L * t.stride(1)) && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "compact_attention: aligned token-major Q/K/V");
  TORCH_CHECK(bias.is_contiguous() && compact_bias.is_contiguous() && bias.numel() == Z * H * L * L && compact_bias.sizes() == bias.sizes(), "compact_attention: bias planes");
  TORCH_CHECK(mask.device() == q.device() && mask.scalar_type() == at::kBool && mask.is_contiguous() && mask.numel() == Z * L, "compact_attention: bool mask [Z, L]");
  for (const auto& t : {indices, counts})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == at::kInt && t.is_contiguous(), "compact_attention: int32 scratch");
  TORCH_CHECK(indices.numel() == Z * L && counts.numel() == Z, "compact_attention: scratch sizes");
  const auto stream = at::cuda::getCurrentCUDAStream();
  compact_mask_kernel<<<Z, 32, 0, stream>>>(mask.data_ptr<bool>(), indices.data_ptr<int>(), counts.data_ptr<int>(), L);
  compact_bias_kernel<<<dim3((L + 255) / 256, L, Z * H), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()), reinterpret_cast<__nv_bfloat16*>(compact_bias.data_ptr()), indices.data_ptr<int>(), L, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  a100::AttnParams p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.bias = reinterpret_cast<const __nv_bfloat16*>(compact_bias.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldq = q.stride(2); p.ldk = k.stride(2); p.ldv = v.stride(2); p.ldo = H * 32; p.L = L; p.H = H; p.scl = scl;
  p.indices = indices.data_ptr<int>(); p.counts = counts.data_ptr<int>();
  // Two K/V stages reduce the compact loader's address/register pressure and
  // shared-memory footprint (60 KiB instead of 72 KiB). Keep two resident CTAs;
  // the dense forward and the backward retain their separate schedules.
  launch_attn_fwd<a100::AttnCfg<2, 2, 48, 2, false, true>>(p, Z);
}

// Attention core forward.  q / k / v / out: token-major bf16 [z * L * L, ld] (element strides ldq / ldk / ldv / ldo, unit column stride; head h = columns
// [32 h, 32 h + 32)); bias [z, H, L, L] bf16 contiguous (masked keys = bf16 min); lse: [z, H, L, L] fp32 or empty.  Schedule: `rows` pair rows
// per CTA, `bn` keys per tile, `minb` CTAs per SM the registers are bounded for.
void attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor out, torch::Tensor lse, int64_t L, int64_t H,
              int64_t Z, int64_t ldq, int64_t ldk, int64_t ldv, int64_t ldo, double scl, int64_t rows, int64_t bn, int64_t minb) {
  for (auto* t : {&q, &k, &v, &bias, &out}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_fwd: bf16 CUDA tensors");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == Z * H * L * L, "attn_fwd: bias [z, H, L, L] contiguous");
  TORCH_CHECK(L % a100::TA_BM == 0 && L >= a100::TA_BM, "attn_fwd: L must be a multiple of 128");
  // q / k / v / out are token-major views (rows of H * 32 channels, element stride ld between tokens: the projections of the front keep q | k | v | g in one
  // [T, 512] buffer), so the extents are the storage's, and the 16-byte loads need aligned bases and strides
  const int64_t T = Z * L * L;
  for (auto [t, ld] : {std::pair<const torch::Tensor*, int64_t>{&q, ldq}, {&k, ldk}, {&v, ldv}, {&out, ldo}}) {
    TORCH_CHECK(ld >= H * 32 && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_fwd: row strides / alignment");
    TORCH_CHECK((t->storage_offset() + (T - 1) * ld + H * 32) * t->element_size() <= (int64_t)t->storage().nbytes(), "attn_fwd: tensor extents");
  }
  TORCH_CHECK(!lse.numel() || (lse.scalar_type() == at::kFloat && lse.numel() == Z * H * L * L && lse.is_contiguous()), "attn_fwd: lse [z, H, L, L] fp32");
  TORCH_CHECK(Z * H <= 65535 && L / rows <= 65535, "attn_fwd: grid");
  a100::AttnParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); p.lse = lse.numel() ? lse.data_ptr<float>() : nullptr;
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.ldo = ldo; p.L = (int)L; p.H = (int)H; p.scl = (float)scl;
  TORCH_CHECK(L % bn == 0, "attn_fwd: L must be a multiple of bn");
  const int64_t cfg = rows * 10000 + bn * 10 + minb;
  if (cfg == 4 * 10000 + 64 * 10 + 1) launch_attn_fwd<a100::AttnCfg<4, 4, 64, 1>>(p, (int)Z);
  else if (cfg == 2 * 10000 + 64 * 10 + 1) launch_attn_fwd<a100::AttnCfg<2, 4, 64, 1>>(p, (int)Z);
  else if (cfg == 2 * 10000 + 32 * 10 + 2) launch_attn_fwd<a100::AttnCfg<2, 4, 32, 2>>(p, (int)Z);
  else if (cfg == 2 * 10000 + 32 * 10 + 1) launch_attn_fwd<a100::AttnCfg<2, 4, 32, 1>>(p, (int)Z);
  else if (cfg == 4 * 10000 + 32 * 10 + 1) launch_attn_fwd<a100::AttnCfg<4, 4, 32, 1>>(p, (int)Z);
  else TORCH_CHECK(false, "attn_fwd: unsupported schedule (rows, bn, minb)");
}

// Attention core forward with the sigmoid gate fused into its epilogue (the generic-width path): aout [z L L, ldao] = bf16(sigmoid(gate) bf16(out)) where gate is the front's g columns
// (token-major view, row stride ldgate); `out` (the plain attention output, needed by the backward) may be an empty tensor.  The default schedule (2 rows, 32 keys, 2 CTAs per SM).
void attn_fwd_gate(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, torch::Tensor out, torch::Tensor lse, torch::Tensor gate, torch::Tensor aout, int64_t L,
                   int64_t H, int64_t Z, int64_t ldq, int64_t ldk, int64_t ldv, int64_t ldo, int64_t ldgate, int64_t ldao, double scl) {
  for (auto* t : {&q, &k, &v, &bias, &gate, &aout}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_fwd_gate: bf16 CUDA tensors");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == Z * H * L * L, "attn_fwd_gate: bias [z, H, L, L] contiguous");
  TORCH_CHECK(L % a100::TA_BM == 0 && L >= a100::TA_BM && L % 32 == 0, "attn_fwd_gate: L must be a multiple of 128");
  const int64_t T = Z * L * L;
  for (auto [t, ld] : {std::pair<const torch::Tensor*, int64_t>{&q, ldq}, {&k, ldk}, {&v, ldv}, {&gate, ldgate}, {&aout, ldao}}) {
    TORCH_CHECK(ld >= H * 32 && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_fwd_gate: row strides / alignment");
    TORCH_CHECK((t->storage_offset() + (T - 1) * ld + H * 32) * t->element_size() <= (int64_t)t->storage().nbytes(), "attn_fwd_gate: tensor extents");
  }
  TORCH_CHECK(!out.numel() || (out.is_cuda() && out.scalar_type() == at::kBFloat16 && ldo >= H * 32 && ldo % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 &&
              (out.storage_offset() + (T - 1) * ldo + H * 32) * out.element_size() <= (int64_t)out.storage().nbytes()), "attn_fwd_gate: out");
  TORCH_CHECK(!lse.numel() || (lse.scalar_type() == at::kFloat && lse.numel() == Z * H * L * L && lse.is_contiguous()), "attn_fwd_gate: lse [z, H, L, L] fp32");
  TORCH_CHECK(Z * H <= 65535 && L / 2 <= 65535, "attn_fwd_gate: grid");
  a100::AttnParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.out = out.numel() ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr; p.lse = lse.numel() ? lse.data_ptr<float>() : nullptr;
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.ldo = ldo; p.L = (int)L; p.H = (int)H; p.scl = (float)scl;
  p.gate = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()); p.aout = reinterpret_cast<__nv_bfloat16*>(aout.data_ptr()); p.ldgate = ldgate; p.ldao = ldao;
  launch_attn_fwd<a100::AttnCfg<2, 4, 32, 2, true>>(p, (int)Z);
}

// The front: input LayerNorm + q | k | v | g + bias projections (f1_sm80.cuh).  x [T, 128] bf16 (T = Z L^2 tokens of Z square pair stacks, L a multiple of
// 128; transposed != 0: the ending node, token (a, b) read from row (b, a)), w [520, 128] bf16 = W' (rows 516-519 zero), bvec [520] fp32, mask [Z, L]
// uint8 or empty, out [T, 512] bf16 (q | k | v | g, token-major), bias [Z, 4, L, L] bf16, lnst [T, 2] fp32 (mean, rstd) or empty.
void front(torch::Tensor x, torch::Tensor w, torch::Tensor bvec, torch::Tensor mask, torch::Tensor out, torch::Tensor bias, torch::Tensor lnst, torch::Tensor xh,
           int64_t L, int64_t transposed, double eps, int64_t nw) {
  const int64_t T = x.numel() / 128;
  TORCH_CHECK(L % 128 == 0 && L >= 128, "front: L must be a multiple of 128");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() == T * 128 && T % (L * L) == 0 && T < (int64_t(1) << 31),
              "front: x [Z L^2, 128] bf16 contiguous");
  const int64_t Z = T / (L * L);
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == 520 * 128, "front: w [520, 128] bf16");
  TORCH_CHECK(bvec.scalar_type() == at::kFloat && bvec.is_contiguous() && bvec.numel() == 520, "front: bvec [520] fp32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == T * 512, "front: out [T, 512] bf16");
  TORCH_CHECK(bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias.numel() == 4 * T, "front: bias [Z, 4, L, L] bf16");
  TORCH_CHECK(!mask.numel() || (mask.scalar_type() == at::kByte && mask.is_contiguous() && mask.numel() == Z * L), "front: mask [Z, L] uint8");
  TORCH_CHECK(!lnst.numel() || (lnst.scalar_type() == at::kFloat && lnst.is_contiguous() && lnst.numel() == 2 * T), "front: lnst [T, 2] fp32");
  a100::F1Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.bvec = bvec.data_ptr<float>(); p.mask = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); p.bias = reinterpret_cast<__nv_bfloat16*>(bias.data_ptr());
  p.lnst = lnst.numel() ? lnst.data_ptr<float>() : nullptr;
  TORCH_CHECK(!xh.numel() || (xh.scalar_type() == at::kBFloat16 && xh.is_contiguous() && xh.numel() == T * 136), "front: xh [T, 136] bf16 (training)");
  p.xh = xh.numel() ? reinterpret_cast<__nv_bfloat16*>(xh.data_ptr()) : nullptr;
  p.L = (int)L; p.transposed = (int)transposed; p.eps = (float)eps; p.ntile = (unsigned)(T / 32);
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (nw == 8) launch_front<a100::F1Cfg<8>>(p, nsm);
  else if (nw == 12) launch_front<a100::F1Cfg<12>>(p, nsm);
  else if (nw == 16) launch_front<a100::F1Cfg<16>>(p, nsm);
  else TORCH_CHECK(false, "front: unsupported warps per CTA");
}

// The front's weights in its layout (f1_pack_kernel): wq / wk / wv / wg [128, 128] bf16, wb [4, 128] bf16 (contiguous), gamma / beta [128] fp32
// -> wp [520, 128] bf16, bvec [520] fp32.
void front_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor gamma, torch::Tensor beta,
                torch::Tensor wp, torch::Tensor bvec) {
  const torch::Tensor* ws[5] = {&wq, &wk, &wv, &wg, &wb};
  a100::F1PackParams p;
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(ws[i]->is_cuda() && ws[i]->scalar_type() == at::kBFloat16 && ws[i]->is_contiguous() && ws[i]->numel() == (i < 4 ? 128 : 4) * 128 &&
                reinterpret_cast<uintptr_t>(ws[i]->data_ptr()) % 8 == 0, "front_pack: weights [128 | 4, 128] bf16 contiguous");
    p.w[i] = reinterpret_cast<const __nv_bfloat16*>(ws[i]->data_ptr());
  }
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && gamma.is_contiguous() && gamma.numel() == 128 && beta.scalar_type() == at::kFloat &&
              beta.is_contiguous() && beta.numel() == 128, "front_pack: gamma / beta [128] fp32");
  TORCH_CHECK(wp.scalar_type() == at::kBFloat16 && wp.is_contiguous() && wp.numel() == 520 * 128 && bvec.scalar_type() == at::kFloat &&
              bvec.is_contiguous() && bvec.numel() == 520, "front_pack: outputs wp [520, 128] bf16, bvec [520] fp32");
  p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>();
  p.wp = reinterpret_cast<__nv_bfloat16*>(wp.data_ptr()); p.bvec = bvec.data_ptr<float>();
  a100::f1_pack_kernel<<<65, 256, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The back (f3_sm80.cuh): out = res + bf16(bf16(sigmoid(g) o) . Wo^T).  o [T, 128] bf16 contiguous (the attention output, token-major), g [T, 128] bf16 with
// row stride ldg = g.stride(0) (the gate columns of the front's [T, 512] buffer), wo [128, 128] bf16 (to_out.weight), res / out [Z, L, L, 128] bf16
// contiguous and distinct (transposed != 0: the ending node, token (a, b) reads / writes row (b, a)).
void back(torch::Tensor o, torch::Tensor g, torch::Tensor wo, torch::Tensor res, torch::Tensor out, torch::Tensor ds, int64_t L, int64_t transposed, int64_t nw) {
  const int64_t T = o.numel() / 128;
  TORCH_CHECK(L % 128 == 0 && L >= 128, "back: L must be a multiple of 128");
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kBFloat16 && o.is_contiguous() && o.numel() == T * 128 && T % (L * L) == 0 && T < (int64_t(1) << 31),
              "back: o [Z L^2, 128] bf16 contiguous");
  TORCH_CHECK(g.scalar_type() == at::kBFloat16 && g.dim() == 2 && g.size(0) == T && g.size(1) == 128 && g.stride(1) == 1 && g.stride(0) >= 128 &&
              g.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0 &&
              (g.storage_offset() + (T - 1) * g.stride(0) + 128) * g.element_size() <= (int64_t)g.storage().nbytes(), "back: g [T, 128] rows of 16-byte aligned bf16");
  TORCH_CHECK(wo.scalar_type() == at::kBFloat16 && wo.is_contiguous() && wo.numel() == 128 * 128, "back: wo [128, 128] bf16");
  TORCH_CHECK(res.scalar_type() == at::kBFloat16 && res.is_contiguous() && res.numel() == T * 128 && out.scalar_type() == at::kBFloat16 &&
              out.is_contiguous() && out.numel() == T * 128 && res.data_ptr() != out.data_ptr(), "back: res / out [Z, L, L, 128] bf16 contiguous, distinct");
  a100::F3Params p;
  p.o = reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()); p.g = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr());
  p.wo = reinterpret_cast<const __nv_bfloat16*>(wo.data_ptr()); p.res = reinterpret_cast<const __nv_bfloat16*>(res.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  TORCH_CHECK(!ds.numel() || (ds.scalar_type() == at::kBFloat16 && ds.is_contiguous() && ds.numel() == (T / L) * 128 && reinterpret_cast<uintptr_t>(ds.data_ptr()) % 16 == 0),
              "back: ds [Z, L, 128] bf16 (the dropout scale, training)");
  p.ds = ds.numel() ? reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr()) : nullptr;
  p.ldg = g.stride(0); p.ntile = (unsigned)(T / 16); p.L = (int)L; p.transposed = (int)transposed;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (nw == 8) launch_back<a100::F3Cfg<8>>(p, nsm);
  else if (nw == 12) launch_back<a100::F3Cfg<12>>(p, nsm);
  else if (nw == 16) launch_back<a100::F3Cfg<16>>(p, nsm);
  else TORCH_CHECK(false, "back: unsupported warps per CTA");
}

// The back's backward (b3_sm80.cuh): dout [T, 128] bf16 (the module's layout; transposed != 0: the ending node, token (a, b) reads row (b, a)), ds [Z, L, 128]
// bf16 or empty, o [T, 128], g [T, 128] (row stride g.stride(0): the gate columns of the front's [T, 512] buffer), wot [128, 128] = Wo^T.  Writes dg
// [T, 128] (row stride dg.stride(0): the gate columns of the [T, 512] gradient buffer), dov, dy, a [T, 128] (starting frame).
template <class G>
void launch_back_bwd(const a100::B3Params& p, int nsm) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::b3_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  const unsigned grid = std::min<unsigned>((unsigned)nsm, (p.ntile + G::NW - 1) / G::NW);
  a100::b3_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void back_bwd(torch::Tensor dout, torch::Tensor ds, torch::Tensor o, torch::Tensor g, torch::Tensor wot, torch::Tensor dg, torch::Tensor dov, torch::Tensor dy,
              torch::Tensor a, torch::Tensor delta, int64_t L, int64_t transposed, int64_t nw) {
  const int64_t T = o.numel() / 128;
  TORCH_CHECK(L % 128 == 0 && L >= 128, "back_bwd: L must be a multiple of 128");
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kBFloat16 && o.is_contiguous() && o.numel() == T * 128 && T % (L * L) == 0 && T < (int64_t(1) << 31),
              "back_bwd: o [Z L^2, 128] bf16 contiguous");
  for (const torch::Tensor* t : {&dout, &dov, &dy, &a})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->numel() == T * 128, "back_bwd: dout / dov / dy / a [T, 128] bf16 contiguous");
  for (const torch::Tensor* t : {&g, &dg})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 2 && t->size(0) == T && t->size(1) == 128 && t->stride(1) == 1 && t->stride(0) >= 128 &&
                t->stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0 &&
                (t->storage_offset() + (T - 1) * t->stride(0) + 128) * t->element_size() <= (int64_t)t->storage().nbytes(),
                "back_bwd: g / dg [T, 128] rows of 16-byte aligned bf16");
  TORCH_CHECK(wot.scalar_type() == at::kBFloat16 && wot.is_contiguous() && wot.numel() == 128 * 128, "back_bwd: wot [128, 128] bf16");
  TORCH_CHECK(!ds.numel() || (ds.scalar_type() == at::kBFloat16 && ds.is_contiguous() && ds.numel() == (T / L) * 128 && reinterpret_cast<uintptr_t>(ds.data_ptr()) % 16 == 0),
              "back_bwd: ds [Z, L, 128] bf16");
  a100::B3Params p;
  p.dout = reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr()); p.ds = ds.numel() ? reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr()) : nullptr;
  p.o = reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()); p.g = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr());
  p.wot = reinterpret_cast<const __nv_bfloat16*>(wot.data_ptr());
  p.dg = reinterpret_cast<__nv_bfloat16*>(dg.data_ptr()); p.dov = reinterpret_cast<__nv_bfloat16*>(dov.data_ptr());
  p.dy = reinterpret_cast<__nv_bfloat16*>(dy.data_ptr()); p.a = reinterpret_cast<__nv_bfloat16*>(a.data_ptr());
  TORCH_CHECK(!delta.numel() || (delta.scalar_type() == at::kFloat && delta.is_contiguous() && delta.numel() == 4 * T), "back_bwd: delta [Z, 4, L, L] fp32");
  p.delta = delta.numel() ? delta.data_ptr<float>() : nullptr;
  p.ldg = g.stride(0); p.lddg = dg.stride(0); p.ntile = (unsigned)(T / 16); p.L = (int)L; p.transposed = (int)transposed;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (nw == 8) launch_back_bwd<a100::F3Cfg<8>>(p, nsm);
  else if (nw == 12) launch_back_bwd<a100::F3Cfg<12>>(p, nsm);
  else TORCH_CHECK(false, "back_bwd: unsupported warps per CTA");
}

// The front's backward (b1_sm80.cuh): d [T, 512] bf16 (dq | dk | dv | dg, row stride d.stride(0) >= 512), db [Z, 4, L, L] bf16, x / dout / dpair [T, 128]
// bf16 in the module's layout (transposed != 0: the ending node, token (a, b) reads / writes row (b, a); dpair must not alias x or dout), lnst [T, 2]
// fp32 (mean, rstd), wt [4, 128, 128] bf16 (W_p^T of q, k, v, g), wbt [128, 4] bf16 (Wb^T), gamma [128] fp32.
template <class G>
void launch_front_bwd(const a100::B1Params& p, int nsm) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::b1_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  const unsigned grid = std::min<unsigned>((unsigned)nsm, (p.ntile + G::NW - 1) / G::NW);
  a100::b1_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void front_bwd(torch::Tensor d, torch::Tensor db, torch::Tensor x, torch::Tensor lnst, torch::Tensor dout, torch::Tensor dpair, torch::Tensor wt, torch::Tensor wbt,
               torch::Tensor gamma, int64_t L, int64_t transposed, int64_t nw) {
  const int64_t T = x.numel() / 128;
  TORCH_CHECK(L % 128 == 0 && L >= 128, "front_bwd: L must be a multiple of 128");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() == T * 128 && T % (L * L) == 0 && T < (int64_t(1) << 31),
              "front_bwd: x [Z L^2, 128] bf16 contiguous");
  for (const torch::Tensor* t : {&dout, &dpair})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->numel() == T * 128, "front_bwd: dout / dpair [T, 128] bf16 contiguous");
  TORCH_CHECK(dpair.data_ptr() != x.data_ptr() && dpair.data_ptr() != dout.data_ptr(), "front_bwd: dpair must not alias x or dout");
  TORCH_CHECK(d.scalar_type() == at::kBFloat16 && d.dim() == 2 && d.size(0) == T && d.size(1) >= 512 && d.stride(1) == 1 && d.stride(0) >= 512 &&
              d.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0 &&
              (d.storage_offset() + (T - 1) * d.stride(0) + 512) * d.element_size() <= (int64_t)d.storage().nbytes(), "front_bwd: d [T, 512] rows of 16-byte aligned bf16");
  TORCH_CHECK(db.scalar_type() == at::kBFloat16 && db.is_contiguous() && db.numel() == 4 * T, "front_bwd: db [Z, 4, L, L] bf16");
  TORCH_CHECK(lnst.scalar_type() == at::kFloat && lnst.is_contiguous() && lnst.numel() == 2 * T, "front_bwd: lnst [T, 2] fp32");
  TORCH_CHECK(wt.scalar_type() == at::kBFloat16 && wt.is_contiguous() && wt.numel() == 4 * 128 * 128 && wbt.scalar_type() == at::kBFloat16 &&
              wbt.is_contiguous() && wbt.numel() == 128 * 4, "front_bwd: wt [4, 128, 128] / wbt [128, 4] bf16");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && gamma.is_contiguous() && gamma.numel() == 128, "front_bwd: gamma [128] fp32");
  a100::B1Params p;
  p.d = reinterpret_cast<const __nv_bfloat16*>(d.data_ptr()); p.db = reinterpret_cast<const __nv_bfloat16*>(db.data_ptr());
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.lnst = lnst.data_ptr<float>();
  p.dout = reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr()); p.dpair = reinterpret_cast<__nv_bfloat16*>(dpair.data_ptr());
  p.wt = reinterpret_cast<const __nv_bfloat16*>(wt.data_ptr()); p.wbt = reinterpret_cast<const __nv_bfloat16*>(wbt.data_ptr());
  p.gamma = gamma.data_ptr<float>();
  p.ldd = d.stride(0); p.ntile = (unsigned)(T / 16); p.L = (int)L; p.transposed = (int)transposed;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (nw == 8) launch_front_bwd<a100::B1Cfg<8>>(p, nsm);
  else if (nw == 12) launch_front_bwd<a100::B1Cfg<12>>(p, nsm);
  else TORCH_CHECK(false, "front_bwd: unsupported warps per CTA");
}

// The attention core's backward, query side (attn_bwd_dq_sm80.cuh): q / k / v token-major views [z L L, ld] bf16 (head h = columns [32 h, 32 h + 32)), dov
// [z L L, lddo] (the attention output's gradient), bias [z, H, L, L] bf16, lse / delta [z, H, L, L] fp32 -> dq [z L L, lddq] bf16 (written in place, e.g. the
// first 128 columns of the [T, 512] gradient buffer) and the partial bias gradients dbp [L / rows, z H, L, L] bf16 (``db_reduce`` sums them).
template <class G>
void launch_attn_bwd_dq(const a100::DqParams& p, int z) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::attn_bwd_dq_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / a100::TA_BM, p.L / G::R, z * p.H);
  a100::attn_bwd_dq_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_bwd_dq(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias, torch::Tensor lse, torch::Tensor delta,
                 torch::Tensor dq, torch::Tensor dbp, int64_t L, int64_t H, int64_t Z, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo, int64_t lddq,
                 double scl, double sm_scale, int64_t rows, int64_t nkv, int64_t bn, int64_t minb) {
  for (auto* t : {&q, &k, &v, &dov, &bias, &dq, &dbp}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_bwd_dq: bf16 CUDA tensors");
  TORCH_CHECK(bias.is_contiguous() && bias.numel() == Z * H * L * L, "attn_bwd_dq: bias [z, H, L, L] contiguous");
  for (auto* t : {&lse, &delta})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == Z * H * L * L, "attn_bwd_dq: lse / delta [z, H, L, L] fp32");
  TORCH_CHECK(L % a100::TA_BM == 0 && L >= a100::TA_BM && L % bn == 0 && L % rows == 0, "attn_bwd_dq: L must be a multiple of 128");
  const int64_t T = Z * L * L;
  for (auto [t, ld] : {std::pair<const torch::Tensor*, int64_t>{&q, ldq}, {&k, ldk}, {&v, ldv}, {&dov, lddo}, {&dq, lddq}}) {
    TORCH_CHECK(ld >= H * 32 && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_bwd_dq: row strides / alignment");
    TORCH_CHECK((t->storage_offset() + (T - 1) * ld + H * 32) * t->element_size() <= (int64_t)t->storage().nbytes(), "attn_bwd_dq: tensor extents");
  }
  TORCH_CHECK(dbp.is_contiguous() && dbp.numel() == (L / rows) * Z * H * L * L, "attn_bwd_dq: dbp [L / rows, z H, L, L] contiguous");
  TORCH_CHECK(Z * H <= 65535 && L / rows <= 65535, "attn_bwd_dq: grid");
  a100::DqParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.dov = reinterpret_cast<const __nv_bfloat16*>(dov.data_ptr());
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
  p.dq = reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()); p.dbp = reinterpret_cast<__nv_bfloat16*>(dbp.data_ptr());
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddq = lddq; p.L = (int)L; p.H = (int)H; p.ZH = (int)(Z * H);
  p.scl = (float)scl; p.sm_scale = (float)sm_scale;
  const int64_t cfg = ((rows * 100 + nkv) * 100 + bn) * 10 + minb;
#define DQ_CFG(R, NKV, BN, MINB) \
  if (cfg == ((int64_t(R) * 100 + (NKV)) * 100 + (BN)) * 10 + (MINB)) { launch_attn_bwd_dq<a100::DqCfg<R, NKV, BN, MINB>>(p, (int)Z); return; }
  DQ_CFG(2, 4, 32, 2) DQ_CFG(2, 3, 32, 2) DQ_CFG(2, 4, 32, 1) DQ_CFG(2, 3, 32, 1) DQ_CFG(4, 3, 32, 1) DQ_CFG(2, 3, 64, 1)
#undef DQ_CFG
  TORCH_CHECK(false, "attn_bwd_dq: unsupported schedule (rows, nkv, bn, minb)");
}

// db [z H, L, L] bf16 = the sum of the `groups` partials dbp [groups, z H, L, L] (fixed order).
void db_reduce(torch::Tensor dbp, torch::Tensor db, int64_t groups) {
  TORCH_CHECK(dbp.is_cuda() && dbp.scalar_type() == at::kBFloat16 && dbp.is_contiguous() && db.scalar_type() == at::kBFloat16 && db.is_contiguous() &&
              dbp.numel() == groups * db.numel() && db.numel() % 8 == 0, "db_reduce: dbp [groups, ...] / db [...] bf16 contiguous");
  const long long plane8 = db.numel() / 8;
  a100::db_reduce_kernel<<<(unsigned)((plane8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dbp.data_ptr()), reinterpret_cast<__nv_bfloat16*>(db.data_ptr()), (int)groups, plane8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The attention core's backward, key side (attn_bwd_dkv_sm80.cuh): as attn_bwd_dq plus bias_t [z H, L, L] = the bias transposed (bias_t[k][j]) -> dk / dv
// token-major views (written in place).  Schedule: `nst` query-tile ring stages, `bn` = 32 queries per tile, `minb` CTAs per SM.
template <class G>
void launch_attn_bwd_dkv(const a100::DkvParams& p, int z) {
  static bool set[64] = {};                                   // the attribute is per device
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(a100::attn_bwd_dkv_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    set[dev] = true;
  }
  dim3 grid(p.L / a100::TA_BM, p.L, z * p.H);
  a100::attn_bwd_dkv_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_bwd_dkv(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dov, torch::Tensor bias_t, torch::Tensor lse, torch::Tensor delta,
                  torch::Tensor dk, torch::Tensor dv, int64_t L, int64_t H, int64_t Z, int64_t ldq, int64_t ldk, int64_t ldv, int64_t lddo, int64_t lddk,
                  int64_t lddv, double scl, double sm_scale, int64_t nst, int64_t bn, int64_t minb) {
  for (auto* t : {&q, &k, &v, &dov, &bias_t, &dk, &dv}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16, "attn_bwd_dkv: bf16 CUDA tensors");
  TORCH_CHECK(bias_t.is_contiguous() && bias_t.numel() == Z * H * L * L, "attn_bwd_dkv: bias_t [z, H, L, L] contiguous");
  for (auto* t : {&lse, &delta})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == Z * H * L * L, "attn_bwd_dkv: lse / delta [z, H, L, L] fp32");
  TORCH_CHECK(L % a100::TA_BM == 0 && L >= a100::TA_BM && L % bn == 0, "attn_bwd_dkv: L must be a multiple of 128");
  const int64_t T = Z * L * L;
  for (auto [t, ld] : {std::pair<const torch::Tensor*, int64_t>{&q, ldq}, {&k, ldk}, {&v, ldv}, {&dov, lddo}, {&dk, lddk}, {&dv, lddv}}) {
    TORCH_CHECK(ld >= H * 32 && ld % 8 == 0 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "attn_bwd_dkv: row strides / alignment");
    TORCH_CHECK((t->storage_offset() + (T - 1) * ld + H * 32) * t->element_size() <= (int64_t)t->storage().nbytes(), "attn_bwd_dkv: tensor extents");
  }
  TORCH_CHECK(Z * H <= 65535 && L <= 65535, "attn_bwd_dkv: grid");
  a100::DkvParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); p.dov = reinterpret_cast<const __nv_bfloat16*>(dov.data_ptr());
  p.bias_t = reinterpret_cast<const __nv_bfloat16*>(bias_t.data_ptr()); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
  p.dk = reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()); p.dv = reinterpret_cast<__nv_bfloat16*>(dv.data_ptr());
  p.ldq = ldq; p.ldk = ldk; p.ldv = ldv; p.lddo = lddo; p.lddk = lddk; p.lddv = lddv; p.L = (int)L; p.H = (int)H;
  p.scl = (float)scl; p.sm_scale = (float)sm_scale;
  const int64_t cfg = (nst * 100 + bn) * 10 + minb;
#define DKV_CFG(NST, BN, MINB) \
  if (cfg == ((int64_t(NST)) * 100 + (BN)) * 10 + (MINB)) { launch_attn_bwd_dkv<a100::DkvCfg<NST, BN, MINB>>(p, (int)Z); return; }
  DKV_CFG(4, 32, 2) DKV_CFG(3, 32, 2) DKV_CFG(5, 32, 2) DKV_CFG(3, 32, 1) DKV_CFG(4, 32, 1)
#undef DKV_CFG
  TORCH_CHECK(false, "attn_bwd_dkv: unsupported schedule (nst, bn, minb)");
}

// bias_t [z H, L, L] = the transpose of each [L, L] plane of bias.
void bias_transpose(torch::Tensor bias, torch::Tensor bias_t, int64_t L) {
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kBFloat16 && bias.is_contiguous() && bias_t.scalar_type() == at::kBFloat16 && bias_t.is_contiguous() &&
              bias.numel() == bias_t.numel() && L % 32 == 0 && bias.numel() % (L * L) == 0, "bias_transpose: [zh, L, L] bf16 contiguous");
  const int64_t zh = bias.numel() / (L * L);
  TORCH_CHECK(zh <= 65535 && L / 32 <= 65535, "bias_transpose: grid");
  a100::bias_transpose_kernel<<<dim3((unsigned)(L / 32), (unsigned)(L / 32), (unsigned)zh), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()), reinterpret_cast<__nv_bfloat16*>(bias_t.data_ptr()), (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The backward's weights in its kernels' layouts (wgrad_sm80.cuh, bwd_pack_kernel): wq / wk / wv / wg [128, 128], wb [4, 128], wo [128, 128] bf16 and the
// LayerNorm scale (bf16 or fp32 [128]) -> wt [4, 128, 128] (W_p^T), wbt [128, 4], wot [128, 128] (Wo^T), gamma32 [128] fp32.
void bwd_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor wo, torch::Tensor gamma, torch::Tensor wt,
              torch::Tensor wbt, torch::Tensor wot, torch::Tensor gamma32) {
  const torch::Tensor* ws[6] = {&wq, &wk, &wv, &wg, &wb, &wo};
  a100::BwdPackParams p;
  for (int i = 0; i < 6; ++i) {
    TORCH_CHECK(ws[i]->is_cuda() && ws[i]->scalar_type() == at::kBFloat16 && ws[i]->is_contiguous() && ws[i]->numel() == (i == 4 ? 4 : 128) * 128,
                "bwd_pack: weights [128 | 4, 128] bf16 contiguous");
    p.w[i] = reinterpret_cast<const __nv_bfloat16*>(ws[i]->data_ptr());
  }
  p.w[6] = nullptr;
  TORCH_CHECK(gamma.is_contiguous() && gamma.numel() == 128 && (gamma.scalar_type() == at::kFloat || gamma.scalar_type() == at::kBFloat16), "bwd_pack: gamma [128] fp32 / bf16");
  TORCH_CHECK(wt.scalar_type() == at::kBFloat16 && wt.is_contiguous() && wt.numel() == 4 * 128 * 128 && wbt.scalar_type() == at::kBFloat16 && wbt.is_contiguous() &&
              wbt.numel() == 512 && wot.scalar_type() == at::kBFloat16 && wot.is_contiguous() && wot.numel() == 128 * 128 && gamma32.scalar_type() == at::kFloat &&
              gamma32.is_contiguous() && gamma32.numel() == 128, "bwd_pack: outputs");
  p.gamma = gamma.data_ptr(); p.gamma_bf16 = gamma.scalar_type() == at::kBFloat16;
  p.wt = reinterpret_cast<__nv_bfloat16*>(wt.data_ptr()); p.wbt = reinterpret_cast<__nv_bfloat16*>(wbt.data_ptr());
  p.wot = reinterpret_cast<__nv_bfloat16*>(wot.data_ptr()); p.gamma32 = gamma32.data_ptr<float>();
  const int total = 5 * 16384 + 512 + 128;
  a100::bwd_pack_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The parameters' gradients from G = D^T [xh | 1] ([516, 136] fp32): dwq / dwk / dwv / dwg [128, 128], dwb [4, 128] (bf16, written), dgamma / dbeta [128]
// (the LayerNorm parameters' dtype, written); gamma / beta: the parameters themselves.
void wgrad_finalize(torch::Tensor g, torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor dwq, torch::Tensor dwk,
                    torch::Tensor dwv, torch::Tensor dwg, torch::Tensor dwb, torch::Tensor gamma, torch::Tensor beta, torch::Tensor dgamma, torch::Tensor dbeta) {
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat && g.is_contiguous() && g.numel() == 516 * 136, "wgrad_finalize: g [516, 136] fp32");
  const torch::Tensor* ws[5] = {&wq, &wk, &wv, &wg, &wb};
  const torch::Tensor* ds[5] = {&dwq, &dwk, &dwv, &dwg, &dwb};
  a100::WgradParams p;
  p.g = g.data_ptr<float>();
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(ws[i]->scalar_type() == at::kBFloat16 && ws[i]->is_contiguous() && ws[i]->numel() == (i == 4 ? 4 : 128) * 128 && ds[i]->scalar_type() == at::kBFloat16 &&
                ds[i]->is_contiguous() && ds[i]->numel() == ws[i]->numel(), "wgrad_finalize: weights / gradients bf16 contiguous");
    p.w[i] = reinterpret_cast<const __nv_bfloat16*>(ws[i]->data_ptr()); p.dw[i] = reinterpret_cast<__nv_bfloat16*>(ds[i]->data_ptr());
  }
  const auto dt = gamma.scalar_type();
  TORCH_CHECK((dt == at::kFloat || dt == at::kBFloat16) && beta.scalar_type() == dt && dgamma.scalar_type() == dt && dbeta.scalar_type() == dt && gamma.is_contiguous() &&
              beta.is_contiguous() && dgamma.is_contiguous() && dbeta.is_contiguous() && gamma.numel() == 128 && beta.numel() == 128 && dgamma.numel() == 128 &&
              dbeta.numel() == 128, "wgrad_finalize: gamma / beta / dgamma / dbeta [128], one dtype (fp32 or bf16)");
  p.gamma = gamma.data_ptr(); p.beta = beta.data_ptr(); p.dgamma = dgamma.data_ptr(); p.dbeta = dbeta.data_ptr(); p.ln_bf16 = dt == at::kBFloat16;
  a100::wgrad_finalize_kernel<<<516 + 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compact_attention", &compact_attention);
  m.def("attn_fwd", &attn_fwd);
  m.def("attn_fwd_gate", &attn_fwd_gate);
  m.def("front", &front);
  m.def("front_pack", &front_pack);
  m.def("back", &back);
  m.def("back_bwd", &back_bwd);
  m.def("front_bwd", &front_bwd);
  m.def("attn_bwd_dq", &attn_bwd_dq);
  m.def("db_reduce", &db_reduce);
  m.def("attn_bwd_dkv", &attn_bwd_dkv);
  m.def("bias_transpose", &bias_transpose);
  m.def("bwd_pack", &bwd_pack);
  m.def("wgrad_finalize", &wgrad_finalize);
}
