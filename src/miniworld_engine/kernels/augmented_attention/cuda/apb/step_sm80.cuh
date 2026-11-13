// step_sm80.cuh -- the A100 AttentionPairBias steps in C++: one call per inference step, per training forward and per training backward.
// Included by apb_rows.cu under -DAPB_SM80, at the end of its anonymous namespace (everything it calls is defined above it).
//
// The step is ~15 launches and ~20 small tensor allocations; written in Python around the per-kernel entry points it costs ~1.2 ms of host time a training step
// (0.2 ms of GPU time at L = 384) -- the pair passes, the attention core and the GEMMs wait for the interpreter. Here the same sequence runs as one
// C++ function: the row kernels and the pair passes through their own host entry points, the attention core through the launchers below (the aa80:: kernels of
// kernels/augmented_attention/cuda/sm80, built into this extension for the head dims the module uses), the GEMMs through ATen (cuBLAS).
//
// Under a CUDA-graph capture (``overlap``) the memory-bound pair passes run on streams of their own next to the small kernels around them; the fork / join events
// are made once (the Python ``Stream.wait_stream`` makes an event per call).

constexpr float AA_L2E = 1.4426950408889634f;
constexpr float APB_MASKED = -1e4f;                                           // masked keys of the training step (natural units)

// ------------------------------------------------------------------------------------------------------------------------------ streams
struct SideStreams {
  c10::cuda::CUDAStream s[3];
  cudaEvent_t ev[8];
};
SideStreams& side_streams(int dev) {
  static SideStreams* table[64] = {};
  SideStreams*& t = table[dev & 63];
  if (t == nullptr) {
    t = new SideStreams{{c10::cuda::getStreamFromPool(false, dev), c10::cuda::getStreamFromPool(false, dev), c10::cuda::getStreamFromPool(false, dev)}, {}};
    for (int i = 0; i < 8; ++i) C10_CUDA_CHECK(cudaEventCreateWithFlags(&t->ev[i], cudaEventDisableTiming));
  }
  return *t;
}
// ``waiter`` waits for what ``on`` has queued so far (event ``slot``)
void stream_wait(int dev, int slot, cudaStream_t waiter, cudaStream_t on) {
  cudaEvent_t e = side_streams(dev).ev[slot];
  C10_CUDA_CHECK(cudaEventRecord(e, on));
  C10_CUDA_CHECK(cudaStreamWaitEvent(waiter, e, 0));
}

// ------------------------------------------------------------------------------------------------------------------------ attention launchers
template <class G>
void sa_launch_fwd(const aa80::FwdParams& p, cudaStream_t st) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) { C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); set[dev] = true; }
  aa80::attn_fwd_kernel<G><<<dim3(p.L / G::BQ, 1, p.H), G::NTHR, G::SMEM, st>>>(p);
}
template <class G>
void sa_launch_dq(const aa80::DqParams& p, cudaStream_t st) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) { C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dq_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); set[dev] = true; }
  aa80::attn_bwd_dq_kernel<G><<<dim3(p.L / G::BQ, 1, p.H), G::NTHR, G::SMEM, st>>>(p);
}
template <class G>
void sa_launch_dkv(const aa80::DkvParams& p, cudaStream_t st) {
  static bool set[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!set[dev]) { C10_CUDA_CHECK(cudaFuncSetAttribute(aa80::attn_bwd_dkv_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); set[dev] = true; }
  aa80::attn_bwd_dkv_kernel<G><<<dim3(p.L / G::BK, 1, p.H), G::NTHR, G::SMEM, st>>>(p);
}

// one sample; the schedules of the Python launchers (``sm80.py``): warps 8 / 4 / 2 -> the usual (nkv 4, 2 CTAs per SM), (nkv 4, 2), (nkv 4, 4)
#define SA_FWD(HD, MODE)                                                                                                                                        \
  if (nw == 8) sa_launch_fwd<aa80::FwdCfg<HD, 1, 4, 2, MODE, 0, 8>>(p, st);                                                                                     \
  else if (nw == 4) sa_launch_fwd<aa80::FwdCfg<HD, 1, 4, 2, MODE, 0, 4>>(p, st);                                                                                \
  else sa_launch_fwd<aa80::FwdCfg<HD, 1, 4, 4, MODE, 0, 2>>(p, st);
template <int MODE>
void sa_fwd(int hd, int nw, const aa80::FwdParams& p, cudaStream_t st, int bn = 32) {
  TORCH_CHECK(nw == 2 || nw == 4 || nw == 8, "attention core: 2, 4 or 8 warps");
  if (bn == 64 && MODE == aa80::MODE_GATE && nw < 8) {                      // 64-key tiles: two independent chains per barrier (registers allow one CTA of 2 per SM)
#define SA_FWD64(HD) if (nw == 4) sa_launch_fwd<aa80::FwdCfg<HD, 1, 4, 2, MODE, 0, 4, 64>>(p, st); else sa_launch_fwd<aa80::FwdCfg<HD, 1, 4, 2, MODE, 0, 2, 64>>(p, st);
    if (hd == 48) { SA_FWD64(48) } else if (hd == 32) { SA_FWD64(32) } else if (hd == 16) { SA_FWD64(16) } else TORCH_CHECK(false, "head dim");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (hd == 48) { SA_FWD(48, MODE) } else if (hd == 32) { SA_FWD(32, MODE) } else if (hd == 16) { SA_FWD(16, MODE) }
  else TORCH_CHECK(false, "attention core: head dim 16, 32 or 48");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#define SA_DQ(HD)                                                                                                                                               \
  if (nw == 8) sa_launch_dq<aa80::DqCfg<HD, 1, 3, 2, 1, 0, 8>>(p, st);                                                                                          \
  else if (nw == 4) sa_launch_dq<aa80::DqCfg<HD, 1, 3, 2, 1, 0, 4>>(p, st);                                                                                     \
  else sa_launch_dq<aa80::DqCfg<HD, 1, 3, 4, 1, 0, 2>>(p, st);
void sa_dq(int hd, int nw, const aa80::DqParams& p, cudaStream_t st) {
  TORCH_CHECK(nw == 2 || nw == 4 || nw == 8, "attention core: 2, 4 or 8 warps");
  if (hd == 48) { SA_DQ(48) } else if (hd == 32) { SA_DQ(32) } else if (hd == 16) { SA_DQ(16) }
  else TORCH_CHECK(false, "attention core: head dim 16, 32 or 48");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// the key side: 8 warps keep the module's nst (3; 4 at head dim 32), 4 warps 3 stages with 2 CTAs per SM, 2 warps 3 stages with 4
#define SA_DKV(HD, NST8)                                                                                                                                        \
  if (nw == 8) sa_launch_dkv<aa80::DkvCfg<HD, NST8, 2, 1, 0, 8>>(p, st);                                                                                        \
  else if (nw == 4) sa_launch_dkv<aa80::DkvCfg<HD, 3, 2, 1, 0, 4>>(p, st);                                                                                      \
  else sa_launch_dkv<aa80::DkvCfg<HD, 3, 4, 1, 0, 2>>(p, st);
void sa_dkv(int hd, int nw, const aa80::DkvParams& p, cudaStream_t st) {
  TORCH_CHECK(nw == 2 || nw == 4 || nw == 8, "attention core: 2, 4 or 8 warps");
  if (hd == 48) { SA_DKV(48, 3) } else if (hd == 32) { SA_DKV(32, 4) } else if (hd == 16) { SA_DKV(16, 3) }
  else TORCH_CHECK(false, "attention core: head dim 16, 32 or 48");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

const __nv_bfloat16* bfp(const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
__nv_bfloat16* bfw(at::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }

// the gated inference core, in place over q: sigmoid(g) o written into the q columns of qkvg [Lp, 4 W] (W = H hd)
// klen (int32 [1], ``kernels._key_len``) or null: the key tiles past the row's last attended key are skipped (the same bits)
void step_core_gate(at::Tensor& qkvg, const at::Tensor& bias, int64_t Lp, int64_t H, int64_t hd, int nw, cudaStream_t st, int bn,
                    const int* klen = nullptr) {
  const int64_t W = H * hd, ld = qkvg.stride(0);
  aa80::FwdParams p{};
  p.q = bfp(qkvg); p.k = p.q + W; p.v = p.q + 2 * W; p.bias = bfp(bias); p.gate = p.q + 3 * W;
  p.out = qkvg.data_ptr(); p.lse = nullptr; p.kpen = nullptr;
  p.ldq = p.ldk = p.ldv = p.ldg = p.ldo = ld; p.ss = Lp; p.kps = 0; p.L = (int)Lp; p.H = (int)H; p.scl = 1.f; p.bscl = 1.f;
  p.klen = klen; p.kls = 0;
  sa_fwd<aa80::MODE_GATE>((int)hd, nw, p, st, bn);
}

// ------------------------------------------------------------------------------------------------------------------------------ inference
// wpack [4 W, D] | bvec [4 W] | wf [H, 128] | wo [D, W] (the padded Wo, or to_out.weight) | lnw, lnb fp32: the cached packs (q and its bias scaled into exp2 units, Wf in exp2 units)
at::Tensor apb_inference(at::Tensor single, at::Tensor pair, c10::optional<at::Tensor> mask, at::Tensor wpack, at::Tensor bvec, at::Tensor wf, at::Tensor wo, at::Tensor lnw,
                         at::Tensor lnb, double eps_s, double eps_p, bool overlap, int64_t warps, int64_t bn,
                         c10::optional<at::Tensor> klen) {
  TORCH_CHECK(!klen.has_value() || (klen->is_cuda() && klen->scalar_type() == at::kInt && klen->numel() == 1), "apb_inference: klen int32 [1]");
  const int64_t L = single.size(1), D = single.size(2), H = wf.size(0), W = wo.size(1), hd = W / H, Lp = (L + 127) / 128 * 128;
  const at::cuda::CUDAGuard gd(single.device());
  const int dev = single.get_device();
  const auto st = at::cuda::getCurrentCUDAStream();
  at::Tensor x = single.view({L, D});
  auto side = side_streams(dev).s[0];
  if (overlap) stream_wait(dev, 0, side, st);                                      // fork first: the pair pass depends on nothing the step makes
  at::Tensor xa = Lp == L ? at::empty_like(x) : at::zeros({Lp, D}, x.options());
  at::Tensor y = at::empty_like(x);                                              // the residual seed: to_out accumulates onto it
  ln_rows(x, lnw, lnb, xa, c10::nullopt, eps_s, y);
  at::Tensor qkvg = at::addmm(bvec, xa, wpack.t());
  at::Tensor bias = at::empty({H, Lp, Lp}, x.options());
  at::Tensor pair2 = pair.view({L * L, DP});
  if (overlap) {                                                                  // the pair pass next to ln_rows and the GEMM: its own stream
    c10::cuda::CUDAStreamGuard g(side);
    pair_bias_fwd(pair2, wf, mask, bias, L, eps_p, -1e30);
    g.reset_stream(st);
    stream_wait(dev, 1, st, side);
    bias.record_stream(st);
  } else {
    pair_bias_fwd(pair2, wf, mask, bias, L, eps_p, -1e30);
  }
  step_core_gate(qkvg, bias, Lp, H, hd, (int)warps, st, (int)bn, klen.has_value() ? klen->data_ptr<int>() : nullptr);
  y.addmm_(qkvg.slice(0, 0, L).slice(1, 0, W), wo.t());
  return y.view({1, L, D});
}

// ------------------------------------------------------------------------------------------------------------------------ training forward
// leaves: the parameters as they are (bf16 or fp32 masters); returns [y, xa, st, qkvg, bias, o, lse, og, wpack, wop, (wf_nat)]
std::vector<at::Tensor> apb_train_forward(at::Tensor single, at::Tensor pair, c10::optional<at::Tensor> mask, at::Tensor lnw, at::Tensor lnb, at::Tensor wq, at::Tensor bq,
                                          at::Tensor wk, at::Tensor wv, at::Tensor wg, at::Tensor wo, at::Tensor lnpw, at::Tensor wb, double eps_s, double eps_p,
                                          bool save_natural, bool overlap, int64_t warps) {
  const int64_t L = single.size(1), D = single.size(2), H = wb.size(0);
  const Geo G = geo(H, D);
  const int64_t W = width(G), hd = G.dhp, Lp = (L + 127) / 128 * 128;   // hd: the head width in memory (24 -> 32); the softmax scale is the real head dim's
  const double scale = std::sqrt((double)G.dh);
  const at::cuda::CUDAGuard gd(single.device());
  const int dev = single.get_device();
  const auto st = at::cuda::getCurrentCUDAStream();
  at::Tensor x = single.view({L, D});
  const auto bfo = x.options();                                                   // bf16 activations
  auto side = side_streams(dev).s[0];
  if (overlap) stream_wait(dev, 0, side, st);                                      // fork first: Wf is formed inside the pair kernel, nothing to wait for
  at::Tensor wpack = at::empty({4 * W, D}, bfo), bvec = at::empty({4 * W}, bfo), wf = at::empty({H, DP}, bfo), wop = at::empty({D, W}, bfo);
  c10::optional<at::Tensor> wf_nat;
  if (save_natural) wf_nat = at::empty({H, DP}, bfo);
  prep_train(wq, bq, wk, wv, wg, wo, wb, lnpw, wpack, bvec, wop, wf, 1.0, scale, wf_nat);
  at::Tensor xa = Lp == L ? at::empty_like(x) : at::zeros({Lp, D}, bfo);
  at::Tensor stat = at::empty({L, 2}, bfo.dtype(at::kFloat));
  at::Tensor y = at::empty_like(x);
  ln_rows(x, lnw, lnb, xa, stat, eps_s, y);
  at::Tensor qkvg = at::addmm(bvec, xa, wpack.t());
  at::Tensor bias = at::empty({H, Lp, Lp}, bfo);
  at::Tensor pair2 = pair.view({L * L, DP});
  if (overlap) {
    c10::cuda::CUDAStreamGuard g(side);
    pair_bias_fwd_raw(pair2, wb, lnpw, scale, mask, bias, L, eps_p, (double)APB_MASKED * scale);
    g.reset_stream(st);
    stream_wait(dev, 1, st, side);
    bias.record_stream(st);
  } else {
    pair_bias_fwd_raw(pair2, wb, lnpw, scale, mask, bias, L, eps_p, (double)APB_MASKED * scale);
  }
  at::Tensor o = at::empty({Lp, W}, bfo), lse = at::empty({1, H, Lp}, bfo.dtype(at::kFloat));
  {
    const int64_t ld = qkvg.stride(0);
    aa80::FwdParams p{};
    p.q = bfp(qkvg); p.k = p.q + W; p.v = p.q + 2 * W; p.bias = bfp(bias); p.gate = p.q;
    p.out = o.data_ptr(); p.lse = lse.data_ptr<float>(); p.kpen = nullptr;
    p.ldq = p.ldk = p.ldv = p.ldg = ld; p.ldo = W; p.ss = Lp; p.kps = 0; p.L = (int)Lp; p.H = (int)H;
    p.scl = (float)((1.0 / scale) * 1.4426950408889634); p.bscl = (float)1.4426950408889634;   // in double, as the Python launcher
    sa_fwd<aa80::MODE_PLAIN>((int)hd, (int)warps, p, st);
  }
  at::Tensor og = at::empty({L, W}, bfo);
  gate_rows_bf(o, qkvg, og, L);
  y.addmm_(og, wop.t());
  std::vector<at::Tensor> out{y.view({1, L, D}), xa, stat, qkvg, bias, o, lse, og, wpack, wop};
  if (save_natural) out.push_back(*wf_nat);
  return out;
}

// ----------------------------------------------------------------------------------------------------------------------- training backward
// saved: xa, st, qkvg, bias, o, lse, og, wpack, wop, (wf_nat); refs: the 11 parameters (their dtypes and shapes are the gradients'); typed: gradients in the parameters' dtypes
// (else fp32). Returns [dx, dz, g0 .. g10].
std::vector<at::Tensor> apb_train_backward(at::Tensor single, at::Tensor pair, at::Tensor lnw, at::Tensor lnpw, at::Tensor wb, double eps_s, double eps_p,
                                           std::vector<at::Tensor> saved, at::Tensor dy, std::vector<at::Tensor> refs, bool typed, bool overlap, bool pair_stream,
                                           int64_t pair_grid, int64_t warps) {
  const int64_t L = single.size(1), D = single.size(2), H = wb.size(0);
  const Geo G = geo(H, D);
  const int64_t W = width(G), hd = G.dhp, Lp = (L + 127) / 128 * 128;
  const at::cuda::CUDAGuard gd(single.device());
  const int dev = single.get_device();
  const auto st = at::cuda::getCurrentCUDAStream();
  at::Tensor &xa = saved[0], &stat = saved[1], &qkvg = saved[2], &bias = saved[3], &o = saved[4], &lse = saved[5], &og = saved[6], &wpack = saved[7], &wop = saved[8];
  at::Tensor wf_nat = saved.size() > 9 ? saved[9] : (wb.to(at::kFloat) * lnpw.to(at::kFloat).unsqueeze(0)).to(at::kBFloat16);
  at::Tensor x = single.view({L, D});
  at::Tensor dy2 = dy.contiguous().view({L, D});
  const auto bfo = x.options(), f32o = bfo.dtype(at::kFloat);
  at::Tensor acc = at::empty({acc_n((int)D, (int)H, (int)W)}, f32o).zero_();       // dlnw | dlnb | dbq | dWf | head sums, added by the kernels
  auto sideA = side_streams(dev).s[1], sideB = side_streams(dev).s[2], sideP = side_streams(dev).s[0];
  at::Tensor dwo, dwp, dz;
  if (overlap) stream_wait(dev, 2, sideA, st);                                    // the weight-gradient GEMMs run beside the critical chain
  at::Tensor dog = at::mm(dy2, wop);                                              // [L, W]: zero in the pads (Wo's pad columns are zero)
  {
    c10::cuda::CUDAStreamGuard g(overlap ? sideA : st);
    dwo = at::mm(dy2.t(), og, at::kFloat);                                        // [D, W]; finalize keeps the real columns
  }
  at::Tensor dob = Lp == L ? at::empty({Lp, W}, bfo) : at::zeros({Lp, W}, bfo);    // the padded queries' gradient is 0
  at::Tensor dqkvg = at::empty({L, 4 * W}, bfo);
  gate_bwd_bf(dog, o, qkvg, dob, dqkvg, L);
  // the attention backward: delta, dq (+ the bias gradient partial, bf16, one sample: the gradient itself), the transposed bias, dk / dv
  at::Tensor delta = at::empty({1, H, Lp}, f32o);
  {
    const int64_t lddo = dob.stride(0), ldo = o.stride(0);
    dim3 grid((unsigned)((Lp + 127) / 128), (unsigned)H);
    if (hd == 48) aa80::attn_delta_kernel<48, __nv_bfloat16><<<grid, 128, 0, st>>>(bfp(dob), bfp(o), delta.data_ptr<float>(), lddo, ldo, (int)Lp, (int)H, Lp);
    else if (hd == 32) aa80::attn_delta_kernel<32, __nv_bfloat16><<<grid, 128, 0, st>>>(bfp(dob), bfp(o), delta.data_ptr<float>(), lddo, ldo, (int)Lp, (int)H, Lp);
    else aa80::attn_delta_kernel<16, __nv_bfloat16><<<grid, 128, 0, st>>>(bfp(dob), bfp(o), delta.data_ptr<float>(), lddo, ldo, (int)Lp, (int)H, Lp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const double sm_d = 1.0 / std::sqrt((double)G.dh);                               // in double, as the Python launcher
  const float sm_scale = (float)sm_d, scl_f = (float)(sm_d * 1.4426950408889634);
  const int nw = (int)warps;
  at::Tensor dq = at::empty({Lp, W}, bfo), dk = at::empty({Lp, W}, bfo), dv = at::empty({Lp, W}, bfo);
  at::Tensor dbp = at::empty({1, H, Lp, Lp}, bfo);
  const int64_t ld = qkvg.stride(0);
  {
    aa80::DqParams p{};
    p.q = bfp(qkvg); p.k = p.q + W; p.v = p.q + 2 * W; p.dov = bfp(dob); p.bias = bfp(bias); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
    p.dq = dq.data_ptr(); p.dbp = bfw(dbp); p.kpen = nullptr; p.kps = 0;
    p.ldq = p.ldk = p.ldv = ld; p.lddo = dob.stride(0); p.lddq = W; p.ss = Lp; p.L = (int)Lp; p.H = (int)H;
    p.scl = scl_f; p.bscl = AA_L2E; p.sm_scale = sm_scale;
    sa_dq((int)hd, nw, p, st);
  }
  // the pair gradient needs only the bias gradient: dbp[0] is it (bf16, as the kernel's partial). Its own stream from here, next to the key side and the GEMMs.
  at::Tensor db = dbp.select(0, 0);
  if (pair_stream) {
    stream_wait(dev, 3, sideP, st);
    c10::cuda::CUDAStreamGuard g(sideP);
    dz = at::empty_like(pair.view({L * L, DP}));
    pair_bias_bwd_grid(pair.view({L * L, DP}), db, wf_nat, dz, acc, L, eps_p, D, pair_grid);
  }
  at::Tensor bias_t = at::empty_like(bias);
  {
    dim3 grid((unsigned)(Lp / 32), (unsigned)(Lp / 32), (unsigned)H);
    aa80::bias_transpose_kernel<<<grid, 256, 0, st>>>(bfp(bias), bfw(bias_t), (int)Lp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  {
    aa80::DkvParams p{};
    p.q = bfp(qkvg); p.k = p.q + W; p.v = p.q + 2 * W; p.dov = bfp(dob); p.bias_t = bfp(bias_t); p.lse = lse.data_ptr<float>(); p.delta = delta.data_ptr<float>();
    p.dk = dk.data_ptr(); p.dv = dv.data_ptr(); p.kpen = nullptr; p.kps = 0;
    p.ldq = p.ldk = p.ldv = ld; p.lddo = dob.stride(0); p.lddk = W; p.lddv = W; p.ss = Lp; p.L = (int)Lp; p.H = (int)H;
    p.scl = scl_f; p.bscl = AA_L2E; p.sm_scale = sm_scale;
    sa_dkv((int)hd, nw, p, st);
  }
  qkv_bwd_bf(dq, dk, dv, dqkvg, acc, D, L);
  dq = at::Tensor(); dk = at::Tensor(); dv = at::Tensor();
  if (overlap) stream_wait(dev, 4, sideB, st);
  {
    c10::cuda::CUDAStreamGuard g(overlap ? sideB : st);
    dwp = at::mm(dqkvg.t(), xa.slice(0, 0, L), at::kFloat);
  }
  at::Tensor dxa = at::mm(dqkvg, wpack);
  at::Tensor dx = at::empty_like(x);
  ln_bwd(dxa, x, stat, lnw, dy2, dx, acc);
  if (!pair_stream) {
    dz = at::empty_like(pair.view({L * L, DP}));
    pair_bias_bwd_grid(pair.view({L * L, DP}), db, wf_nat, dz, acc, L, eps_p, D, 0);
  }
  if (overlap) {
    stream_wait(dev, 5, st, sideA); stream_wait(dev, 6, st, sideB);
    dwo.record_stream(st); dwp.record_stream(st);
  }
  if (pair_stream) { stream_wait(dev, 7, st, sideP); dz.record_stream(st); }
  std::vector<at::Tensor> outs;
  for (const auto& r : refs) outs.push_back(at::empty(r.sizes(), r.options().dtype(typed ? r.scalar_type() : at::kFloat)));
  finalize_typed(dwp, dwo, acc, wb, lnpw, outs);
  std::vector<at::Tensor> ret{dx.view(single.sizes()), dz.view(pair.sizes())};
  for (auto& t : outs) ret.push_back(t);
  return ret;
}
