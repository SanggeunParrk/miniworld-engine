// apb_rows_pdl_sm100.cuh -- B200 only (included by apb_rows_sm100.cu after the shared apb_rows.cu; no other build reads it).
// The PDL forms of the shared row kernels (2026-11-17): gate_rows_k, gate_bwd_k, gate_bwd_hl_k, qkv_bwd_k and ln_bwd_k copied unchanged
// from apb_rows.cu (the same statements, so the same bits) under a _pk name, each opening with APB_PDL_ENTRY (apb_rows_sm100.cu: griddepcontrol.wait, nothing read
// before it) + launch_dependents. Launched plainly both are no-ops; apb_rows_sm100.cu's pdl_go starts them as programmatic dependents
// of the previous kernel on the stream. The shared file stays byte-identical to the other cards' builds.
#pragma once
namespace {
// og = sigmoid(g) o over W = 128 V columns (g: the qkvg row's last quarter)
template <int V>
__global__ void __launch_bounds__(NT) gate_rows_pk(const float* __restrict__ O, const bf* __restrict__ QKVG, bf* __restrict__ OG, int M) {
  APB_PDL_ENTRY();
  constexpr int WD = 128 * V;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  float4 o[V], g[V];
  ldv<V>(O + r * WD, o); ldv<V>(QKVG + r * 4 * WD + 3 * WD, g);
#pragma unroll
  for (int k = 0; k < V; ++k) o[k] = mul4(sig4(g[k]), o[k]);
  stv<V>(OG + r * WD, o);
}

// og = sigmoid(g) o: dO = dog s (bf16, the core's input), D = rowsum_head(dO o) as [NH, M], dg = dog o s (1 - s). A head is
// HL lanes (W / NH / (4 V)).
template <int V, int HL>
__global__ void __launch_bounds__(NT) gate_bwd_pk(const bf* __restrict__ DOG, const float* __restrict__ O, const bf* __restrict__ QKVG,
    bf* __restrict__ DOB, float* __restrict__ DD, bf* __restrict__ DQKVG, int M) {
  APB_PDL_ENTRY();
  constexpr int WD = 128 * V;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  if (r >= M) return;
  const int lane = threadIdx.x & 31;
  float4 dog[V], o[V], s[V], d_o[V], dg[V];
  ldv<V>(DOG + r * WD, dog); ldv<V>(O + r * WD, o); ldv<V>(QKVG + r * 4 * WD + 3 * WD, s);
#pragma unroll
  for (int k = 0; k < V; ++k) {
    s[k] = sig4(s[k]);
    d_o[k] = mul4(dog[k], s[k]);
    const float4 t = mul4(dog[k], o[k]);
    dg[k] = make_float4(t.x * s[k].x * (1.f - s[k].x), t.y * s[k].y * (1.f - s[k].y), t.z * s[k].z * (1.f - s[k].z),
                        t.w * s[k].w * (1.f - s[k].w));
  }
  stv<V>(DOB + r * WD, d_o);
  stv<V>(DQKVG + r * 4 * WD + 3 * WD, dg);
  float dsum = dotv<V>(d_o, o);
#pragma unroll
  for (int o_ = 1; o_ < HL; o_ <<= 1) dsum += __shfl_xor_sync(0xffffffffu, dsum, o_);
  if (lane % HL == 0) DD[(long)(lane / HL) * M + r] = dsum;
}

// gate_bwd with a head per lane (24 x 16, 12 x 32; W = 384): lane h < NH owns head h's HC columns, the rest idle
template <int NH, int HC>
__global__ void __launch_bounds__(NT) gate_bwd_hl_pk(const bf* __restrict__ DOG, const float* __restrict__ O, const bf* __restrict__ QKVG,
    bf* __restrict__ DOB, float* __restrict__ DD, bf* __restrict__ DQKVG, int M) {
  APB_PDL_ENTRY();
  constexpr int WD = NH * HC, V = HC / 4;
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (r >= M || lane >= NH) return;
  const int c = HC * lane;
  float4 dog[V], o[V], s[V], d_o[V], dg[V];
#pragma unroll
  for (int k = 0; k < V; ++k) {
    dog[k] = V4<bf>::load(DOG + r * WD + c + 4 * k); o[k] = V4<float>::load(O + r * WD + c + 4 * k);
    s[k] = sig4(V4<bf>::load(QKVG + r * 4 * WD + 3 * WD + c + 4 * k));
    d_o[k] = mul4(dog[k], s[k]);
    const float4 t = mul4(dog[k], o[k]);
    dg[k] = make_float4(t.x * s[k].x * (1.f - s[k].x), t.y * s[k].y * (1.f - s[k].y), t.z * s[k].z * (1.f - s[k].z),
                        t.w * s[k].w * (1.f - s[k].w));
    V4<bf>::store(DOB + r * WD + c + 4 * k, d_o[k]);
    V4<bf>::store(DQKVG + r * 4 * WD + 3 * WD + c + 4 * k, dg[k]);
  }
  DD[(long)lane * M + r] = dotv<V>(d_o, o);
}

// dq (NCH key-chunk partials [NCH, M, W], added in chunk order: deterministic) | dk | dv fp32 [M, W] -> dqkvg[:, 0:3W] bf16;
// ABQ[c] += sum_r dq (bf16-rounded, what the GEMMs see)
template <int V>
__global__ void __launch_bounds__(NT) qkv_bwd_pk(const float* __restrict__ DQ, const float* __restrict__ DK, const float* __restrict__ DV,
    bf* __restrict__ DQKVG, float* __restrict__ ABQ, int M, int NCH, int acc_stride) {
  APB_PDL_ENTRY();
  constexpr int WD = 128 * V;
  __shared__ __align__(16) float red[RW * WD];
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  float4 v[V] = {};
  if (r < M) {
    ldv<V>(DQ + r * WD, v);
    for (int c = 1; c < NCH; ++c) {
      float4 p[V];
      ldv<V>(DQ + ((long)c * M + r) * WD, p);
#pragma unroll
      for (int k = 0; k < V; ++k) v[k] = add4(v[k], p[k]);
    }
#pragma unroll
    for (int k = 0; k < V; ++k) v[k] = rnd<bf>(v[k]);
    stv<V>(DQKVG + r * 4 * WD, v);
    float4 t[V];
    ldv<V>(DK + r * WD, t); stv<V>(DQKVG + r * 4 * WD + WD, t);
    ldv<V>(DV + r * WD, t); stv<V>(DQKVG + r * 4 * WD + 2 * WD, t);
  }
  colsum_add<V>(v, red, ABQ, acc_stride);
}

// x^ = LN(x); dx = dy + rstd (dxa w - mean(dxa w) - x^ mean(dxa w x^)); ALNW += dxa x^, ALNB += dxa
template <typename XT, typename OT, int V, typename AT = bf>
__global__ void __launch_bounds__(NT) ln_bwd_pk(const AT* __restrict__ DXA, const XT* __restrict__ X, const float2* __restrict__ ST,
    const float* __restrict__ LW, const OT* __restrict__ DY, OT* __restrict__ DX, float* __restrict__ ALNW, float* __restrict__ ALNB, int M,
    int acc_stride) {
  APB_PDL_ENTRY();
  constexpr int D = 128 * V;
  __shared__ __align__(16) float red[RW * D];
  const long r = (long)blockIdx.x * RW + (threadIdx.x >> 5);
  float4 aw[V] = {}, ab[V] = {};
  if (r < M) {
    float4 dxa[V], xh[V], w[V], g[V], dy[V];
    ldv<V>(DXA + r * D, dxa); ldv<V>(X + r * D, xh); ldv<V>(LW, w); ldv<V>(DY + r * D, dy);
    const float2 st = ST[r];
#pragma unroll
    for (int k = 0; k < V; ++k) {
      xh[k] = make_float4((xh[k].x - st.x) * st.y, (xh[k].y - st.x) * st.y, (xh[k].z - st.x) * st.y, (xh[k].w - st.x) * st.y);
      g[k] = mul4(dxa[k], w[k]);
      aw[k] = mul4(dxa[k], xh[k]);
      ab[k] = dxa[k];
    }
    const float m1 = warp_sum(sumv<V>(g)) * (1.f / D), m2 = warp_sum(dotv<V>(g, xh)) * (1.f / D);
#pragma unroll
    for (int k = 0; k < V; ++k)
      dy[k] = make_float4(dy[k].x + st.y * (g[k].x - m1 - xh[k].x * m2), dy[k].y + st.y * (g[k].y - m1 - xh[k].y * m2),
                          dy[k].z + st.y * (g[k].z - m1 - xh[k].z * m2), dy[k].w + st.y * (g[k].w - m1 - xh[k].w * m2));
    stv<V>(DX + r * D, dy);
  }
  colsum_add<V>(aw, red, ALNW, acc_stride);
  __syncthreads();
  colsum_add<V>(ab, red, ALNB, acc_stride);
}
}  // namespace
