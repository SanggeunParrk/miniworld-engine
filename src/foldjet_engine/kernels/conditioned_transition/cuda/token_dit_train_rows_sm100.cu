// token_dit_train_rows_sm100.cu -- the B200 (sm_100a) build of token_dit_train_rows.cu (2026-11-17). The shared source (the H100 / A100 /
// RTX 5090 builds) is included unchanged; its module body becomes a function this file calls, and three entry points are replaced by the
// B200's own kernels below (same Python names, so the callers do not branch on the card):
//   pair_bias      one more argument, ``masked``: the value of the masked keys (-inf for the training blocks, -1e30 for the B200 fp32
//                  inference bias, whose fully masked rows stay finite) -- pair_bias_m_k
//   train_pack90   UNR (8) 4-element vectors per thread, every vector's loads issued before the first store: one vector per thread made
//   train_final90  the ~10 M element pack ~10000 short-lived blocks, bound by block turnover (15-18 us) -- train_pack100_k / train_final100_k
//   unfold_lnw     the cond-LN weight gradients as per-block partial rows added in block order (the shared kernel adds them with float
//                  atomics: RoseTTAFold3's ln_cond gradients differed run to run) -- unfold_lnw100_k / lnw_reduce100_k
#include <torch/extension.h>

#include <limits>

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, var) static void tdt_shared_bindings(pybind11::module_& var)
#include "token_dit_train_rows.cu"
#pragma pop_macro("PYBIND11_MODULE")

namespace {

// ---- pair_bias with the masked keys' value (the shared pair_bias_k, NEG in place of -inf) ----
template <typename ZT, typename OT>
__global__ void __launch_bounds__(256) pair_bias_m_k(const ZT* __restrict__ Z, long sz, const float* __restrict__ WF,
    const bool* __restrict__ MASK, OT* __restrict__ BIAS, float2* __restrict__ PST, long R, int L, float eps,
    const float* __restrict__ EX, float EXS, float NEG) {
  constexpr int S = PB_S, C4 = DPW / 4;
  extern __shared__ __align__(16) float pb_smem[];
  float* wf = pb_smem;                                                     // [HPAD][S]
  float* ph = pb_smem + HPAD * S;                                          // [PT][S]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
  for (int i = threadIdx.x; i < HPAD * C4; i += 256) {
    const int h = i / C4, c4 = i % C4;
    *reinterpret_cast<float4*>(wf + h * S + 4 * c4) = h < NHH ? reinterpret_cast<const float4*>(WF)[h * C4 + c4] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  const long ntile = (R + PT - 1) / PT;
  for (long tl = blockIdx.x; tl < ntile; tl += gridDim.x) {
    const long r0 = tl * PT;
    __syncthreads();                                                       // wf in; the previous tile's LN(z) consumed
    for (int j = warp; j < PT; j += 8) {
      const long r = r0 + j;
      float4 xh[NQ];
#pragma unroll
      for (int q = 0; q < NQ; ++q) xh[q] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (r < R) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) xh[q] = V4<ZT>::load(Z + r * sz + q * 128 + lane * 4);
        float s = sum4(xh[0]);
#pragma unroll
        for (int q = 1; q < NQ; ++q) s += sum4(xh[q]);
        const float mean = warp_sum(s) / (float)DPW;
        float ss = 0.f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
          xh[q] = make_float4(xh[q].x - mean, xh[q].y - mean, xh[q].z - mean, xh[q].w - mean);
          ss = q == 0 ? sum4(mul4(xh[q], xh[q])) : ss + sum4(mul4(xh[q], xh[q]));
        }
        const float rstd = rsqrtf(warp_sum(ss) / (float)DPW + eps);
#pragma unroll
        for (int q = 0; q < NQ; ++q) xh[q] = make_float4(xh[q].x * rstd, xh[q].y * rstd, xh[q].z * rstd, xh[q].w * rstd);
        if (lane == 0) PST[r] = make_float2(mean, rstd);
      }
#pragma unroll
      for (int q = 0; q < NQ; ++q) *reinterpret_cast<float4*>(ph + j * S + q * 128 + lane * 4) = xh[q];
    }
    __syncthreads();
    // bias^T [32 rows x HPAD] = LN(z) [32 x DPW] Wf^T [DPW x HPAD]: (row tile, head tile) items, one per warp
    constexpr int NIT = 2 * (HPAD / 8);
    if (warp < NIT) {
      const int mt = warp & 1, nt = warp >> 1;
      const float* pa = ph + (mt * 16 + g) * S + t;
      const float* pb = wf + (nt * 8 + g) * S + t;
      float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k0 = 0; k0 < DPW; k0 += 8) mma_tf32(d, pa[k0], pa[8 * S + k0], pa[k0 + 4], pa[8 * S + k0 + 4], pb[k0], pb[k0 + 4]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const long r = r0 + mt * 16 + g + (e >= 2 ? 8 : 0);
        const int h = nt * 8 + 2 * t + (e & 1);
        if (r < R && h < NHH) put_bias(BIAS + (long)h * R + r, (MASK != nullptr && !MASK[r % L]) ? NEG : d[e] + (EX != nullptr ? EX[r] * EXS : 0.f));   // r = query L + key
      }
    }
  }
}

// z [R, DPW] (fp32 or bf16 rows), wf [NHH, DPW] fp32, mask [L] bool or none -> bias [NHH, R] (head-major, masked keys ``masked``),
// pst [R, 2] (mean, rstd)
void pair_bias_m(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, at::Tensor bias, at::Tensor pst, int64_t L, double eps,
                 c10::optional<at::Tensor> extra, double extra_scale, double masked) {
  const int64_t R = z.size(0);
  TORCH_CHECK(z.size(1) == DPW && z.stride(1) == 1, "pair_bias: z [R, d_pair] (the build's TD_DP)");
  f32c(wf, "wf"); f32c(pst, "pst");
  TORCH_CHECK(bias.is_contiguous() && (bias.scalar_type() == at::kFloat || bias.scalar_type() == at::kBFloat16), "pair_bias: bias fp32 or bf16");
  if (extra) { f32c(*extra, "extra"); TORCH_CHECK(extra->numel() == R, "pair_bias: extra [L, L] fp32"); }
  TORCH_CHECK(wf.numel() == NHH * DPW && bias.numel() == NHH * R && R == L * L, "pair_bias: wf [heads, d_pair], bias [heads, L, L]");
  if (mask) TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == L, "pair_bias: mask [L] bool");
  const at::cuda::CUDAGuard g(z.device());
  const bool* mp = mask ? reinterpret_cast<const bool*>(mask->data_ptr()) : nullptr;
  const float* ex = extra ? CP<float>(*extra) : nullptr;
  TDT_DISPATCH(z.scalar_type(), ZT, [&] {
    TDT_DISPATCH(bias.scalar_type(), OT, [&] {
      const int64_t grid = std::max<int64_t>(1, std::min<int64_t>((R + PT - 1) / PT, (int64_t)nsm() * occupancy(pair_bias_m_k<ZT, OT>, 256, PB_SMEM)));
      pair_bias_m_k<ZT, OT><<<(unsigned)grid, 256, PB_SMEM, S()>>>(CP<ZT>(z), z.stride(0), CP<float>(wf), mp, P<OT>(bias), P<float2>(pst), R,
                                                             (int)L, (float)eps, ex, (float)extra_scale, (float)masked);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the weight pack and the final gradients, UNR vectors per thread ----
constexpr int UNR = 8;

// the 4 elements at i of pack output o (train_pack90_k's values)
__device__ __forceinline__ float4 train_pack_value(const TrainPack90& p,int o,int i) {
 float4 v=make_float4(0,0,0,0);
 if(o==0||o==1)v=train_read4(p,o==0?6:14,i);
 else if(o==2||o==3) {
  int grp=i/(D*DC),j=i%(D*DC),c=i%DC;
  v=train_read4(p,grp==0?7:grp==1?9:grp==2?15:17,j);
  if(o==3)v=mul4(v,train_read4(p,grp<2?6:14,c));
 }else if(o==4)v=train_read4(p,i/(D*DC)==0?10:18,i%(D*DC));
 else if(o>=5&&o<=8)v=train_read4(p,o==5?8:o==6?16:o==7?11:19,i);
 else if(o==9){int grp=i/(D*D);v=train_read4(p,grp==0?0:grp==1?2:grp==2?3:4,i%(D*D));}
 else if(o==10){if(i<D)v=train_read4(p,1,i);}
 else if(o==11||o==12)v=p.qk?train_read4(p,o==11?23:24,i):make_float4(1,1,1,1);
 else if(o==13)v=train_read4(p,12,i);
 else if(o==14)v=train_read4(p,13,i);
 else if(o>=15&&o<=17){
  v=mul4(train_read4(p,13,i),train_read4(p,12,i%DPW));
  if(o==17)v=mul4(v,make_float4(1.4426950408889634f,1.4426950408889634f,1.4426950408889634f,1.4426950408889634f));
 }else if(o==18)v=train_read4(p,5,i);
 else if(o==19)v=train_read4(p,i/(HID*D)==0?20:21,i%(HID*D));
 else v=train_read4(p,22,i);
 return v;
}
__global__ void train_pack100_k(TrainPack90 p) {
 int block=blockIdx.x,o=0;
 for(;o<21;++o){if(block<p.blocks[o])break;block-=p.blocks[o];}
 if(o==21)return;
 const int i0=(block*256*UNR+threadIdx.x)*4;
 if(i0>=p.size[o])return;
 float4 vv[UNR];
 #pragma unroll
 for(int u=0;u<UNR;++u){const int i=i0+u*1024;if(i<p.size[o])vv[u]=train_pack_value(p,o,i);}
 #pragma unroll
 for(int u=0;u<UNR;++u){
  const int i=i0+u*1024;
  if(i<p.size[o]){if(p.df[o])V4<float>::store((float*)p.dst[o]+i,vv[u]);else V4<bf>::store((bf*)p.dst[o]+i,vv[u]);}
 }
}
// train_pack100_k over a grid of any size (block b takes the pack blocks b, b + grid, ...): a pack beside the chain on a few SMs leaves the
// others to the chain's kernels (a full grid holds every SM until it drains)
__global__ void train_pack100s_k(TrainPack90 p, int total) {
 for (int bb = blockIdx.x; bb < total; bb += gridDim.x) {
  int block = bb, o = 0;
  for (; o < 21; ++o) { if (block < p.blocks[o]) break; block -= p.blocks[o]; }
  if (o == 21) return;
  const int i0 = (block * 256 * UNR + threadIdx.x) * 4;
  if (i0 >= p.size[o]) continue;
  float4 vv[UNR];
#pragma unroll
  for (int u = 0; u < UNR; ++u) { const int i = i0 + u * 1024; if (i < p.size[o]) vv[u] = train_pack_value(p, o, i); }
#pragma unroll
  for (int u = 0; u < UNR; ++u) {
   const int i = i0 + u * 1024;
   if (i < p.size[o]) { if (p.df[o]) V4<float>::store((float*)p.dst[o] + i, vv[u]); else V4<bf>::store((bf*)p.dst[o] + i, vv[u]); }
  }
 }
}
std::vector<at::Tensor> train_pack100(std::vector<at::Tensor> weights,bool qk,bool fp32) {
 TORCH_CHECK((D==768||D==1024)&&weights.size()==(qk?25:23),"native token pack: 768 or 1024 wide weights");
 const at::cuda::CUDAGuard guard(weights[0].device());
 TrainPack90 p{};p.qk=qk;
 for(int j=0;j<(int)weights.size();++j){
  TORCH_CHECK(weights[j].is_cuda()&&weights[j].is_contiguous()&&weights[j].device()==weights[0].device()&&(weights[j].scalar_type()==at::kFloat||weights[j].scalar_type()==at::kBFloat16),"native token weight layout/dtype");
  p.src[j]=weights[j].data_ptr();p.sf[j]=weights[j].scalar_type()==at::kFloat;
 }
 std::vector<std::vector<int64_t>> shapes={{DC},{DC},{4*D,DC},{4*D,DC},{2*D,DC},{D},{D},{D},{D},
 {4*D,D},{4*D},{HD},{HD},{DPW},{NHH,DPW},{NHH,DPW},{NHH,DPW},{NHH,DPW},{D,D},{2*HID,D},{D,HID}};
 const int aliases[21]={6,14,-1,-1,-1,8,16,11,19,-1,-1,qk?23:-1,qk?24:-1,12,13,-1,-1,-1,5,-1,22};
 std::vector<at::Tensor> out;int total=0;
 for(int o=0;o<21;++o){
  bool isact=(o==3||o==4||o==9||o==10||o==16||o==17||o>=18);
  auto dt=isact&&!fp32?at::kBFloat16:at::kFloat;p.df[o]=dt==at::kFloat;
  const int a=aliases[o];
  if(a>=0&&weights[a].scalar_type()==dt){out.push_back(weights[a]);continue;}
  out.push_back(at::empty(shapes[o],weights[0].options().dtype(dt)));
  p.dst[o]=out.back().data_ptr();p.size[o]=out.back().numel();p.blocks[o]=(p.size[o]+1024*UNR-1)/(1024*UNR);total+=p.blocks[o];
 }
 train_pack100_k<<<total,256,0,S()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// the 4 elements at i of gradient output o (train_final90_k's values)
__device__ __forceinline__ float4 train_final_value(const TrainGrad90& p,int o,int i) {
 float4 v=make_float4(0,0,0,0);
 if(o==6||o==14){
  int off=o==6?0:p.lnparts;
  for(int r=0;r<p.lnparts;++r)v=add4(v,V4<float>::load(p.lnp+(r+off)*DC+i));
 }else if(o==7||o==9||o==15||o==17) {
  int group=o==7?0:o==9?1:o==15?2:3;
  v=mul4(V4<float>::load(p.dwn+group*D*DC+i),V4<float>::load((group<2?p.w1:p.w2)+i%DC));
 }else if(o==1||o==8||o==11||o==16||o==19) {
  int group=o==1?4:o==8?3:o==11?2:o==16?1:0;
  for(int r=0;r<p.parts;++r)v=add4(v,V4<float>::load(p.allp+r*7*D+group*D+i));
 }else if(o==12){
  for(int h=0;h<NHH;++h)v=add4(v,mul4(V4<float>::load(p.dwf+h*DPW+i),V4<float>::load(p.wb+h*DPW+i)));
 }else if(o==13)v=mul4(V4<float>::load(p.dwf+i),V4<float>::load(p.wp+i%DPW));
 else if(o==23||o==24){
  for(int h=0;h<NHH;++h)v=add4(v,V4<float>::load(p.qkw+((o-23)*NHH+h)*HD+i));
 }else v=V4<float>::load(p.src[o]+i);
 return v;
}
__global__ void train_final100_k(TrainGrad90 p) {
 int block=blockIdx.x,o=0;
 for(;o<p.n;++o){if(block<p.blocks[o])break;block-=p.blocks[o];}
 if(o==p.n)return;
 const int i0=(block*256*UNR+threadIdx.x)*4;
 if(i0>=p.size[o])return;
 float4 vv[UNR];
 #pragma unroll
 for(int u=0;u<UNR;++u){const int i=i0+u*1024;if(i<p.size[o])vv[u]=train_final_value(p,o,i);}
 #pragma unroll
 for(int u=0;u<UNR;++u){
  const int i=i0+u*1024;
  if(i<p.size[o]){if(p.fp32[o])V4<float>::store((float*)p.dst[o]+i,vv[u]);else V4<bf>::store((bf*)p.dst[o]+i,vv[u]);}
 }
}
std::vector<at::Tensor> train_final100(std::vector<at::Tensor> params,std::vector<at::Tensor> grads,
 at::Tensor dwn,at::Tensor raw,at::Tensor w1,at::Tensor w2,at::Tensor allp,at::Tensor dwf,at::Tensor wb,at::Tensor wp,bool hierarchical=false) {
 TORCH_CHECK((D==768||D==1024)&&(params.size()==23||params.size()==25)&&grads.size()==params.size(),"native token grad layout");
 const at::cuda::CUDAGuard guard(dwn.device());
 auto parts=at::empty({4*D/16,DC},dwn.options());
 train_lnw_parts90_k<<<4*D/16,NTC,0,S()>>>(CP<float>(dwn),CP<float>(raw),P<float>(parts));
 auto rp=allp,lp=parts;
 if(hierarchical){
   const int groups=(allp.size(0)+15)/16;
   rp=at::empty({groups,7*D},allp.options());lp=at::empty({D/64,DC},parts.options());
   train_reduce_parts90_k<<<dim3(((params.size()==25?7:5)*D+1023)/1024,std::max(groups,D/64)),256,0,S()>>>(
       CP<float>(allp),CP<float>(parts),P<float>(rp),P<float>(lp),allp.size(0),(params.size()==25?7:5)*D);
 }
 TrainGrad90 p{};p.n=params.size();p.parts=rp.size(0);p.lnparts=hierarchical?D/128:2*D/16;
 at::Tensor qkw;
 if(p.n==25){qkw=at::empty({2,NHH,HD},dwn.options());train_qkw90_k<<<dim3(2,NHH),256,0,S()>>>(CP<float>(rp),p.parts,P<float>(qkw));p.qkw=CP<float>(qkw);}
 p.dwn=CP<float>(dwn);p.w1=CP<float>(w1);p.w2=CP<float>(w2);p.allp=CP<float>(rp);
 p.lnp=CP<float>(lp);p.dwf=CP<float>(dwf);p.wb=CP<float>(wb);p.wp=CP<float>(wp);
 std::vector<at::Tensor> out;int total=0;
 for(int o=0;o<p.n;++o){
  out.push_back(at::empty(params[o].sizes(),params[o].options()));
  TORCH_CHECK(grads[o].scalar_type()==at::kFloat&&grads[o].is_contiguous(),"native gradient source");
  p.src[o]=CP<float>(grads[o]);p.dst[o]=out.back().data_ptr();p.fp32[o]=out.back().scalar_type()==at::kFloat;
  p.size[o]=out.back().numel();p.blocks[o]=(p.size[o]+1024*UNR-1)/(1024*UNR);total+=p.blocks[o];
 }
 train_final100_k<<<total,256,0,S()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// ---- unfold_lnw, bitwise repeatable: unfold_lnw_k with the block's sums stored to its own partial row PART[block] (4 D / 16 rows),
// then dw12[j] = the 2 D / 16 rows of groups 2 j, 2 j + 1 added in block order
__global__ void __launch_bounds__(NTC) unfold_lnw100_k(const float* __restrict__ DWN, const float* __restrict__ WS1,
    const float* __restrict__ WB1, const float* __restrict__ WS2, const float* __restrict__ WB2, const float* __restrict__ W1,
    const float* __restrict__ W2, float* __restrict__ DW, float* __restrict__ PART) {
  const int col = threadIdx.x * 4, i0 = blockIdx.x * 16, grp = i0 / D;
  const float* Wg = grp == 0 ? WS1 : grp == 1 ? WB1 : grp == 2 ? WS2 : WB2;
  const float4 lw = V4<float>::load((grp < 2 ? W1 : W2) + col);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = i0; i < i0 + 16; ++i) {
    const float4 d = V4<float>::load(DWN + (long)i * DC + col);
    V4<float>::store(DW + (long)i * DC + col, mul4(d, lw));
    acc = add4(acc, mul4(d, V4<float>::load(Wg + (long)(i % D) * DC + col)));
  }
  V4<float>::store(PART + (long)blockIdx.x * DC + col, acc);
}
__global__ void __launch_bounds__(NTC) lnw_reduce100_k(const float* __restrict__ PART, float* __restrict__ DW12) {
  const int col = threadIdx.x * 4, n = 2 * D / 16;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int r = 0; r < n; ++r) acc = add4(acc, V4<float>::load(PART + (long)(blockIdx.x * n + r) * DC + col));
  V4<float>::store(DW12 + blockIdx.x * DC + col, acc);
}
void unfold_lnw100(at::Tensor dwn, at::Tensor ws1, at::Tensor wb1, at::Tensor ws2, at::Tensor wb2, at::Tensor w1, at::Tensor w2,
                   at::Tensor dw, at::Tensor dw12) {
  TORCH_CHECK(dwn.size(0) == 4 * D && dwn.size(1) == DC, "unfold_lnw: dWn [4 D, DC]");
  f32c(dw12, "dw12");
  TORCH_CHECK(dw12.numel() == 2 * DC, "unfold_lnw: dw12 [2, DC]");
  const at::cuda::CUDAGuard g(dwn.device());
  auto part = at::empty({4 * D / 16, DC}, dwn.options());
  unfold_lnw100_k<<<4 * D / 16, NTC, 0, S()>>>(CP<float>(dwn), CP<float>(ws1), CP<float>(wb1), CP<float>(ws2), CP<float>(wb2), CP<float>(w1),
                                               CP<float>(w2), P<float>(dw), P<float>(part));
  lnw_reduce100_k<<<2, NTC, 0, S()>>>(CP<float>(part), P<float>(dw12));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the final gradients split by their producers (2026-11-17, part 3): a captured training step ran the cond-LN weight partials,
// the partial-row reductions and ONE final kernel (a copy of every weight gradient into its leaf, ~57 MB at d 768) as a 25 us tail
// after the last row pass. The weight-gradient GEMMs now write their leaves themselves; what remains runs beside the chain:
//   final_lnw100     the four cond-AdaLN weight leaves dW = dWn diag(w) (straight into their leaves) and the cond-LN weight partial rows
//                    PART[block] = sum over the block's 16 rows of dWn * W (the raw leaves read in place), as train_lnw_parts90_k
//   final_reduce100  the partial rows of the row kernels (groups of 16) and PART (D / 64 groups), train_reduce_parts90_k
//   final_rest100    the leaves no producer wrote (biases, cond-LN weights, the pair weights, QK-norm weights): train_final100_k over
//                    those outputs only
constexpr int LNW_R = 8;                   // dWn rows a block of final_lnw100_k (4 D / 8 blocks: twice the 16-row grid's parallelism)
__global__ void __launch_bounds__(NTC) final_lnw100_k(const float* __restrict__ DWN, const float* __restrict__ WS1,
    const float* __restrict__ WB1, const float* __restrict__ WS2, const float* __restrict__ WB2, const float* __restrict__ W1,
    const float* __restrict__ W2, float* __restrict__ O0, float* __restrict__ O1, float* __restrict__ O2, float* __restrict__ O3,
    float* __restrict__ PART) {
  const int col = threadIdx.x * 4, i0 = blockIdx.x * LNW_R, grp = i0 / D;
  const float* Wg = grp == 0 ? WS1 : grp == 1 ? WB1 : grp == 2 ? WS2 : WB2;
  float* Og = grp == 0 ? O0 : grp == 1 ? O1 : grp == 2 ? O2 : O3;
  const float4 lw = V4<float>::load((grp < 2 ? W1 : W2) + col);
  float4 d[LNW_R], w[LNW_R];
#pragma unroll
  for (int r = 0; r < LNW_R; ++r) {
    d[r] = V4<float>::load(DWN + (long)(i0 + r) * DC + col);
    w[r] = V4<float>::load(Wg + (long)((i0 + r) % D) * DC + col);
  }
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int r = 0; r < LNW_R; ++r) {
    V4<float>::store(Og + (long)((i0 + r) % D) * DC + col, mul4(d[r], lw));
    acc = add4(acc, mul4(d[r], w[r]));
  }
  V4<float>::store(PART + (long)blockIdx.x * DC + col, acc);
}
// train_reduce_parts90_k with the lnw partial rows of final_lnw100_k: D / 64 groups (D / 128 per branch) of 4 D / LNW_R / (D / 64) rows each
__global__ void final_reduce100_k(const float* allp, const float* lnp, float* out, float* lout, int rows, int width) {
  const int c = (blockIdx.x * 256 + threadIdx.x) * 4;
  if (c >= width) return;
  const int g = blockIdx.y, r0 = g * 16;
  float4 v = make_float4(0, 0, 0, 0);
  if (r0 < rows) {
#pragma unroll
    for (int r = 0; r < 16; ++r) if (r0 + r < rows) v = add4(v, V4<float>::load(allp + (r0 + r) * 7 * D + c));
    V4<float>::store(out + g * 7 * D + c, v);
  }
  constexpr int PR = 4 * D / LNW_R / (D / 64);
  if (blockIdx.x == 0 && c < DC && g < D / 64) {
    float4 z = make_float4(0, 0, 0, 0);
#pragma unroll
    for (int r = 0; r < PR; ++r) z = add4(z, V4<float>::load(lnp + (g * PR + r) * DC + c));
    V4<float>::store(lout + g * DC + c, z);
  }
}
at::Tensor final_lnw100(at::Tensor dwn, at::Tensor ws1, at::Tensor wb1, at::Tensor ws2, at::Tensor wb2, at::Tensor w1, at::Tensor w2,
                        std::vector<at::Tensor> outs) {
  TORCH_CHECK(dwn.size(0) == 4 * D && dwn.size(1) == DC && outs.size() == 4, "final_lnw: dWn [4 D, DC], four leaves");
  for (auto* t : {&dwn, &ws1, &wb1, &ws2, &wb2, &w1, &w2}) f32c(*t, "final_lnw operand");
  for (auto& o : outs) { f32c(o, "final_lnw leaf"); TORCH_CHECK(o.numel() == D * DC, "final_lnw: leaf [D, DC]"); }
  const at::cuda::CUDAGuard g(dwn.device());
  auto part = at::empty({4 * D / LNW_R, DC}, dwn.options());
  final_lnw100_k<<<4 * D / LNW_R, NTC, 0, S()>>>(CP<float>(dwn), CP<float>(ws1), CP<float>(wb1), CP<float>(ws2), CP<float>(wb2),
                                               CP<float>(w1), CP<float>(w2), P<float>(outs[0]), P<float>(outs[1]), P<float>(outs[2]),
                                               P<float>(outs[3]), P<float>(part));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return part;
}
std::vector<at::Tensor> final_reduce100(at::Tensor allp, at::Tensor part, int64_t width) {
  f32c(allp, "allp"); f32c(part, "part");
  TORCH_CHECK(allp.size(1) == 7 * D && part.size(0) == 4 * D / LNW_R && (width == 5 * D || width == 7 * D), "final_reduce: allp [rows, 7 D]");
  const at::cuda::CUDAGuard g(allp.device());
  const int groups = (allp.size(0) + 15) / 16;
  auto rp = at::empty({groups, 7 * D}, allp.options()), lp = at::empty({D / 64, DC}, part.options());
  final_reduce100_k<<<dim3(((int)width + 1023) / 1024, std::max(groups, D / 64)), 256, 0, S()>>>(
      CP<float>(allp), CP<float>(part), P<float>(rp), P<float>(lp), allp.size(0), (int)width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {rp, lp};
}
void final_rest100(std::vector<at::Tensor> outs, std::vector<bool> fill, at::Tensor rp, at::Tensor lp, at::Tensor dwf, at::Tensor wb,
                   at::Tensor wp) {
  TORCH_CHECK((D == 768 || D == 1024) && (outs.size() == 23 || outs.size() == 25) && fill.size() == outs.size(), "final_rest layout");
  const at::cuda::CUDAGuard guard(rp.device());
  TrainGrad90 p{};
  p.n = outs.size(); p.parts = rp.size(0); p.lnparts = lp.size(0) / 2;
  at::Tensor qkw;
  if (p.n == 25 && (fill[23] || fill[24])) {
    qkw = at::empty({2, NHH, HD}, rp.options());
    train_qkw90_k<<<dim3(2, NHH), 256, 0, S()>>>(CP<float>(rp), p.parts, P<float>(qkw));
    p.qkw = CP<float>(qkw);
  }
  p.allp = CP<float>(rp); p.lnp = CP<float>(lp); p.dwf = CP<float>(dwf); p.wb = CP<float>(wb); p.wp = CP<float>(wp);
  int total = 0;
  for (int o = 0; o < p.n; ++o) {
    if (!fill[o]) continue;
    TORCH_CHECK(o == 1 || o == 6 || o == 8 || o == 11 || o == 12 || o == 13 || o == 14 || o == 16 || o == 19 || o == 23 || o == 24,
                "final_rest: output ", o, " has a producer of its own");
    TORCH_CHECK(outs[o].is_contiguous(), "final_rest: contiguous leaves");
    p.dst[o] = outs[o].data_ptr(); p.fp32[o] = outs[o].scalar_type() == at::kFloat;
    p.size[o] = outs[o].numel(); p.blocks[o] = (p.size[o] + 1024 * UNR - 1) / (1024 * UNR); total += p.blocks[o];
  }
  if (total) train_final100_k<<<total, 256, 0, S()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// replace the shared binding ``name`` (module_::def would add an overload behind it)
template <typename F, typename... Extra>
void rebind(pybind11::module_& m, const char* name, F&& f, const Extra&... extra) {
  m.add_object(name, pybind11::cpp_function(std::forward<F>(f), pybind11::name(name), pybind11::scope(m), extra...), true);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  tdt_shared_bindings(m);
  namespace py = pybind11;
  rebind(m, "pair_bias", &pair_bias_m, py::arg("z"), py::arg("wf"), py::arg("mask"), py::arg("bias"), py::arg("pst"), py::arg("L"),
         py::arg("eps"), py::arg("extra") = py::none(), py::arg("extra_scale") = 1.0,
         py::arg("masked") = -std::numeric_limits<double>::infinity());
  rebind(m, "train_pack90", &train_pack100);
  rebind(m, "unfold_lnw", &unfold_lnw100);
  rebind(m, "train_final90", &train_final100, py::arg("params"), py::arg("grads"), py::arg("dwn"), py::arg("raw"), py::arg("w1"),
         py::arg("w2"), py::arg("allp"), py::arg("dwf"), py::arg("wb"), py::arg("wp"), py::arg("hierarchical") = false);
  m.def("final_lnw100", &final_lnw100);
  m.def("final_reduce100", &final_reduce100);
  m.def("final_rest100", &final_rest100);
}
