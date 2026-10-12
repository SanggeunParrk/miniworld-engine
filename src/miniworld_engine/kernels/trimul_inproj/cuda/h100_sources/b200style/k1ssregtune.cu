// Hopper port of B200's resident normalized-input / streamed-weight front.
// Normalized input lives in shared memory rather than a large RS fragment.
// WB reads saved input statistics, recomputes g/p, and never writes g/p to HBM.
#include "tmn_kernels.cuh"
#include "wgmma_n.cuh"
using namespace tmn;
using namespace tmn::sm90;
using bf = __nv_bfloat16;
#ifndef CACHEK
#define CACHEK (WIDTH / 128)
#endif
constexpr int D = WIDTH, H = HIDDEN, MT = TOKENS, NS = STAGES;
constexpr int BN = BLOCKN, NA = BN / 2, NP = BN / 2;
constexpr int NWG = MT / 64, NT = 128 * (NWG + 1), KB = D / 64, NB = 4 * H / BN;
constexpr int AB = MT * 128, A_BYTES = KB * AB, SLOT = BN * 128, W_BYTES = NS * SLOT;
#ifndef PRIVATESTORE
#define PRIVATESTORE 0
#endif
#ifndef WAITN
#define WAITN 1
#endif
static_assert(WAITN < NS && WAITN > 0, "WGMMA pipeline depth");
constexpr bool PRIVATE = BACKWARD && PRIVATESTORE;
constexpr int O_BYTES = NWG * (PRIVATE ? 24576 : 16384);
constexpr int SMEM = A_BYTES + W_BYTES + O_BYTES + 512;
static_assert(SMEM <= 232448 && MT % 64 == 0, "front shared memory");
struct Params {
  CUtensorMap x, w, ab, da, dc, xn;
  const bf* mask; const float *gamma, *beta, *xs; float* stats;
  int M, L, tiles, emit_xn;
};
TMN_DEVI uint32_t read_word(uint32_t addr) {
  uint32_t v; asm volatile("ld.shared.b32 %0,[%1];" : "=r"(v) : "r"(addr)); return v;
}
TMN_DEVI void store2(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
template<int N> TMN_DEVI void dot(float (&v)[N/2],uint64_t a,uint64_t b,int acc) {
  if constexpr(N==128) mw_mma128<0>(v,a,b,acc);
  else mw_mma256<0>(v,a,b,acc);
}
extern "C" __global__ __launch_bounds__(NT, 1)
void KERNEL_NAME(__grid_constant__ const Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* sa = sm; uint8_t* sw = sa + A_BYTES; uint8_t* so = sw + W_BYTES;
  uint64_t* bar = reinterpret_cast<uint64_t*>(so + O_BYTES);
  uint64_t* af = bar; uint64_t* ae = af + 1;
  uint64_t* wf = ae + 1; uint64_t* we = wf + NS; uint64_t* df = we + NS;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wg = tid / 128;
  if (tid == 0) {
    mbar_init(af, 1); mbar_init(ae, 4 * NWG);
    for (int s = 0; s < NS; ++s) { mbar_init(wf + s, 1); mbar_init(we + s, 4 * NWG); }
    for (int s = 0; s < NWG * (PRIVATE ? 4 : 1); ++s) mbar_init(df + s, 1);
    fence_barrier_init(); tma_prefetch_desc(&p.x); tma_prefetch_desc(&p.w);
    if (BACKWARD) { tma_prefetch_desc(&p.da); tma_prefetch_desc(&p.dc); if(p.emit_xn) tma_prefetch_desc(&p.xn); }
    else tma_prefetch_desc(&p.ab);
  }
  __syncthreads();
  const int ni = (p.tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
  if (wg == 0) {
    setmaxnreg_dec<24>();
    if (warp == 0 && lane == 0) {
      for (int i = 0; i < ni; ++i) {
        if (i) mbar_wait(ae, (i - 1) & 1);
        const int tok = (blockIdx.x + i * gridDim.x) * MT;
        mbar_arrive_expect_tx(af, A_BYTES);
        for (int k = 0; k < KB; ++k) tma_load_2d(sa + k * AB, &p.x, af, k * 64, tok);
      }
    } else if (warp == 1 && lane == 0) {
      for (int i = 0; i < ni * NB * KB; ++i) {
        const int slot = i % NS, phase = i / NS;
        if (phase) mbar_wait(we + slot, (phase - 1) & 1);
        mbar_arrive_expect_tx(wf + slot, SLOT);
        tma_load_2d(sw + slot * SLOT, &p.w, wf + slot, (i % KB) * 64, ((i / KB) % NB) * BN);
      }
    }
    return;
  }
  setmaxnreg_inc<240>();
  const int cw = wg - 1, wiw = warp % 4, idx = lane / 8, rr = lane % 8;
  const int rowa = 16 * wiw + lane / 4, rowb = rowa + 8;
  const bool elect = wiw == 0 && lane == 0;
  const int di = PRIVATE ? cw*4+wiw : cw;
  const bool stelect = PRIVATE ? lane==0 : elect;
  uint8_t* stage = so + (PRIVATE ? di*6144 : cw*16384); const uint32_t stage_u = smem_u32(stage);
  const uint32_t plane_off = swz128(8 * (idx >> 1) + rr, (16 * wiw + 8 * (idx & 1)) * 2);
  auto release = [&](int it) {
    fence_proxy_async(); __syncwarp();
    if (lane == 0) mbar_arrive(we + it % NS);
  };
  int it = 0;
  for (int ti = 0; ti < ni; ++ti) {
    const int tok0 = (blockIdx.x + ti * gridDim.x) * MT;
    mbar_wait(af, ti & 1);
    // Four lanes share a token row; vector shared accesses avoid the scalar
    // row-stride bank conflicts. WB reuses the saved statistics.
    for (int r = (warp-4)*8+lane/4; r < MT; r += NWG*32) {
      float mu = 0.f, rs = 1.f;
      auto addr = [&](int k, int c) { return smem_u32(sa) + k * AB + swz128(r, c * 2); };
      constexpr int CK=CACHEK;
      uint32_t raw[CK][8];
      if constexpr(!BACKWARD) {
#pragma unroll
        for(int k=0;k<CK;++k)
#pragma unroll
          for(int j=0;j<8;++j) raw[k][j]=read_word(addr(k,2*(lane&3)+8*j));
      }
      if (BACKWARD) { mu = p.xs[2 * (tok0 + r)]; rs = p.xs[2 * (tok0 + r) + 1]; }
      else {
#pragma unroll
        for (int k = 0; k < CK; ++k) {
#pragma unroll
          for (int c = 2*(lane&3); c < 64; c += 8) { const uint32_t v = raw[k][c/8]; mu += bf16lo(v) + bf16hi(v); }
        }
#pragma unroll 1
        for(int k=CK;k<KB;++k) {
#pragma unroll 4
          for(int c=2*(lane&3);c<64;c+=8) { const uint32_t v=read_word(addr(k,c));mu+=bf16lo(v)+bf16hi(v); }
        }
        mu = quad_sum(mu) * (1.f / D);
        float vv = 0.f;
#pragma unroll
        for (int k = 0; k < CK; ++k) {
#pragma unroll
          for (int c = 2*(lane&3); c < 64; c += 8) {
            const uint32_t v = raw[k][c/8]; const float a = bf16lo(v) - mu, b = bf16hi(v) - mu;
            vv = fmaf(a,a,vv); vv = fmaf(b,b,vv);
          }
        }
#pragma unroll 1
        for(int k=CK;k<KB;++k) {
#pragma unroll 4
          for(int c=2*(lane&3);c<64;c+=8) {
            const uint32_t v=read_word(addr(k,c)); const float a=bf16lo(v)-mu,b=bf16hi(v)-mu;
            vv=fmaf(a,a,vv);vv=fmaf(b,b,vv);
          }
        }
        rs = rsqrtf(quad_sum(vv) * (1.f / D) + 1e-5f);
        if (p.stats && (lane&3)==0) { p.stats[2 * (tok0+r)] = mu; p.stats[2 * (tok0+r)+1] = rs; }
      }
      if constexpr(!BACKWARD) {
#pragma unroll
      for (int k = 0; k < CK; ++k) {
#pragma unroll
        for (int c = 2*(lane&3); c < 64; c += 8) {
          const uint32_t a = addr(k,c), v = raw[k][c/8]; const int ch = k * 64 + c;
          sts32(a, pack_bf16(math::ln_affine(bf16lo(v),mu,rs,p.gamma[ch],p.beta[ch]),
                            math::ln_affine(bf16hi(v),mu,rs,p.gamma[ch+1],p.beta[ch+1])));
        }
      }
      }
#pragma unroll 1
      for(int k=BACKWARD?0:CK;k<KB;++k) {
#pragma unroll 4
        for(int c=2*(lane&3);c<64;c+=8) {
          const uint32_t a=addr(k,c),v=read_word(a);const int ch=k*64+c;
          sts32(a,pack_bf16(math::ln_affine(bf16lo(v),mu,rs,p.gamma[ch],p.beta[ch]),
                           math::ln_affine(bf16hi(v),mu,rs,p.gamma[ch+1],p.beta[ch+1])));
        }
      }
    }
    fence_proxy_async(); named_bar_sync(1, 128 * NWG);
    if(BACKWARD && p.emit_xn && tid==128) {
      for(int k=0;k<KB;++k) store2(&p.xn,sa+k*AB,k*64,tok0);
      tma_store_commit();
    }
    const int ta = tok0 + 64 * cw + rowa, tb = tok0 + 64 * cw + rowb;
    const float ma = p.mask ? __bfloat162float(p.mask[ta]) : 1.f;
    const float mb = p.mask ? __bfloat162float(p.mask[tb]) : 1.f;
    for (int block = 0; block < NB; ++block) {
      if (BACKWARD) {
        if (stelect && (block || ti)) tma_store_wait_read<0>();
        fence_proxy_async();
        if constexpr(PRIVATE) __syncwarp(); else named_bar_sync(2 + cw, 128);
        if (stelect) { mbar_arrive_expect_tx(df + di, PRIVATE ? 2048 : 8192); tma_load_2d(stage, &p.da, df + di, tok0 + cw*64 + (PRIVATE ? 16*wiw : 0), block*NP); }
      }
      float ac[NA];
#pragma unroll
      for (int j = 0; j < NA; ++j) ac[j] = 0.f;
      for (int k = 0; k < KB; ++k, ++it) {
        const int s = it % NS;
        mbar_wait(wf+s, (it/NS)&1);
        const uint32_t a = smem_u32(sa) + k*AB + cw*8192, b = smem_u32(sw) + s*SLOT;
        fence_regs(ac); wgmma_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          dot<BN>(ac,smem_desc(a+q*32,16,1024,1),smem_desc(b+q*32,16,1024,1),k>0 || q>0);
        }
        wgmma_commit();
        if (k>=WAITN) { wgmma_wait<WAITN>(); fence_regs(ac); release(it-WAITN); }
      }
      wgmma_wait<0>(); fence_regs(ac);
#pragma unroll
      for(int back= (WAITN < KB ? WAITN : KB); back > 0; --back) release(it-back);
      if (BACKWARD) {
#pragma unroll
        for(int part=0;part<NP/64;++part) {
        if(part>0) {
          if(stelect) tma_store_wait_read<0>();
          fence_proxy_async();
          if constexpr(PRIVATE) __syncwarp(); else named_bar_sync(2+cw,128);
          if(stelect) { mbar_arrive_expect_tx(df+di,PRIVATE ? 2048 : 8192); tma_load_2d(stage,&p.da,df+di,tok0+cw*64+(PRIVATE ? 16*wiw : 0),block*NP+part*64); }
        }
        mbar_wait(df+di, ((ti*NB+block)*(NP/64)+part)&1);
        uint32_t da[4][4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ch=(q/2)*32+(q%2)*16+8*(idx>>1)+rr;
          const uint32_t off=PRIVATE ? ch*32+((16*(idx&1))^(((ch>>2)&1)*16)) : (q/2)*4096+plane_off+(q%2)*2048;
          ldsm_x4_t(da[q],stage_u+off);
        }
        uint32_t pk[8][4];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int cq = part*8+q, g = cq + 4*(cq/4), pp = g + 4;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const uint32_t dd = da[q/2][2*(q%2)+e]; const float mask = e ? mb : ma;
            const float d0 = bf16lo(dd)*mask, d1 = bf16hi(dd)*mask;
            const float s0 = math::sigmoid(ac[4*g+2*e]), s1 = math::sigmoid(ac[4*g+2*e+1]);
            pk[q][2*e] = pack_bf16(d0*s0*ac[4*pp+2*e]*(1.f-s0),d1*s1*ac[4*pp+2*e+1]*(1.f-s1));
            pk[q][2*e+1] = pack_bf16(d0*s0,d1*s1);
          }
        }
        if constexpr(PRIVATE) __syncwarp(); else named_bar_sync(2+cw,128);
        const int tr = (PRIVATE ? 0 : 16*wiw) + 8*(idx&1)+rr, cb = 8*(idx>>1);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const uint32_t base = stage_u+(PRIVATE ? 2048+half*2048 : half*8192); const int q=half*4;
          stsm_x4(base+swz128(tr,2*cb),pk[q][0],pk[q][2],pk[q+1][0],pk[q+1][2]);
          stsm_x4(base+swz128(tr,2*(cb+16)),pk[q+2][0],pk[q+2][2],pk[q+3][0],pk[q+3][2]);
          stsm_x4(base+swz128(tr,2*(cb+32)),pk[q][1],pk[q][3],pk[q+1][1],pk[q+1][3]);
          stsm_x4(base+swz128(tr,2*(cb+48)),pk[q+2][1],pk[q+2][3],pk[q+3][1],pk[q+3][3]);
        }
        fence_proxy_async();
        if constexpr(PRIVATE) __syncwarp(); else named_bar_sync(2+cw,128);
        if (stelect) {
          store2(&p.dc,stage+(PRIVATE ? 2048 : 0),block*BN+part*128,tok0+cw*64+(PRIVATE ? 16*wiw : 0));
          store2(&p.dc,stage+(PRIVATE ? 4096 : 8192),block*BN+part*128+64,tok0+cw*64+(PRIVATE ? 16*wiw : 0)); tma_store_commit();
        }
        }
      } else {
        uint32_t pk[NP/8][2];
#pragma unroll
        for (int q=0;q<NP/8;++q) {
          const int g=q+4*(q/4), pp=g+4;
          pk[q][0]=pack_bf16(math::gate(ac[4*g],ac[4*pp],ma),math::gate(ac[4*g+1],ac[4*pp+1],ma));
          pk[q][1]=pack_bf16(math::gate(ac[4*g+2],ac[4*pp+2],mb),math::gate(ac[4*g+3],ac[4*pp+3],mb));
        }
        if (elect) tma_store_wait_read<0>();
        named_bar_sync(2+cw,128);
#pragma unroll
        for(int half=0;half<NP/32;++half) {
          const int q=half*4; const uint32_t base=stage_u+half*4096;
          stsm_x4_t(base+plane_off,pk[q][0],pk[q][1],pk[q+1][0],pk[q+1][1]);
          stsm_x4_t(base+plane_off+2048,pk[q+2][0],pk[q+2][1],pk[q+3][0],pk[q+3][1]);
        }
        fence_proxy_async(); named_bar_sync(2+cw,128);
        if(elect) {
          for(int half=0;half<NP/32;++half) store2(&p.ab,stage+half*4096,tok0+cw*64,block*NP+half*32);
          tma_store_commit();
        }
      }
    }
    // All SS operand reads of this tile have retired before the next TMA fill.
    if(BACKWARD && p.emit_xn && tid==128) tma_store_wait_read<0>();
    fence_proxy_async(); __syncwarp(); if(lane==0) mbar_arrive(ae);
  }
  if(stelect) tma_store_wait_all();
}
