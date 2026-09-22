#ifndef B1_STREAM_AFFINE_UNROLL
#define B1_STREAM_AFFINE_UNROLL 1
#endif
// Same balanced reductions and affine expression as Anthropic's reference
// fragment LN, but re-read the on-chip raw tile instead of retaining 64 packs.
TMN_DEVI LnStats stream_ln(uint8_t* src,uint8_t* dst,const float* gamma,const float* beta){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8,r8=lane%8;
 float gA[16],gB[16];
 static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;uint32_t f[4];
  ldsm_x4_t(f,smem_u32(src)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
  gA[k]=__fadd_rn(__fadd_rn(bf16lo(f[0]),bf16hi(f[0])),__fadd_rn(bf16lo(f[2]),bf16hi(f[2])));
  gB[k]=__fadd_rn(__fadd_rn(bf16lo(f[1]),bf16hi(f[1])),__fadd_rn(bf16lo(f[3]),bf16hi(f[3])));
 });
 float ma=math::ln_mean(quad_sum(math::tree_sum(gA)),1.f/256),mb=math::ln_mean(quad_sum(math::tree_sum(gB)),1.f/256);
 static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;uint32_t f[4];
  ldsm_x4_t(f,smem_u32(src)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
  gA[k]=__fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(f[0]),ma),bf16hi(f[0]),ma),math::ln_sq_acc(math::ln_sq(bf16lo(f[2]),ma),bf16hi(f[2]),ma));
  gB[k]=__fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(f[1]),mb),bf16hi(f[1]),mb),math::ln_sq_acc(math::ln_sq(bf16lo(f[3]),mb),bf16hi(f[3]),mb));
 });
 float ra=math::ln_rstd(quad_sum(math::tree_sum(gA)),1.f/256,1e-5f),rb=math::ln_rstd(quad_sum(math::tree_sum(gB)),1.f/256,1e-5f);
 // Runtime loop prevents hoisting all raw and affine fragments at once.
 constexpr int affine_unroll = B1_STREAM_AFFINE_UNROLL;
 #pragma unroll affine_unroll
 for(int k=0;k<16;++k){uint32_t f[4];ldsm_x4_t(f,smem_u32(src)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));
  int c=16*k+2*(lane&3);float2 g0=lds64f(smem_u32(gamma)+4*c),b0=lds64f(smem_u32(beta)+4*c),g1=lds64f(smem_u32(gamma)+4*c+32),b1=lds64f(smem_u32(beta)+4*c+32);
  f[0]=pack_bf16(math::ln_affine(bf16lo(f[0]),ma,ra,g0.x,b0.x),math::ln_affine(bf16hi(f[0]),ma,ra,g0.y,b0.y));
  f[1]=pack_bf16(math::ln_affine(bf16lo(f[1]),mb,rb,g0.x,b0.x),math::ln_affine(bf16hi(f[1]),mb,rb,g0.y,b0.y));
  f[2]=pack_bf16(math::ln_affine(bf16lo(f[2]),ma,ra,g1.x,b1.x),math::ln_affine(bf16hi(f[2]),ma,ra,g1.y,b1.y));
  f[3]=pack_bf16(math::ln_affine(bf16lo(f[3]),mb,rb,g1.x,b1.x),math::ln_affine(bf16hi(f[3]),mb,rb,g1.y,b1.y));
  stsm_x4(smem_u32(dst+(k/4)*8192)+swz128(w*16+r8+8*(mat&1),(2*(k%4)+(mat>>1))*16),f[0],f[1],f[2],f[3]);
 }
 return LnStats{ma,ra,mb,rb};
}
