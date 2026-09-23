TMN_DEVI void gp_ss64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
// SPDX-License-Identifier: Apache-2.0
// Keep the validated N64 descriptor layout; one WG computes both N halves.
TMN_DEVI void mma_dxpacked(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 1, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void gate128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;auto& part=*reinterpret_cast<float(*)[32]>(d+h*32);mma_dgrad(part,a,b+h*(16384>>4),accumulate);});}
TMN_DEVI void proj128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;auto& part=*reinterpret_cast<float(*)[32]>(d+h*32);mma_input64(part,a,b+h*(8192>>4),accumulate);});}
TMN_DEVI void single_dx(float (&d)[64],uint64_t a,uint64_t b,int accumulate){static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;auto& part=*reinterpret_cast<float(*)[32]>(d+h*32);mma_dxpacked(part,a,b+h*(8192>>4),accumulate);});}

TMN_DEVI void half_derivatives(uint8_t* xn,uint8_t* wb,uint8_t* dy_s,uint8_t* sg,uint8_t* sp,uint32_t ma,uint32_t mb){
 int lane=threadIdx.x%32,w=(threadIdx.x%128)/32,mat=lane/8;float pre[32]={};fence_regs(pre);wgmma_fence();
 static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;gp_ss64(pre,smem_desc(smem_u32(xn+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(wb+(k/4)*8192+(k%4)*32),16,1024,1),k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(pre);
 static_for<2>([&](auto qq){constexpr int q=decltype(qq)::value;uint32_t dy[4],dg[4],dp[4];uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);ldsm_x4_t(dy,smem_u32(dy_s)+addr);
 static_for<4>([&](auto jj){constexpr int j=decltype(jj)::value;uint32_t masked,mask=(j&1)?mb:ma;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(mask));float ga=math::sigmoid(math::round_bf16(pre[q*8+j*2])),gb=math::sigmoid(math::round_bf16(pre[q*8+j*2+1]));float pa=math::round_bf16(pre[(q+2)*8+j*2]),pb=math::round_bf16(pre[(q+2)*8+j*2+1]);dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1-ga),((bf16hi(masked)*pb)*gb)*(1-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);});stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sp)+addr,dp[0],dp[1],dp[2],dp[3]);});fence_proxy_async();allsync();
}
