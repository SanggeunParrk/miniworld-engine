from pathlib import Path
p=Path(__file__).resolve().parent
src=(p/'front_cached_off_r32_224.cu').read_text().replace('uint64_t tx[5]','uint64_t tx[6]').replace('for(int i=0;i<5;++i)','for(int i=0;i<6;++i)')
a=src.index('TMN_DEVI void producer');helper='''TMN_DEVI void load_tail(const Params& p,uint8_t* sm,Bars* b,int side){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+5,16384);for(int c=0;c<2;++c)tma_load_2d(sm+65536+c*8192,side?&p.wr:&p.wl,b->tx+5,192,c*64);
}
''';src=src[:a]+helper+src[a:]
src=src.replace('glu_cached(p,sm,1,row);publish(b,1);','glu_cached(p,sm,1,row);publish(b,1);load_tail(p,sm,b,0);').replace('glu_cached(p,sm,3,row);publish(b,1);','glu_cached(p,sm,3,row);publish(b,1);load_tail(p,sm,b,1);')
for ck,pr,co in [(12,40,216),(14,32,224),(15,32,224),(12,48,208)]:
 s=src.replace('pw[2][16][4]','pw[2][%d][4]'%ck).replace('load_frag_bf16<16,8192>(pw','load_frag_bf16<%d,8192>(pw'%ck)
 s=s.replace('uint64_t bd=smem_desc(smem_u32(sm+half*49152),16,1024,1);fence_regs(acc);wgmma_fence();','''uint32_t tail[%d][4];if constexpr(half==1){mbar_wait(b->tx+5,side);static_for<%d>([&](auto ti){constexpr int q=decltype(ti)::value;int mat=lane/8,rr=w*16+(lane%%8)+((mat&1)?8:0),kk=(%d+q-12)*16+((mat&2)?8:0);ldsm_x4(tail[q],smem_u32(sm+65536+wi*8192)+swz128(rr,kk*2));});}uint64_t bd=smem_desc(smem_u32(sm+half*49152),16,1024,1);fence_regs(acc);wgmma_fence();'''%(16-ck,16-ck,ck))
 s=s.replace('mma_projection_rs<k*2048>(acc,pw[side][half*8+k],uint32_t(bd),uint32_t(bd>>32),1);','if constexpr(half*8+k<%d)mma_projection_rs<k*2048>(acc,pw[side][half*8+k],uint32_t(bd),uint32_t(bd>>32),1);else mma_projection_rs<k*2048>(acc,tail[half*8+k-%d],uint32_t(bd),uint32_t(bd>>32),1);'%(ck,ck))
 s=s.replace('static_for<16>([&](auto ki){constexpr int k=decltype(ki)::value;fence_regs(pw','static_for<%d>([&](auto ki){constexpr int k=decltype(ki)::value;fence_regs(pw'%ck)
 s=s.replace('setmaxnreg_dec<32>()','setmaxnreg_dec<%d>()'%pr).replace('setmaxnreg_inc<224>()','setmaxnreg_inc<%d>()'%co)
 name='front_cached_tail%d_r%d_%d'%(ck,pr,co);(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text('{"direct_weights":true}\n')
