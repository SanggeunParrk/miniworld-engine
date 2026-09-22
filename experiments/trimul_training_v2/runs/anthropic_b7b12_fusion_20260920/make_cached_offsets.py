from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_cached_dx_q2_r32_224.cu').read_text()
operands=','.join('%%%d'%i for i in range(32));outs=','.join('"+f"(d[%d])'%i for i in range(32))
helper='''template<int AO,int BO,int BT> TMN_DEVI void mma_cached_ss(float (&d)[32],uint32_t al,uint32_t ah,uint32_t bl,uint32_t bh,int scale){
 asm volatile("{ .reg .pred p; .reg .b32 alo,blo; .reg .b64 ad,bd; setp.ne.b32 p, %36, 0; add.u32 alo,%32,%37; add.u32 blo,%34,%38; mov.b64 ad,{alo,%33}; mov.b64 bd,{blo,%35}; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {'''+operands+'''},ad,bd,p,1,1,0,%39; }" : '''+outs+''' : "r"(al),"r"(ah),"r"(bl),"r"(bh),"r"(scale),"n"(AO>>4),"n"(BO>>4),"n"(BT));
}
'''
a=s.index('// dX-only');s=s[:a]+helper+s[a:]
s=s.replace('{float gate[32]={};uint8_t* s=sm+49152;fence_regs(gate);wgmma_fence();static_for<8>', '{float gate[32]={};uint8_t* s=sm+49152;uint64_t ad=smem_desc(smem_u32(s+16384+wi*16384),16,1024,1),bd=smem_desc(smem_u32(s),16,1024,1);fence_regs(gate);wgmma_fence();static_for<8>')
s=s.replace('mma_dgrad(gate,smem_desc(smem_u32(s+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(s+(k/4)*8192+(k%4)*32),16,1024,1),k>0);','mma_cached_ss<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0>(gate,uint32_t(ad),uint32_t(ad>>32),uint32_t(bd),uint32_t(bd>>32),k>0);')
s=s.replace('mbar_wait(b->ready+half,side);fence_regs(acc);wgmma_fence();','mbar_wait(b->ready+half,side);uint64_t ad=smem_desc(smem_u32(sm+98304+side*65536+wi*32768+half*16384),16,1024,1),bd=smem_desc(smem_u32(s+32768),16,1024,1);fence_regs(acc);wgmma_fence();')
s=s.replace('mma_weight64(acc,smem_desc(smem_u32(sm+98304+side*65536+wi*32768+(half*2+k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(s+32768+k*2048),16,1024,1),side>0||half>0||k>0);','mma_cached_ss<(k/4)*8192+(k%4)*32,k*2048,1>(acc,uint32_t(ad),uint32_t(ad>>32),uint32_t(bd),uint32_t(bd>>32),side>0||half>0||k>0);')
for pr,co in [(32,224),(40,216),(24,232)]:
 name='front_cached_off_r%d_%d'%(pr,co);(p/(name+'.cu')).write_text(s.replace('setmaxnreg_dec<32>()','setmaxnreg_dec<%d>()'%pr).replace('setmaxnreg_inc<224>()','setmaxnreg_inc<%d>()'%co));(p/(name+'.launch.json')).write_text('{"direct_weights":true}\n')
