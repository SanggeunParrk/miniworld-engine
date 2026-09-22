from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_ln_policy_v4_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py',P/'replace_core.py',P/'save_k3.cu',P/'bench.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'b1_fused.cu').read_text().replace('*dg,*dwg,*dwp;', '*dg,*dwg,*dwp,*dtri_ptr;')
s=s.replace('49152+(XHAT_FP32==-1?512:0)', '(XHAT_FP32==1?16384:49152)+(XHAT_FP32==-1?512:0)')
s=s.replace(' for(int c=0;c<2;++c)for(int k=0;k<2;++k)tma_load_2d(sm+114688+k*16384+c*8192,&p.wg,bar+slot,c*64,k*64);', ' // Gate weights stay resident. xhat loads after the input gate consumes x_n.')
s=s.replace('tma_load_2d(x+16384,&p.xhat,bars+4,row,0);tma_load_2d(sm+114688,&p.xhat,bars+4,row+32,0);','for(int q=0;q<4;++q)tma_load_2d(q<3?x+q*16384:sm,&p.xhat,bars+4,row+16*q,0);')
s=s.replace('ldsm_x4(dy,smem_u32(sm+wi*8192)+off);','ldsm_x4(dy,smem_u32((XHAT_FP32==1?norm+32768:sm)+wi*8192)+off);')
s=s.replace('  stsm_x4(smem_u32(sm+wi*8192)+off,dg[0],dg[1],dg[2],dg[3]);','''#if XHAT_FP32 == 1
  #pragma unroll
  for(int j=0;j<4;++j){int r=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j>>1);stg32(p.dg+(size_t)(row+r)*128+wi*64+c,dg[j]);}
#else
  stsm_x4(smem_u32(sm+wi*8192)+off,dg[0],dg[1],dg[2],dg[3]);
#endif''')
s=s.replace(' if(threadIdx.x==0){dg_store(', ' if(XHAT_FP32!=1 && threadIdx.x==0){dg_store(')
s=s.replace('  load_dy(p,sm,bars+2,tile*64);','  load_dy(p,XHAT_FP32==1?sm+16384+(1-slot)*49152+32768:sm,bars+2,tile*64);')
s=s.replace('  for(int j=threadIdx.x;j<4096;j+=256)', '  if(XHAT_FP32!=1)for(int j=threadIdx.x;j<4096;j+=256)')
s=s.replace('// Wg has finished: reuse its 32 KiB tile for the second half of xhat.', '// Wg has finished: x_n and dy staging are free; keep Wg resident.')
(R/'b1_fused.cu').write_text(s)
s=(R/'xhat_read.inc').read_text().replace('uint8_t* p=row<32?sm+32768+slot*49152:sm+114688;\n return *reinterpret_cast<float*>(p+swz128(c,(row%32)*4));', 'int q=row/16;uint8_t* p=q<3?sm+16384+slot*49152+q*16384:sm;\n return *reinterpret_cast<float*>(p+c*64+(((row%16)*4)^(((c>>1)&3)*16)));')
(R/'xhat_read.inc').write_text(s)
s=(R/'b1_lowreg.inc').read_text().replace('uint8_t* sn=sm+16384+(1-slot)*49152;', 'uint8_t* sn=sm+16384+(1-slot)*49152;uint8_t* dp=XHAT_FP32==1?sn+32768:sm;')
s=s.replace('sm+(k/4)*8192+(k%4)*32','dp+(k/4)*8192+(k%4)*32')
a=s.index('#if XHAT_FP32 == 1\n   uint8_t* dst=');b=s.index('#else',a)
s=s[:a]+'''#if XHAT_FP32 == 1
   #pragma unroll
   for(int j=0;j<4;++j){int r=m0+w*16+lane/4+8*(j&1),c=n*64+q*16+2*(lane%4)+8*(j>>1);reinterpret_cast<uint16_t*>(p.dtri_ptr)[(size_t)c*p.M+r]=out[j]&65535;reinterpret_cast<uint16_t*>(p.dtri_ptr)[(size_t)(c+1)*p.M+r]=out[j]>>16;}
'''+s[b:]
s=s.replace('sync_group();if(tid==0){for(int ch=', 'sync_group();if(XHAT_FP32!=1 && tid==0){for(int ch=')
(R/'b1_lowreg.inc').write_text(s)
# stats path uses a separate include but Params gained a pointer only; unchanged math.
s=(R/'replace_plan.py').read_text().replace("tm(xhat,[128//es,256],[m,256],[m*es])", "L.tensor_map(xhat,[16,256],dims=[m,256],strides_bytes=[m*es],swizzle='64B',l2='128B') if es==4 else tm(xhat,[64,256],[m,256],[m*es])")
s=s.replace('rstd,dg,dwg,dwp,dgo', 'rstd,dg,dwg,dwp,dt,dgo')
(R/'replace_plan.py').write_text(s)
s=(P/'run.sbatch').read_text().replace('trimul_ln_policy_v4_20260921','trimul_ln_policy_v5_20260921');(R/'run.sbatch').write_text(s)
print('generated V5')
