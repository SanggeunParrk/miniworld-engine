from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_ln_policy_tune_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py',P/'replace_core.py',P/'save_k3.cu',P/'bench.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'b1_fused.cu').read_text();a=s.index(' if(wi==1){',s.index('TMN_DEVI void prepare_shared'));b=s.index('\n fence_proxy_async();allsync();\n recompute_gemm<256',a);block=s[a:b];s=s[:a]+'#if XHAT_FP32 != -1\n'+block+'\n#endif'+s[b:]
a=s.index(' // Input x_n',s.index('TMN_DEVI void prepare_shared'));s=s[:a]+'#if XHAT_FP32 == -1\n'+block+'\n#endif\n'+s[a:];s=s.replace('constexpr int affine_unroll = 4','constexpr int affine_unroll = B1_STREAM_AFFINE_UNROLL')
(R/'b1_fused.cu').write_text(s)
s=(R/'save_k3.cu').read_text().replace('int row, int M)', 'int row, int M, const CUtensorMap* outmap, uint8_t* stage)')
s=s.replace('    // Save pre-affine normalization','''#if XHAT_FP32 == 1
    if((ks&1)==0){if(lane==0)tma_store_wait_read<0>();__syncwarp();}
#endif
    // Save pre-affine normalization''')
s=s.replace('     reinterpret_cast<float*>(output)[(size_t)cc*M+rr]=a;reinterpret_cast<float*>(output)[(size_t)(cc+1)*M+rr]=b;', '''     int tr=lane/4+8*(j&1),c0=cc%32;
     *reinterpret_cast<float*>(stage+c0*64+((tr*4)^((c0&3)*16)))=a;
     *reinterpret_cast<float*>(stage+(c0+1)*64+((tr*4)^(((c0+1)&3)*16)))=b;''')
s=s.replace('    const uint32_t k0 =', '''#if XHAT_FP32 == 1
    if(ks&1){fence_proxy_async();__syncwarp();if(lane==0){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2,%3}], [%1];"::"l"(outmap),"r"(smem_u32(stage)),"r"(row-lane/4),"r"((ks/2)*32):"memory");tma_store_commit();}}
#endif
    const uint32_t k0 =''')
s=s.replace('  return LnStats{meanA, rA, meanB, rB};', '''#if XHAT_FP32 == 1
  if(lane==0)tma_store_wait_read<0>();__syncwarp();
#endif
  return LnStats{meanA, rA, meanB, rB};''')
s=s.replace('p.N*p.N);','p.N*p.N,&tp.tm_lnout,sOut+(4*cw+wiw)*OB);')
(R/'save_k3.cu').write_text(s)
s=(R/'replace_core.py').read_text();s=s.replace(" p=L.Struct([base,d['ds'],om('xn',128),maps[-1]", " outmap=L.tensor_map(saves['xnout'],[16,32],dims=[m,256],strides_bytes=[m*4],swizzle='64B',l2='128B') if method==1 else maps[-1]\n p=L.Struct([base,d['ds'],om('xn',128),outmap")
(R/'replace_core.py').write_text(s)
s=(R/'bench.py').read_text().replace(",'xhat_fp16':Replacement(a,2),'xhat_fp16_recover':Replacement(a,3)","")
(R/'bench.py').write_text(s)
s=(P/'run.sbatch').read_text().replace('trimul_ln_policy_tune_20260921','trimul_ln_policy_v2_20260921');(R/'run.sbatch').write_text(s)
print('generated V2')
