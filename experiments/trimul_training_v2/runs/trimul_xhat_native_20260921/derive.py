from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_xhat_push_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py',P/'replace_core.py',P/'save_k3.cu',P/'bench.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'save_k3.cu').read_text()
a=s.index('#if XHAT_FP32 >= 0\n');b=s.index('    const uint32_t k0 =',a)
s=s[:a]+'''    // Native tile layout: [tile64, ks16, warp4, channel_half2, lane32, float4].
    // Each lane writes two aligned vectors with all normalization bits intact.
    float z[8];
    #pragma unroll
    for(int j=0;j<4;++j){float mu=(j&1)?meanB:meanA,rs=(j&1)?rB:rA;
     z[j*2]=__fmul_rn(__fsub_rn(bf16lo(fa[ks][j]),mu),rs);z[j*2+1]=__fmul_rn(__fsub_rn(bf16hi(fa[ks][j]),mu),rs);
    }
    size_t offset=((size_t)(row/64)*16+ks)*1024+((row%64)/16)*256+lane*4;
    stg128(reinterpret_cast<float*>(output)+offset,make_uint4(__float_as_uint(z[0]),__float_as_uint(z[1]),__float_as_uint(z[2]),__float_as_uint(z[3])));
    stg128(reinterpret_cast<float*>(output)+offset+128,make_uint4(__float_as_uint(z[4]),__float_as_uint(z[5]),__float_as_uint(z[6]),__float_as_uint(z[7])));
''' + s[b:]
s=s.replace('#if XHAT_FP32 == 1\n  if(lane==0)tma_store_wait_read<0>();__syncwarp();\n#endif','')
(R/'save_k3.cu').write_text(s)
s=(R/'replace_core.py').read_text().replace('torch.empty((256,m),device=x.device','torch.empty((m//64,16,4,2,32,4),device=x.device');(R/'replace_core.py').write_text(s)
s=(R/'b1_fused.cu').read_text().replace('*saved_rs;', '*saved_rs,*xhat_ptr;')
s=s.replace('for(int q=0;q<4;++q)tma_load_2d(q<3?x+q*16384:sm,&p.xhat,bars+4,row+16*q,0);', '''for(int q=0;q<4;++q){void* dst=q<3?x+q*16384:sm;const float* src=p.xhat_ptr+(size_t)(row/64)*16384+q*4096;
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"::"r"(smem_u32(dst)),"l"(src),"r"(16384),"r"(smem_u32(bars+4)):"memory");}''')
s=s.replace('for(int k=0;k<16;++k){uint32_t f[4];int c=16*k+2*(lane&3);', 'for(int k=0;k<16;++k){uint32_t f[4];int c=16*k+2*(lane&3);float h[8];xhat_frag(h,sm,slot,w,lane,k);')
s=s.replace('float a=xhat_at(sm,slot,r,cc),b=xhat_at(sm,slot,r,cc+1);','float a=h[j*2],b=h[j*2+1];')
(R/'b1_fused.cu').write_text(s)
(R/'xhat_read.inc').write_text('''// Native FP32 normalized-value layout. No tri / mean input.
static_assert(XHAT_FP32==1 && !PRODUCER && LOWREG && GATE_PHASE,"native saved-xhat schedule only");
TMN_DEVI void xhat_frag(float (&h)[8],uint8_t* sm,int slot,int w,int lane,int ks){
 uint8_t* p=ks<12?sm+16384+slot*49152:sm;int local=ks<12?ks:ks-12;
 uint32_t offset=local*4096+w*1024+lane*16;
 uint4 a=lds128(smem_u32(p)+offset),b=lds128(smem_u32(p)+offset+512);
 h[0]=__uint_as_float(a.x);h[1]=__uint_as_float(a.y);h[2]=__uint_as_float(a.z);h[3]=__uint_as_float(a.w);
 h[4]=__uint_as_float(b.x);h[5]=__uint_as_float(b.y);h[6]=__uint_as_float(b.z);h[7]=__uint_as_float(b.w);
}
''')
s=(R/'b1_lowreg.inc').read_text()
s=s.replace('uint32_t dn[4];\n   static_for<4>', 'uint32_t dn[4];float h[8];xhat_frag(h,sm,slot,w,lane,n*4+q);\n   static_for<4>')
s=s.replace('float xa=xhat_at(sm,slot,rr?rb:ra,c),xb=xhat_at(sm,slot,rr?rb:ra,c+1);', 'float xa=h[j*2],xb=h[j*2+1];')
s=s.replace('uint32_t dn[4],out[4];','uint32_t dn[4],out[4];float h[8];xhat_frag(h,sm,slot,w,lane,n*4+q);')
s=s.replace('float xaa=xhat_at(sm,slot,ra,c),xab=xhat_at(sm,slot,ra,c+1);','float xaa=h[j*2],xab=h[j*2+1];').replace('float xba=xhat_at(sm,slot,rb,c),xbb=xhat_at(sm,slot,rb,c+1);','float xba=h[j*2+2],xbb=h[j*2+3];')
(R/'b1_lowreg.inc').write_text(s)
s=(R/'replace_plan.py').read_text().replace('rstd,dg,dwg,dwp,dt,dgo','rstd,xhat,dg,dwg,dwp,dt,dgo');(R/'replace_plan.py').write_text(s)
s=(P/'run.sbatch').read_text().replace('trimul_xhat_push_20260921','trimul_xhat_native_20260921');(R/'run.sbatch').write_text(s)
print('Generated native normalized layout')
