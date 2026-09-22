from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;src=R.parent/'trimul_all_recompute_20260920/no_save_k3.cu';s=src.read_text()
s=s.replace('const __nv_bfloat16* dropscale;','const __nv_bfloat16* dropscale; CUtensorMap tm_lnin,tm_lnout; __nv_bfloat16 *lnin,*lnout;')
pos=s.index('template <class G, int LNM, bool UPD = false>')
helper=r'''
// Store the existing BF16 affine LayerNorm result; no additional LN math,
// statistics, projection, or gate saves. Each row is emitted once.
template<int KS>
TMN_DEVI void emit_ln(const uint32_t (&f)[KS][4],const CUtensorMap* map,
                     __nv_bfloat16* out,uint8_t* stage,int iw,int jw,int n){
 const int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
#if MW_STORE_METHOD == 0
 #pragma unroll
 for(int c64=0;c64<KS/4;++c64){
  if(lane==0)tma_store_wait_read<0>();__syncwarp();
  #pragma unroll
  for(int slab=0;slab<4;++slab){int k=4*c64+slab;
   stsm_x4(smem_u32(stage)+swz128(lane%8+8*(mat&1),(2*slab+(mat>>1))*16),f[k][0],f[k][1],f[k][2],f[k][3]);
  }
  fence_proxy_async();__syncwarp();
  if(lane==0){tma_store_3d(map,stage,64*c64,jw+16*w,iw);tma_store_commit();}
 }
 if(lane==0)tma_store_wait_read<0>();__syncwarp();
#else
 #pragma unroll
 for(int k=0;k<KS;++k){
  #pragma unroll
  for(int j=0;j<4;++j){int row=jw+16*w+lane/4+8*(j&1),col=k*16+2*(lane%4)+8*(j>>1);
   if(iw<n && row<n)stg32(out+((size_t)iw*n+row)*(KS*16)+col,f[k][j]);
  }
 }
#endif
}
'''
s=s[:pos]+helper+s[pos:]
s=s.replace('    PF(3);                                               // 3: LN_out','''#if MW_SAVE_OUT
    if(!SPLITN || cw==0)emit_ln(fx,&tp.tm_lnout,tp.lnout,sOut+(4*cw+wiw)*OB,iw,jw,p.N);
#endif
    PF(3);                                               // 3: LN_out''')
s=s.replace('    PF(6);                                               // 6: LN_in','''#if MW_SAVE_IN
    if(!SPLITN || cw==0)emit_ln(fz,&tp.tm_lnin,tp.lnin,sOut+(4*cw+wiw)*OB,iw,jw,p.N);
#endif
    PF(6);                                               // 6: LN_in''')
(R/'ln_only_k3.cu').write_text(s)
(R/'derivation.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'save_location':'K3 already recomputes both LNs; K1 unchanged','save_values':'BF16 affine LN outputs only; no mean/rstd or other intermediates added'},indent=2))
