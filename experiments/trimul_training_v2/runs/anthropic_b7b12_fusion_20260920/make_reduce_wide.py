from pathlib import Path
p=Path(__file__).resolve().parent
for base in ['front_prefetch_lnpair_storepipe','front_ring96_cache3']:
 for htile in [4,8,16,32]:
  s=(p/(base+'.cu')).read_text()
  old=' for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);'
  assert old in s
  code='''
 constexpr int HT=%d,NT=1024/HT;
 if(blockIdx.x<NT){int tile=blockIdx.x,group=tile/(128/HT),kind=(tile/(64/HT))%%2,hbase=(tile%%(64/HT))*HT;
  __nv_bfloat16* redsm=reinterpret_cast<__nv_bfloat16*>(sm);
  for(int z=threadIdx.x;z<HT*128;z+=256){int h=z/128,c=z%%128,j=kind*8192+(hbase+h)*128+c;float v=0;
   for(int q=0;q<DW_SPLITS*2;++q)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*2+q)*16384+j];
   redsm[h*129+c]=__float2bfloat16_rn(v);
  }allsync();
  for(int z=threadIdx.x;z<HT*64;z+=256){int c=z/(HT/2),h=(z%%(HT/2))*2,out=(group/4)*2+(kind==0?1:0),gh=(group%%4)*64+hbase+h;
   uint32_t pair=static_cast<unsigned>(__bfloat16_as_ushort(redsm[h*129+c]))|(static_cast<unsigned>(__bfloat16_as_ushort(redsm[(h+1)*129+c]))<<16);
   *reinterpret_cast<uint32_t*>(p.dw+(out*128+c)*256+gh)=pair;
  }
 }
 if(blockIdx.x==NT)reduce_at(p,131072+threadIdx.x);
'''%htile
  s=s.replace(old,code);name=base+'_redwide%d'%htile;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(base+'.launch.json')).read_text())
