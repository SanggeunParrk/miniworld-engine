from pathlib import Path
p=Path(__file__).resolve().parent;src=(p/'front_prefetch_lnpair.cu').read_text()
for dw in [False,True]:
 for ln in [False,True]:
  if not dw and not ln:continue
  s=src
  old=' for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);'
  code=''
  if dw:
   code+='''
 __nv_bfloat16* redsm=reinterpret_cast<__nv_bfloat16*>(sm);
 for(int tile=blockIdx.x;tile<512;tile+=UCOUNT){int tid=threadIdx.x,group=tile/64,kind=(tile/32)%2,hbase=((tile/8)%4)*16,cbase=(tile%8)*16;
  int j=kind*8192+(hbase+tid/16)*128+cbase+tid%16;float v=0;
  for(int q=0;q<DW_SPLITS*2;++q)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*2+q)*16384+j];
  redsm[(tid/16)*17+tid%16]=__float2bfloat16_rn(v);allsync();
  int out=(group/4)*2+(kind==0?1:0),c=cbase+tid/16,h=(group%4)*64+hbase+tid%16;
  p.dw[(out*128+c)*256+h]=redsm[(tid%16)*17+tid/16];allsync();
 }
'''
  else:code+=' for(int i=blockIdx.x*256+threadIdx.x;i<131072;i+=UCOUNT*256)reduce_at(p,i);\n'
  if ln:
   code+='''
 {int c=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(c<256){float v=0;
  for(int q=lane;q<DXCOUNT;q+=32)v+=reinterpret_cast<volatile float*>(p.partln)[q*256+c];
  for(int sh=16;sh>0;sh/=2)v+=__shfl_down_sync(0xffffffff,v,sh);
  if(lane==0)(c<128?p.dgam:p.dbeta)[c%128]=v;
 }}
'''
  else:code+=' if(blockIdx.x==0)reduce_at(p,131072+threadIdx.x);\n'
  assert old in s;s=s.replace(old,code);name='front_prefetch_reduce'+('dw' if dw else '')+('ln' if ln else '');(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
