from pathlib import Path
p=Path(__file__).resolve().parent;src=(p/'front_prefetch_lnpair.cu').read_text()
for which in ['dw','both']:
 s=src
 a=s.index('TMN_DEVI void glu_small');b=s.index('TMN_DEVI void load_dw',a);f=s[a:b].replace('glu_small(', 'glu_shared(').replace('int row){','int row,const uint8_t* masksm){').replace('p.mask+row+(tid%32)*2','masksm+(tid%32)*4')
 helper='''TMN_DEVI void mask_load(const Params& p,uint8_t* dst,uint64_t* bar,int row){asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"::"r"(smem_u32(dst)),"l"(p.mask+row),"r"(smem_u32(bar)):"memory");}
'''
 s=s[:b]+f+helper+s[b:]
 s=s.replace('int row,int group){\n if(threadIdx.x)return;','int row,int group,uint8_t* masksm){\n if(threadIdx.x)return;')
 s=s.replace('mbar_arrive_expect_tx(b+slot,40960);\n tma_load_2d(s,&p.pre,b+slot,row,side*512+h*2);','mbar_arrive_expect_tx(b+slot,41088);mask_load(p,masksm+slot*128,b+slot,row);\n tma_load_2d(s,&p.pre,b+slot,row,side*512+h*2);')
 s=s.replace('void weight_role(const Params& p,uint8_t* sm,uint64_t* b){','void weight_role(const Params& p,uint8_t* sm,uint64_t* b,uint8_t* masksm){')
 a=s.index('TMN_DEVI void weight_role');b=s.index('TMN_DEVI void load_g',a);v=s[a:b]
 v=v.replace('split*64,group);','split*64,group,masksm);').replace('(split+DW_SPLITS)*64,group);','(split+DW_SPLITS)*64,group,masksm);').replace('(tile+2*DW_SPLITS)*64,group);','(tile+2*DW_SPLITS)*64,group,masksm);')
 v=v.replace('glu_small(p,s,s+40960,s+49152,tile*64);','glu_shared(p,s,s+40960,s+49152,tile*64,masksm+slot*128);');s=s[:a]+v+s[b:]
 s=s.replace('__shared__ uint64_t bar[4];','__shared__ __align__(128) uint8_t masksm[256];__shared__ uint64_t bar[4];').replace('weight_role(p,sm,bar);','weight_role(p,sm,bar,masksm);')
 if which=='both':
  s=s.replace('void issue_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row){','void issue_gate(const Params& p,uint8_t* sm,uint64_t* bar,int row,uint8_t* masksm){')
  s=s.replace('mbar_arrive_expect_tx(bar,49152);','mbar_arrive_expect_tx(bar,49280);mask_load(p,masksm,bar,row);')
  s=s.replace('const float* gamma){','const float* gamma,uint8_t* masksm){').replace('input_role(p,sm,bar,gamma);','input_role(p,sm,bar,gamma,masksm);')
  s=s.replace('issue_gate(p,sm,bar+2,split*64);','issue_gate(p,sm,bar+2,split*64,masksm);').replace('issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64);','issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64,masksm);')
  s=s.replace('glu_small(p,s,s+16384,sm+81920+h*8192,row);','glu_shared(p,s,s+16384,sm+81920+h*8192,row,masksm);')
 name='front_prefetch_lnpair_masktma_'+which;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
