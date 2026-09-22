from pathlib import Path
import re
p=Path(__file__).resolve().parent;s=(p/'front_kindunroll8_inter.cu').read_text()
a=s.index('TMN_DEVI void input_role');s=s[:a]+'''TMN_DEVI void load_ln_next(const Params& p,uint8_t* sm,uint64_t* bar,int row){if(threadIdx.x)return;mbar_arrive_expect_tx(bar,32768);for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,bar,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,bar,c*64,row);}}\n'''+s[a:]
a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);f=s[a:b]
f=f.replace('for(int tile=split;tile<p.tiles;tile+=DXCOUNT){','if(split<p.tiles)issue_gate(p,sm,bar+2,split*64);int round=0;for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){')
f=f.replace('issue_gate(p,sm,bar,row);mbar_wait(bar,0);','mbar_wait(bar+2,round&1);').replace('1^(h/2)^slot','h/2')
f=f.replace('if(h<2)load_p(p,sm,bar,side,h+2);','if(h<2)load_p(p,sm,bar,side,h+2);if(side==1&&h==1)load_ln_next(p,sm+65536,bar+3,row);')
a1=f.index('  // B10 outputs BF16');f=f[:a1]+'  if(tile+DXCOUNT<p.tiles)issue_gate(p,sm,bar+2,(tile+DXCOUNT)*64);\n'+f[a1:]
start=f.index('  if(threadIdx.x==0){mbar_arrive_expect_tx(bar,32768);');end=f.index('  float mu[2]',start);f=f[:start]+'  uint8_t* lnsm=sm+65536;mbar_wait(bar+3,round&1);\n'+re.sub(r'\bsm\b','lnsm',f[end:])
s=s[:a]+f+s[b:];s=s.replace('uint64_t bar[2]','uint64_t bar[4]').replace('mbar_init(bar,1);mbar_init(bar+1,1);','for(int i=0;i<4;++i)mbar_init(bar+i,1);')
(p/'front_kindprefetch.cu').write_text(s);(p/'front_kindprefetch.launch.json').write_text((p/'front_kindunroll8_inter.launch.json').read_text())
