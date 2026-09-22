from pathlib import Path
import json
r=Path(__file__).resolve().parent
s=(r/'dual_n32.cu').read_text()
s='// 3 warpgroups:128 dW registers/thread,12 warps/CTA; same64-row fusion.\n#define CTA_THREADS 384\n'+s
s=s.replace('named_bar_sync(0,256)','named_bar_sync(0,384)').replace('__launch_bounds__(256,1)','__launch_bounds__(384,1)')
s=s.replace('float* mus=stats+256','float* mus=stats+384')
s=s.replace('gam[threadIdx.x]=p.gamma[threadIdx.x];','if(threadIdx.x<256)gam[threadIdx.x]=p.gamma[threadIdx.x];')
s=s.replace('for(int qn=0;qn<4;++qn){int n=wi*4+qn;','for(int qn=0;qn<3;++qn){int n=wi+qn*3;if(n>=8)continue;')
s=s.replace('stats[r*2+1]+stats[128+r*2+1]','(stats[r*2+1]+stats[128+r*2+1])+stats[256+r*2+1]').replace('stats[r*2]+stats[128+r*2]','(stats[r*2]+stats[128+r*2])+stats[256+r*2]')
s=s.replace('stats[(r+1)*2+1]+stats[128+(r+1)*2+1]','(stats[(r+1)*2+1]+stats[128+(r+1)*2+1])+stats[256+(r+1)*2+1]').replace('stats[(r+1)*2]+stats[128+(r+1)*2]','(stats[(r+1)*2]+stats[128+(r+1)*2])+stats[256+(r+1)*2]')
s=s.replace('for(int b=0;b<8;++b){int c=wi*128+b*16+w*4+lane/8;','for(int b=0;b<6;++b){int c=wi*16+b*48+w*4+lane/8;if(c>=256)continue;')
s=s.replace('for(int c=wi*128;c<(wi+1)*128;c+=16)','for(int c=wi*16;c<256;c+=48)')
s=s.replace('float acc[3][64]={}','float acc[2][64]={}').replace('static_for<3>','static_for<2>').replace('int t=wi*3+n','int t=wi*2+n').replace(';fence_regs(acc[2])','')
s=s.replace('i+=256','i+=384').replace('j+=256','j+=384').replace('i/256','i/384').replace('blockIdx.x*256+threadIdx.x','blockIdx.x*384+threadIdx.x').replace('i+=UCOUNT*256','i+=UCOUNT*384')
# The separate fallback reducer intentionally stays at194 CTAs x256 threads.
s=s.replace('int i=blockIdx.x*384+threadIdx.x;\n if(i<49152)','int i=blockIdx.x*256+threadIdx.x;\n if(i<49152)')
(r/'dual_wg3.cu').write_text(s)
(r/'dual_wg3.launch.json').write_text(json.dumps({'threads':384}))
(r/'dual_wg3.py').write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,*args,**kwargs):\n        super().__init__(*args,source="dual_wg3",**kwargs)\n')
