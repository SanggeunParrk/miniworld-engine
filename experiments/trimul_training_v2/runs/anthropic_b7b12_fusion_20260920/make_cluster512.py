from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_cluster_pipe.cu').read_text()
helpers='\nTMN_DEVI void sync512(){named_bar_sync(0,512);}\n#define allsync sync512\n'
for name,ta in [('mma_gate32',0),('mma_front32',1)]:
 regs=','.join('%%%d'%i for i in range(16));outs=','.join('"+f"(d[%d])'%i for i in range(16));helpers+='''TMN_DEVI void %s(float (&d)[16],uint64_t a,uint64_t b,int scale){asm volatile("{.reg .pred p;setp.ne.b32 p,%%18,0;wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%s},%%16,%%17,p,1,1,%d,0;}" : %s : "l"(a),"l"(b),"r"(scale));}\n'''%(name,regs,ta,outs)
s=s.replace('#include <cooperative_groups.h>','#include <cooperative_groups.h>'+helpers).replace('__launch_bounds__(256,1)','__launch_bounds__(512,1)')
a=s.index('TMN_DEVI void producer');b=s.index('TMN_DEVI void load_weight',a);f=s[a:b];f=f.replace('float acc[2][64]={}','float acc[64]={}').replace('j<16;++j','j<8;++j').replace('threadIdx.x+j*256','threadIdx.x+j*512')
x=f.index('   static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;fence_regs(acc[half])');y=f.index('\n   if((r+1)%segment',x)
f=f[:x]+'''   int half=wi%2;fence_regs(acc);wgmma_fence();static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight128(acc,smem_desc(smem_u32(out+(wi/2)*16384+half*8192+k*32),16,1024,1),smem_desc(smem_u32(inp+49152+k*2048),8192,1024,1),r%segment>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();'''+f[y:]
f=f.replace('+wi*16384','+(wi/2)*16384').replace('static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;static_for<16>','{static_for<16>').replace('acc[half][','acc[').replace('});});\n   }','});}\n   }');s=s[:a]+f+s[b:]
a=s.index('TMN_DEVI void consumer');b=s.index('TMN_DEVI void reduce_at',a);f=s[a:b]
f=f.replace('gate_packed[16]','gate_packed[8]').replace('float acc[32]','float acc[16]').replace('float gate[32]','float gate[16]').replace('mma_dgrad(gate','mma_gate32(gate').replace('mma_input64(acc','mma_front32(acc').replace('static_for<16>([&](auto ji)','static_for<8>([&](auto ji)')
f=f.replace('+wi*16384','+(wi/2)*16384+(wi%2)*4096')
f=f.replace('static_for<4>([&](auto qi)','static_for<2>([&](auto qi)').replace('wi*64+c','wi*32+c')
for op in ['get','put']:
 for base in ['sm+wi*8192','sm+16384+wi*8192']:
  for row in ['r','ra','rb']:
   for col in ['c','c+1']:
    before='%s(%s,%s,%s%s'%(op,base,row,col,')' if op=='get' else ',');after='%s(%s,%s,%s+(wi%%2)*32%s'%(op,base.replace('wi*8192','(wi/2)*8192'),row,col,')' if op=='get' else ',');f=f.replace(before,after)
f=f.replace('stats[ra*2]+stats[128+ra*2]','(stats[ra*2]+stats[128+ra*2])+(stats[256+ra*2]+stats[384+ra*2])').replace('stats[rb*2]+stats[128+rb*2]','(stats[rb*2]+stats[128+rb*2])+(stats[256+rb*2]+stats[384+rb*2])').replace('stats[ra*2+1]+stats[128+ra*2+1]','(stats[ra*2+1]+stats[128+ra*2+1])+(stats[256+ra*2+1]+stats[384+ra*2+1])').replace('stats[rb*2+1]+stats[128+rb*2+1]','(stats[rb*2+1]+stats[128+rb*2+1])+(stats[256+rb*2+1]+stats[384+rb*2+1])')
f=f.replace('running+=(tmp[','if(threadIdx.x<256)running+=(tmp[').replace('p.partln[split*256+threadIdx.x]=running;','if(threadIdx.x<256)p.partln[split*256+threadIdx.x]=running;')
s=s[:a]+f+s[b:];s=s.replace('blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256','blockIdx.x*512+threadIdx.x;i<131328;i+=UCOUNT*512')
(p/'front_cluster512.cu').write_text(s);(p/'front_cluster512.launch.json').write_text('{"direct_weights":true,"wgrad_slices":2,"threads":512}\n')
