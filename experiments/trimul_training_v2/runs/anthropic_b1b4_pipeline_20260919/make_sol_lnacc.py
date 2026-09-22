"""Accumulate per-thread LN parameter gradients across tiles before reducing."""
from pathlib import Path
r = Path(__file__).resolve().parent
s = (r/'dual_balanced.cu').read_text()
s = s.replace('int m0,int slot){', 'int m0,int slot,float (&pg)[32],float (&pb)[32]){')
start = s.index('    // Same columns, eight row groups')
end = s.index('\n   });', start)
s = s[:start] + '''    constexpr int z=(nl*4+q)*4+pair*2;
    pg[z]+=dga;pg[z+1]+=dgb;pb[z]+=dba0;pb[z+1]+=dbb0;
''' + s[end:]
s = s.replace(' float* tmp=reinterpret_cast<float*>(sm+65536);', '')
start = s.index(' allsync(); // Publish all four warps')
end = s.index(' fence_proxy_async();sync_group();', start)
s = s[:start] + s[end:]
s = s.replace(' int split=blockIdx.x-DWCOUNT;int mi=0,round=0;',
''' int split=blockIdx.x-DWCOUNT;int mi=0,round=0;
 float pg[32]={},pb[32]={};''')
s = s.replace('dual_dgrad(p,sm,it*64,slot);', 'dual_dgrad(p,sm,it*64,slot,pg,pb);')
old = ' for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];'
new = ''' // All input tiles and dtri TMA stores are complete. Reuse dead slot0
 // for the one-time warp and CTA reduction of parameter gradients.
 float* tmp=reinterpret_cast<float*>(sm);
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x%128)/32;
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;
   static_for<2>([&](auto pi){constexpr int pair=decltype(pi)::value;
    constexpr int z=(nl*4+q)*4+pair*2;
    int c=(wi*2+nl)*64+q*16+2*(lane%4)+8*pair;
    float ga=pg[z],gb=pg[z+1],ba=pb[z],bb=pb[z+1];
#pragma unroll
    for(int sh=4;sh<32;sh*=2){ga+=__shfl_xor_sync(0xffffffff,ga,sh);gb+=__shfl_xor_sync(0xffffffff,gb,sh);ba+=__shfl_xor_sync(0xffffffff,ba,sh);bb+=__shfl_xor_sync(0xffffffff,bb,sh);}
    if(lane<4){tmp[w*512+c]=ga;tmp[w*512+c+1]=gb;tmp[w*512+256+c]=ba;tmp[w*512+256+c+1]=bb;}
   });
  });
 });
 allsync();
 int c=threadIdx.x;
 p.partln[split*512+c]=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);
 p.partln[split*512+256+c]=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);'''
assert old in s
s = s.replace(old,new)
(r/'dual_ln_acc.cu').write_text('// Experiment: accumulate LN parameter gradients across tiles.\n'+s)
(r/'dual_ln_acc.py').write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,"dual_ln_acc")\n')
