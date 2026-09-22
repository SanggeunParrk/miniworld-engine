"""Reuse the selected C64-per-warpgroup LN schedule after two-row WGMMA."""
from pathlib import Path
import json
R = Path(__file__).resolve().parent
selected = (R / 'front_ring96_cache3.cu').read_text()
base = (R / 'front_ring_pair256.cu').read_text()
start = selected.index('  float mu[2]=', selected.index('TMN_DEVI void input_role'))
end = selected.index('\n }\n p.partln', start)
body = selected[start:end]
body = body.replace('reinterpret_cast<float*>(lnsm+36864)',
                    'reinterpret_cast<float*>(sm+69632)')
body = body.replace('reinterpret_cast<float*>(lnsm+32768)',
                    'reinterpret_cast<float*>(sm+65536)')
helper = '''TMN_DEVI void pair_ln64(const Params& p,uint8_t* sm,const float* gamma,int base_row,float& running){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int ra=w*16+lane/4,rb=ra+8;
 for(int sub=0;sub<2;++sub){
  int row=base_row+sub*64;uint8_t* lnsm=sm+sub*32768;
  uint8_t* value=sm+PAIR_VALUE+sub*16384+wi*8192;
  float acc[32];
  static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4);
   acc[q*4]=get(value,ra,c);acc[q*4+1]=get(value,ra,c+1);
   acc[q*4+2]=get(value,rb,c);acc[q*4+3]=get(value,rb,c+1);
  });
''' + body + '\n }\n}\n'
a = base.index('TMN_DEVI void pair_ln(')
b = base.index('TMN_DEVI void input_role(', a)
s = base[:a] + helper + base[b:]
s = s.replace('float run_g=0,run_b=0;', 'float running=0;')
s = s.replace('pair_ln(p,sm,gamma,row,run_g,run_b);',
              'pair_ln64(p,sm,gamma,row,running);')
old = ''' float* tmp=reinterpret_cast<float*>(sm);tmp[wi*256+tid]=run_g;tmp[wi*256+128+tid]=run_b;
 allsync();p.partln[split*256+threadIdx.x]=tmp[threadIdx.x]+tmp[256+threadIdx.x];'''
assert old in s
s = s.replace(old, ' p.partln[split*256+threadIdx.x]=running;')
name = 'front_ring_pair256_ln64'
(R / (name + '.cu')).write_text(s)
(R / (name + '.launch.json')).write_text((R / 'front_ring_pair256.launch.json').read_text())
print(name)
