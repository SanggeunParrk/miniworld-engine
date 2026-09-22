from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_wait_folding_20260921'
for p in [S/'replace_plan.py',*S.glob('*.cu'),*S.glob('*.cuh'),*S.glob('*.inc')]:
 (R/p.name).write_text(p.read_text())
s=(R/'lowreg_stats.inc').read_text()
helper='''
#ifndef B1_PARAM_HYBRID
#define B1_PARAM_HYBRID 0
#endif
#if B1_PARAM_HYBRID
TMN_DEVI float param_reduce_hybrid(uint8_t* sm,int slot,int c,int base){
 constexpr int rows=8>>B1_PARAM_HYBRID;
 float parts[4];
 static_for<4>([&](auto ww){constexpr int w=decltype(ww)::value;float v[rows];
  static_for<rows>([&](auto rr){constexpr int r=decltype(rr)::value;asm volatile("ld.shared.f32 %0,[%1];":"=f"(v[r]):"r"(param_ptr(sm,slot,base+w*rows+r,c)):"memory");});
  if constexpr(rows==4)parts[w]=__fadd_rn(__fadd_rn(v[0],v[1]),__fadd_rn(v[2],v[3]));
  else if constexpr(rows==2)parts[w]=__fadd_rn(v[0],v[1]);else parts[w]=v[0];
 });
 return __fadd_rn(__fadd_rn(parts[0],parts[1]),__fadd_rn(parts[2],parts[3]));
}
#endif
'''
pos=s.index('TMN_DEVI void param_sync()');s=s[:pos]+helper+s[pos:]
old='''#if B1_PARAM_SHARED
    param_store(sm,slot,w*8+lane/4,c,dga,dgb);'''
new='''#if B1_PARAM_HYBRID
    static_for<B1_PARAM_HYBRID>([&](auto kk){constexpr int sh=4<<decltype(kk)::value;
     dga=__fadd_rn(dga,__shfl_xor_sync(0xffffffff,dga,sh));dgb=__fadd_rn(dgb,__shfl_xor_sync(0xffffffff,dgb,sh));
     dba0=__fadd_rn(dba0,__shfl_xor_sync(0xffffffff,dba0,sh));dbb0=__fadd_rn(dbb0,__shfl_xor_sync(0xffffffff,dbb0,sh));
    });
    if((lane&(((1<<B1_PARAM_HYBRID)-1)*4))==0){
     int rr=w*(8>>B1_PARAM_HYBRID)+(lane>>(2+B1_PARAM_HYBRID));
     param_store(sm,slot,rr,c,dga,dgb);param_store(sm,slot,(32>>B1_PARAM_HYBRID)+rr,c,dba0,dbb0);
    }
#elif B1_PARAM_SHARED
    param_store(sm,slot,w*8+lane/4,c,dga,dgb);'''
assert old in s;s=s.replace(old,new)
old='''#if B1_PARAM_SHARED
 red[c]+=param_reduce(sm,slot,c);param_sync();'''
new='''#if B1_PARAM_HYBRID
 red[c]+=param_reduce_hybrid(sm,slot,c,0);
 red[256+c]+=param_reduce_hybrid(sm,slot,c,32>>B1_PARAM_HYBRID);
#elif B1_PARAM_SHARED
 red[c]+=param_reduce(sm,slot,c);param_sync();'''
assert old in s;s=s.replace(old,new);(R/'lowreg_stats.inc').write_text(s)
