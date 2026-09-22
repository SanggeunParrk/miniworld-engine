from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_cached_dx.cu').read_text()
s=s.replace('int row=tile*64,ph=round&1;float acc[32]={};uint32_t gatepacked[16];','int row=tile*64,ph=round&1;uint32_t gatepacked[16];').replace('});}release(b,2);\n  static_for<2>', '});}release(b,2);float acc[32]={};\n  static_for<2>')
a=s.index('TMN_DEVI void glu_cached');b=s.index('TMN_DEVI void producer',a)
for chunk,pr,co in [(2,32,224),(2,40,216),(4,40,216),(4,48,208)]:
 sub=s[a:b].replace('static_for<2>([&](auto hi)', 'static_for<%d>([&](auto hi)'%(16//chunk)).replace('pp[8]','pp[%d]'%chunk).replace('static_for<8>([&](auto qi)','static_for<%d>([&](auto qi)'%chunk).replace('half*8','half*%d'%chunk)
 t=(s[:a]+sub+s[b:]).replace('setmaxnreg_dec<48>()','setmaxnreg_dec<%d>()'%pr).replace('setmaxnreg_inc<208>()','setmaxnreg_inc<%d>()'%co)
 name='front_cached_dx_q%d_r%d_%d'%(chunk,pr,co);(p/(name+'.cu')).write_text(t);(p/(name+'.launch.json')).write_text('{"direct_weights":true}\n')
