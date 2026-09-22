from pathlib import Path
p=Path(__file__).resolve().parent;src=(p/'front_prefetch_lnpair.cu').read_text()
for mode in ['dense','interleave']:
 helper='''
#define COMPACT_SIGMOID_LUT 1
__device__ float sigmoid_lut_compact[3328];
extern "C" __global__ void init_sigmoid_lut(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<3328){int ab=ABSINDEX+0x3b80,sign=SIGNINDEX;sigmoid_lut_compact[i]=math::sigmoid(bf16lo(ab|(sign<<15)));}}
TMN_DEVI float gate_bits(unsigned raw){unsigned ab=raw&0x7fff;if(ab>=0x3b80&&ab<0x4200){unsigned i=LOOKUP;return __ldg(sigmoid_lut_compact+i);}return math::sigmoid(bf16lo(raw));}
'''
 if mode=='dense':helper=helper.replace('ABSINDEX','i%1664').replace('SIGNINDEX','i/1664').replace('LOOKUP','ab-0x3b80+(raw>>15)*1664')
 else:helper=helper.replace('ABSINDEX','i/2').replace('SIGNINDEX','i%2').replace('LOOKUP','(ab-0x3b80)*2+(raw>>15)')
 a=src.index('TMN_DEVI void glu_small');s=src[:a]+helper+src[a:];s=s.replace('math::sigmoid(bf16lo(gl))','gate_bits(gl&0xffff)').replace('math::sigmoid(bf16hi(gl))','gate_bits(gl>>16)');name='front_prefetch_compact_'+mode;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
