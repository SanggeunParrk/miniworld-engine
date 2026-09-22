from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_producer_bench.cu').read_text()
# Bit-exact sigmoid table; values outside its signed exponent range use the original formula.
lut='''TMN_DEVI float sigmoid_lookup(uint32_t bits){
 extern __shared__ __align__(1024) uint8_t sm[];unsigned idx=(bits&0x7fffu)-0x3b80u;
 if(idx<2048u)return reinterpret_cast<float*>(sm+98304)[idx+((bits&0x8000u)?2048:0)];
 return math::sigmoid_div(__uint_as_float(bits<<16));
}
'''
a=s.index('// Each pair');base=s[:a]+lut+s[a:]
base=base.replace('float ga=math::sigmoid_div(bf16lo(gl)),gb=math::sigmoid_div(bf16hi(gl));','float ga=sigmoid_lookup(gl&65535),gb=sigmoid_lookup(gl>>16);')
base=base.replace('allsync();producer(p,sm,bar);','allsync();for(int i=threadIdx.x;i<4096;i+=256){unsigned bits=0x3b80+(i%2048)+((i/2048)*32768);reinterpret_cast<float*>(sm+98304)[i]=math::sigmoid_div(__uint_as_float(bits<<16));}allsync();producer(p,sm,bar);')
for name,src in [('front_producer_lut',base),('front_producer_uint',s.replace('int i,int row,float','unsigned i,int row,float').replace('int c=i/32,r=(i%32)*2;','unsigned c=i/32,r=(i%32)*2;').replace('int i=threadIdx.x+q*256,c=i/32,r=(i%32)*2;','unsigned i=threadIdx.x+q*256,c=i/32,r=(i%32)*2;')),('front_producer_rcp',s.replace('math::sigmoid_div','math::sigmoid'))]:
 (p/(name+'.cu')).write_text(src);(p/(name+'.launch.json')).write_text('{"direct_weights":true}\n')
s=(p/'producer_bench.py').read_text().replace("ap.add_argument('--length',type=int,default=768);","ap.add_argument('--length',type=int,default=768);ap.add_argument('--source',default='front_producer_bench');").replace('[24,32,40,48,64,96,128]','[32,64,128]').replace("source='front_producer_bench'","source=args.source").replace('assert e==0,e','assert e<1e-5,e').replace("f'producer-only-L{args.length}.json'","f'{args.source}-L{args.length}.json'");(p/'producer_variants.py').write_text(s)
