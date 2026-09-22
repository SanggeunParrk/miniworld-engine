from pathlib import Path
r=Path(__file__).resolve().parent
ptx=['{','.reg .f32 d<128>;','.reg .b64 a,b,pa,pb,pc,pd,out,adr,adr2;','.reg .u32 iter,niter,tid,lane,warp,row,col,off,stride,delta;','.reg .pred more,useacc;','mov.u32 iter,0;','mov.u32 niter,%4;','mov.b64 pa,%0;','mov.b64 pb,%1;','mov.b64 pc,%2;','mov.b64 pd,%3;','mov.b64 out,%5;','mov.u32 stride,%6;','mov.u32 delta,%7;','LOOP:','bar.sync 0,512;','setp.ne.u32 useacc,iter,0;','wgmma.fence.sync.aligned;']
for n,(aa,bb) in enumerate([('pa','pb'),('pc','pd')]):
 for k in range(4):
  ptx+= [f'add.u64 a,{aa},{k*128};',f'add.u64 b,{bb},{k*128};',f'wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {{{",".join("d"+str(q+n*64) for q in range(64))}}},a,b,{"useacc" if k==0 else "1"},1,1,1,1;']
ptx+=['wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;','bar.sync 0,512;','add.u32 iter,iter,1;','setp.lt.u32 more,iter,niter;','@more bra LOOP;','mov.u32 tid,%tid.x;','and.b32 lane,tid,31;','and.b32 warp,tid,127;','shr.u32 warp,warp,5;','shr.u32 row,lane,2;','mad.lo.u32 row,warp,16,row;','and.b32 col,lane,3;','shl.b32 col,col,1;','mad.lo.u32 off,row,stride,col;','mul.wide.u32 adr,off,4;','add.u64 out,out,adr;']
for n in range(2):
 for q in range(16):
  ptx+= [f'add.u64 adr,out,{q*32};','mul.wide.u32 adr2,stride,32;','add.u64 adr2,adr,adr2;',f'st.global.v2.f32 [adr],{{d{n*64+q*4},d{n*64+q*4+1}}};',f'st.global.v2.f32 [adr2],{{d{n*64+q*4+2},d{n*64+q*4+3}}};']
 if n==0:ptx+=['cvt.u64.u32 adr,delta;','add.u64 out,out,adr;']
ptx+=['}']
# Literal PTX special register requires %% in extended asm.
src='#include "unified_v2.cu"\nTMN_DEVI void asm_weight(const Params& p,uint8_t* sm){\nconst int wi=__shfl_sync(0xffffffffu,threadIdx.x/128,0);int count=p.tiles*(blockIdx.x+1)/UCOUNT-p.tiles*blockIdx.x/UCOUNT;\nuint8_t* sa=wi==1?sm+49152:sm+(wi-2)*8192;uint8_t* sb=wi==1?sm+16384:sm+65536;\nuint64_t a0=smem_desc(smem_u32(sa),16,1024,1),b0=smem_desc(smem_u32(sb),8192,1024,1),a1=smem_desc(smem_u32(sa+(wi==1?8192:0)),16,1024,1),b1=smem_desc(smem_u32(sb+(wi==1?0:16384)),8192,1024,1);\nfloat* out=p.partw+((wi-1)*UCOUNT+blockIdx.x)*16384;int stride=wi==1?128:256,delta=wi==1?32768:512;\nasm volatile(\n'
for z in ptx:src+='"'+z.replace('%tid','%%tid')+'\\n"\n'
src+=': : "l"(a0),"l"(b0),"l"(a1),"l"(b1),"r"(count),"l"(out),"r"(stride),"r"(delta) : "memory");}\n'
src+='''extern "C" __global__ __launch_bounds__(512,1) void unifiedasm_b1b4(__grid_constant__ const Params p){
extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bars[2];__shared__ int last[4];
if(threadIdx.x==0){for(int i=0;i<2;++i)mbar_init(bars+i,1);fence_barrier_init();}
for(int i=threadIdx.x;i<2048;i+=512)reinterpret_cast<float*>(sm+180224)[i]=0;allsync();
const int wg=__shfl_sync(0xffffffffu,threadIdx.x>>7,0);
if(wg==0){setmaxnreg_dec<80>();urun<false>(p,sm,bars,last);}
else{setmaxnreg_inc<144>();asm_weight(p,sm);}
}
'''
(r/'unifiedasm.cu').write_text(src)
a=(r/'unified.py').read_text().replace("source=R/'unified.cu'","source=R/'unifiedasm.cu'").replace("source.read_bytes()+(R/'fused.cu').read_bytes()","source.read_bytes()+(R/'unified_v2.cu').read_bytes()+(R/'fused.cu').read_bytes()").replace("kernel('unified_b1b4')","kernel('unifiedasm_b1b4')");(r/'unifiedasm.py').write_text(a)
a=(r/'check_reduce.py').read_text().replace('from unified import','from unifiedasm import').replace('for n in (384,768):','for n in (64,384,768):').replace('for part in (0,1):','for part in (1,):').replace('132,part','min(132,n*n//64),part');(r/'check_unifiedasm.py').write_text(a)
