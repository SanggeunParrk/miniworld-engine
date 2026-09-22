from pathlib import Path
R=Path(__file__).resolve().parent
for base in ('parallel','compact_parallel'):
 s=(R/(base+'.cu')).read_text();pos=s.index('  const int which = idx',s.index('extern "C" __global__ void reduce_partials'))
 code='''  if (blockIdx.x >= 512) {
    // Reduce a 16x16 tile, then transpose in shared memory for coalesced dWs stores.
    const int tile=blockIdx.x-512, h0=(tile/8)*16, d0=(tile%8)*16;
    const int h=h0+tid/16, d=d0+tid%16, slice=h/64, hs=h%64;
    float v=0.f;
    for(int r=0;r<DW_REPL;++r)v+=ws[((size_t)(r*8+slice)*3+2)*HS*D_+hs*D_+d];
    __shared__ float transpose[16][17];
    transpose[tid/16][tid%16]=v;__syncthreads();
    dWs[(size_t)(d0+tid/16)*H_+h0+tid%16]=__float2bfloat16_rn(transpose[tid%16][tid/16]);
    return;
  }
'''
 (R/(base+'_transpose.cu')).write_text(s[:pos]+code+s[pos:])
