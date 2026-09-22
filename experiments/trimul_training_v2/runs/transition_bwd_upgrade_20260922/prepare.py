from pathlib import Path
R=Path(__file__).resolve().parent;s=(R/'baseline.cu').read_text()
start=s.index('  const int idx = blockIdx.x * blockDim.x + threadIdx.x;',s.index('extern "C" __global__ void reduce_partials'))
end=s.index('  const int which = idx',start)
parallel='''  const int tid=threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  if (blockIdx.x >= 768) {
    const int c=(blockIdx.x-768)*32+(tid&31), part=tid>>5;
    float g=0.f,b=0.f;
    for(int r=part;r<NDX*8;r+=8){g+=dgbw[(size_t)r*256+c];b+=dgbw[(size_t)r*256+128+c];}
    __shared__ float scratch[512];scratch[tid]=g;scratch[256+tid]=b;__syncthreads();
    if(tid<32){float gs=0.f,bs=0.f;
      #pragma unroll
      for(int p=0;p<8;++p){gs+=scratch[p*32+tid];bs+=scratch[256+p*32+tid];}
      dgam[c]=gs;dbeta[c]=bs;
    }return;
  }
'''
p=s[:start]+parallel+s[end:];(R/'parallel.cu').write_text(p)
old='''        row[col] += dgp[2 * g]; row[col + 1] += dgp[2 * g + 1];
        row[128 + col] += dbp[2 * g]; row[128 + col + 1] += dbp[2 * g + 1];'''
new='''        float2 u=*reinterpret_cast<float2*>(row+col),v=*reinterpret_cast<float2*>(row+128+col);
        u.x+=dgp[2*g];u.y+=dgp[2*g+1];v.x+=dbp[2*g];v.y+=dbp[2*g+1];
        *reinterpret_cast<float2*>(row+col)=u;*reinterpret_cast<float2*>(row+128+col)=v;'''
assert old in p;(R/'vector_parallel.cu').write_text(p.replace(old,new))
c=p.replace('  for (int q = tid; q < 2048; q += 256) dgw[q] = 0.f;','  float ln_run=0.f;')
start=c.index('    if (lane < 4) {',c.index('// ---- dgamma / dbeta of this tile'))
end=c.index('    if (wtid == 0) mbar_arrive(in_free + buf);',start)
c=c[:start]+'''    // Current xn/x storage is dead after this CTA has completed its LN epilogue.
    __syncthreads();
    float* tmp=reinterpret_cast<float*>(sm+X_IN+buf*X_INB+X_XN);
    if(lane<4){float* row=tmp+(wg*4+warp)*256;
      #pragma unroll
      for(int g=0;g<16;++g){int col=8*g+2*lane;
        row[col]=dgp[2*g];row[col+1]=dgp[2*g+1];row[128+col]=dbp[2*g];row[128+col+1]=dbp[2*g+1];
      }
    }
    __syncthreads();
    float tile_sum=0.f;
    #pragma unroll
    for(int w=0;w<8;++w)tile_sum+=tmp[w*256+tid];
    ln_run+=tile_sum;
    __syncthreads();
'''+c[end:]
needle='''  }
}

extern "C" __global__ void __launch_bounds__(256, 1)
transition_bwd_fused''';assert needle in c
c=c.replace(needle,'''  }
  dgw[tid]=ln_run;
}

extern "C" __global__ void __launch_bounds__(256, 1)
transition_bwd_fused''')
c=c.replace('r<NDX*8;r+=8','r<NDX;r+=8').replace('dgbw[(size_t)r*256+c]','dgbw[(size_t)r*2048+c]').replace('dgbw[(size_t)r*256+128+c]','dgbw[(size_t)r*2048+128+c]')
(R/'compact_parallel.cu').write_text(c)
