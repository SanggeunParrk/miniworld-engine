// OPM fused epilogue (sm_90a) -- v4.
//
// The outer-product GEMM is left to cuBLAS in its GROUPED layout
//     O[(i,c), (j,e)] = sum_s A2[(i,c), s] * BT[(j,e), s]      (one NT GEMM; no transpose exists)
// and this kernel consumes that layout directly, doing in ONE pass what the stock module spends
// four HBM passes on: the [i,j,c,e] -> [(i,j),(c,e)] permute, the / num_mask division, the bf16
// cast and the c_hidden^2 -> c_z projection (+bias).  The division sits in the fp32 accumulator
// (the projection is linear, so (O/n)*W == (O*W)/n) which also drops one bf16 rounding of P.
//
// The mask normaliser arrives in fp32: this engine's OuterProductMean counts the mask in fp32 and
// clamps at 1, and at S > 256 a bf16 count is no longer exact, so taking it in fp32 keeps the fused
// path on the module's own semantics rather than on the upstream module's bf16 num_mask.
//
// v1-v3 used the wmma API and were bound by the L1/MIO pipe, not by bandwidth or by the tensor
// cores: NCU put L1/TEX throughput at 98 % with DRAM at 10 %, and 65 % of the warp stalls were
// short-scoreboard + MIO-throttle.  wmma::load_matrix_sync issues a fragment as many narrow
// per-thread loads, and with one B fragment per mma the loads outnumbered the math.  v4 drops to
// mma.sync + ldmatrix:
//   * A: one ldmatrix.x4 per 16x16 tile per k step.
//   * B: the projection weight is CONSTANT, so it is pre-swizzled on the host into mma fragment
//     order; the kernel reads each fragment as one 8 B per-thread load (no ldmatrix needed).
//   * warp tile 2 row x 4 n tiles: 6 loads per 8 mma, against 5 loads per 4 mma in v3.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda.h>
#include <cudaTypedefs.h>

#ifdef MINIWORLD_OPM_OWNED_BLAS

#include <ATen/Context.h>
#include <cublas_v2.h>
#include <map>
#include <memory>
static void opm_owned_count(torch::Tensor norm,torch::Tensor mf) {
 struct Handle {cublasHandle_t value;Handle(){TORCH_CHECK(cublasCreate(&value)==CUBLAS_STATUS_SUCCESS,"OPM cublasCreate");}~Handle(){cublasDestroy(value);}};
 static thread_local std::map<int,std::unique_ptr<Handle>> handles;
 auto& h=handles[mf.get_device()];if(!h)h=std::make_unique<Handle>();
 auto handle=h->value;
 auto ck=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"OPM owned count cuBLAS ",int(s));};
 ck(cublasSetStream(handle,at::cuda::getCurrentCUDAStream()));
 ck(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
 ck(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
 ck(cublasSetMathMode(handle,at::globalContext().allowTF32CuBLAS()?CUBLAS_TF32_TENSOR_OP_MATH:CUBLAS_DEFAULT_MATH));
 size_t bytes=at::cuda::getChosenWorkspaceSize();
 auto workspace=torch::empty({int64_t(bytes)},mf.options().dtype(at::kByte));
 ck(cublasSetWorkspace(handle,workspace.data_ptr(),bytes));
 int n=mf.size(1),s=mf.size(0);float alpha=1,beta=0;
 ck(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,n,n,s,&alpha,mf.data_ptr<float>(),n,mf.data_ptr<float>(),n,&beta,norm.data_ptr<float>(),n));
}

#endif

namespace {

constexpr int CH = 32;             // c_hidden
// d_msa and d_pair are compile-time: the extension is built once per (OPM_CM, OPM_CZ) pair.  The H100
// kernels were written for 64 / 128; 128 / 256 and 128 / 384 reuse the same tiles -- the c_z axis is cut
// into 128-wide slices that run as separate CTAs (epilogue, dWo), or run as a longer contraction (dgrad),
// and the prologue backward widens its LayerNorm phase to 128 channels.
#ifndef OPM_CZ
#define OPM_CZ 128
#endif
#ifndef OPM_CM
#define OPM_CM 64
#endif
constexpr int CM_MSA = OPM_CM;     // d_msa (the LayerNorm width in the prologue backward)
constexpr int CZ = OPM_CZ;         // d_pair
constexpr int ZW = 128;            // c_z slice per CTA (the wgmma n of the epilogue, the m of dWo)
constexpr int NZS = CZ / ZW;       // slices
static_assert(CZ % ZW == 0, "d_pair must be a multiple of 128");
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_HALF) || defined(MINIWORLD_OPM_EPILOGUE_TMA_WIDE) || defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)
static_assert(CZ == 128 && CM_MSA == 64, "the gathered-TMA epilogue variants are written for the MiniWorld width");
#endif
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_HALF)
constexpr int BI = 2;              // one warpgroup shares each gathered TMA stage
#elif defined(MINIWORLD_OPM_EPILOGUE_TMA_WIDE)
constexpr int BI = 8;              // four warpgroups share each projection-weight stage
#else
constexpr int BI = 4;              // i tokens per CTA
#endif
constexpr int BJ = 32;             // j tokens per CTA  (BJ*CH = 512 contiguous columns of O per read)
constexpr int NP = BI * BJ;        // 64 (i,j) pairs per CTA
constexpr int KC = 64;             // K chunk = 2 c values
// The staged tile is XOR-swizzled instead of padded.  A padded row stride cannot satisfy both
// readers: cp.async wants consecutive rows 16 banks apart (so a warp's eight rows pack into whole
// 128-byte wavefronts) while ldmatrix wants them 4 apart (so sixteen rows tile all 32 banks).  At
// KC+16 both were wrong, at KC+8 ldmatrix was ideal and the cp.async writes were 15-way conflicted.
// Swizzling the 16-byte chunk index by the row makes both ideal and drops the padding entirely.
constexpr int KCP = KC;
constexpr int NCHUNK = CH * CH / KC;

template <typename Kernel, typename... Args>
void opm_native_launch(Kernel fn, dim3 grid, dim3 block, Args... args) {
  cudaKernel_t kernel;
  C10_CUDA_CHECK(cudaGetKernel(&kernel, fn));
  void* argv[] = {static_cast<void*>(&args)...};
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, argv, 0,
                                at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Native auxiliary path.  Live BF16/FP32 master parameters are read on every
// replay; no cached casts, ATen arithmetic, or data-dependent host work.
__device__ float opm_read(const void* p, int i, bool fp32) {
  return fp32 ? static_cast<const float*>(p)[i]
              : __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}
__global__ void opm_pack_native(const void* g, const void* b, const void* wa,
    const void* wb, const void* wo, const void* bo, int types, float* gf, float* bf,
    __nv_bfloat16* a, __nv_bfloat16* c, __nv_bfloat16* o, float* ob,
    __nv_bfloat16* ot, __nv_bfloat16* bw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64) { gf[i] = opm_read(g,i,types&1); bf[i] = opm_read(b,i,types&2); }
  if (i < 2048) {
    a[i] = __float2bfloat16_rn(opm_read(wa,i,types&4));
    c[i] = __float2bfloat16_rn(opm_read(wb,i,types&8));
  }
  if (i < 4096) {
    int k = i / 64, ch = i % 64;
    float w = opm_read(ch < 32 ? wa : wb, (ch%32)*64+k, types & (ch<32?4:8));
    bw[i] = __float2bfloat16_rn(w * opm_read(g,k,types&1));
  }
  if (i < 128) ob[i] = __bfloat162float(__float2bfloat16_rn(opm_read(bo,i,types&32)));
  if (i < 131072) {
    __nv_bfloat16 v = __float2bfloat16_rn(opm_read(wo,i,types&16));
    o[i] = v; ot[(i%1024)*128+i/1024] = v;
  }
}
__device__ float opm_warp_sum(float x) {
  for (int d=16;d;d>>=1) x += __shfl_xor_sync(0xffffffff,x,d);
  return x;
}
__global__ void opm_prologue_native_kernel(const __nv_bfloat16* m, const bool* mask,
    const float* gamma, const float* beta, const __nv_bfloat16* wa,
    const __nv_bfloat16* wb, __nv_bfloat16* a, __nv_bfloat16* b,
    __nv_bfloat16* mask16, float* stats, int S, int N, float eps) {
  int lane=threadIdx.x%32;
  int r=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if (r>=S*N) return;
  float x0=__bfloat162float(m[(long)r*64+lane]);
  float x1=__bfloat162float(m[(long)r*64+lane+32]);
  float mean=opm_warp_sum(x0+x1)*(1.f/64);
  x0-=mean; x1-=mean;
  float var=opm_warp_sum(x0*x0+x1*x1)*(1.f/64);
  float rs=rsqrtf(var+eps);
  float y0=__bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(x0*rs,gamma[lane]),beta[lane])));
  float y1=__bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(x1*rs,gamma[lane+32]),beta[lane+32])));
  float va=0, vb=0;
  #pragma unroll
  for(int k=0;k<64;++k) {
    float y=__shfl_sync(0xffffffff,k<32?y0:y1,k%32);
    va=fmaf(y,__bfloat162float(wa[lane*64+k]),va);
    vb=fmaf(y,__bfloat162float(wb[lane*64+k]),vb);
  }
  bool keep=mask[r]; int s=r/N, n=r%N;
  a[((long)n*32+lane)*S+s]=__float2bfloat16_rn(keep?va:0.f);
  b[((long)n*32+lane)*S+s]=__float2bfloat16_rn(keep?vb:0.f);
  if(lane==0) {
    mask16[r]=__float2bfloat16_rn(keep?1.f:0.f);
    if(stats) {stats[r*2]=mean;stats[r*2+1]=rs;}
  }
}
__global__ void opm_mask_bits_native(const bool* mask, unsigned* bits, int S, int N) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  int word=t/32, lane=t%32;
  if(word>=N*(S/32)) return;
  int n=word/(S/32), s=(word%(S/32))*32+lane;
  unsigned v=__ballot_sync(0xffffffff,mask[(long)s*N+n]);
  if(lane==0)bits[word]=v;
}
__global__ void opm_mask_fp32_selective(const bool* mask, float* output, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)output[i]=mask[i]?1.f:0.f;
}
__global__ void opm_norm_clamp_selective(float* norm, int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)norm[i]=fmaxf(norm[i],1.f);
}
__global__ void opm_count_native(const unsigned* bits, float* norm, int S, int N) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p>=N*N)return;
  int i=p/N,j=p%N,v=0;
  for(int k=0;k<S/32;++k)v+=__popc(bits[i*(S/32)+k]&bits[j*(S/32)+k]);
  norm[p]=float(max(v,1));
}
// Tiled mask count: a 16 x 16 tile of (i, j) pairs per block, both rows' bit words staged in shared memory
// (row stride W + 1 words, so the 16 j-rows hit 16 distinct banks and the i-rows broadcast).  Replaces the
// thread-per-pair loop whose loads strided across the whole bits array.  Exact: popcount of the AND.
#define OPM_COUNT_TILE 16
__global__ void __launch_bounds__(256) opm_count_tiled(const unsigned* __restrict__ bits, float* __restrict__ norm, int W, int N) {
  extern __shared__ unsigned count_sm[];
  const int ld=W+1, i0=blockIdx.y*OPM_COUNT_TILE, j0=blockIdx.x*OPM_COUNT_TILE;
  unsigned* bi=count_sm; unsigned* bj=count_sm+OPM_COUNT_TILE*ld;
  const int t=threadIdx.y*OPM_COUNT_TILE+threadIdx.x;
  for(int e=t;e<OPM_COUNT_TILE*W;e+=256) {
    int r=e/W,k=e%W;
    bi[r*ld+k]=bits[(long)(i0+r)*W+k];
    bj[r*ld+k]=bits[(long)(j0+r)*W+k];
  }
  __syncthreads();
  const unsigned* pi=bi+threadIdx.y*ld; const unsigned* pj=bj+threadIdx.x*ld;
  int v=0;
  #pragma unroll 8
  for(int k=0;k<W;++k) v+=__popc(pi[k]&pj[k]);
  norm[(long)(i0+threadIdx.y)*N+j0+threadIdx.x]=float(max(v,1));
}
static void opm_count_launch(const unsigned* bits, float* norm, int S, int N) {
  const int W=S/32;
  TORCH_CHECK(N%OPM_COUNT_TILE==0 && S%32==0,"OPM count tile contract");
  cudaKernel_t kernel; C10_CUDA_CHECK(cudaGetKernel(&kernel, opm_count_tiled));
  const unsigned* bp=bits; float* np=norm; int w=W, n=N;
  void* argv[]={&bp,&np,&w,&n};
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(N/OPM_COUNT_TILE,N/OPM_COUNT_TILE),
      dim3(OPM_COUNT_TILE,OPM_COUNT_TILE),argv,size_t(2*OPM_COUNT_TILE*(W+1))*sizeof(unsigned),at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void opm_finish_native(const float* red,const float* gamma,const float* beta,
    const __nv_bfloat16* wa,const __nv_bfloat16* wb,float* da,float* db,float* dg,float* de) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<2048) {
    int c=i/64,k=i%64;
    da[i]=__fadd_rn(__fmul_rn(red[i],gamma[k]),red[4096+c]*beta[k]);
    db[i]=__fadd_rn(__fmul_rn(red[2048+i],gamma[k]),red[4128+c]*beta[k]);
  }
  if(i<64) {
    float g=0,b=0;
    for(int c=0;c<64;++c) {
      float w=__bfloat162float((c<32?wa:wb)[(c%32)*64+i]);
      g=__fadd_rn(g,__fmul_rn(w,red[c*64+i]));
      b=__fadd_rn(b,__fmul_rn(w,red[4096+c]));
    }
    // Raw projection weights also cover gamma==0.  Dividing a gamma-folded
    // matrix by gamma loses that derivative and produced NaNs in the old path.
    dg[i]=g;de[i]=b;
  }
}
__global__ void opm_cast_native_kernel(const float* src,__nv_bfloat16* dst,long count) {
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)dst[i]=__float2bfloat16_rn(src[i]);
}
struct OpmGradCast { const float* src[6]; void* dst[6]; int sizes[6],fp32[6]; };
__global__ void opm_grad_cast_native(OpmGradCast p) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  #pragma unroll
  for(int k=0;k<6;++k)if(i<p.sizes[k]) {
    float v=p.src[k][i];
    if(p.fp32[k])static_cast<float*>(p.dst[k])[i]=v;
    else static_cast<__nv_bfloat16*>(p.dst[k])[i]=__float2bfloat16_rn(v);
  }
}

// ---------------------------------------------------------------------------------------------
// Hopper warpgroup mma.  Both operands are addressed in shared memory through descriptors, so there
// is no ldmatrix and no LDS for the weight: one instruction per 16-deep k step covers a 64x128 tile
// that four mma.sync warps would have fed with 3 KiB of shared reads each.  That shared traffic is
// what binds this kernel -- 68% of L1 peak against 47% of DRAM.
//
// The tile layout is the 128-byte-swizzle canonical form: rows of 64 bf16 (128 B), with the 16-byte
// chunk index XOR-ed by the row.  It is the one form that is both a plain row-major destination for
// cp.async and something wgmma can address; a padded stride cannot be both (see KCP above).
namespace wg {
constexpr int ROWS = 128;                       // rows per swizzled block, both operands
constexpr uint32_t SBO = 64;                    // uint128 between one group of 8 rows and the next

// element offset of an 8-value chunk.  kc in [0,8) selects the chunk inside a 64-element row block;
// `rows` is how many rows one block of the tile holds, so blocks stack without a gap.
__device__ __forceinline__ int off_n(int rows, int blk, int mn, int kc) {
  return ((blk * rows + mn) * 8 + (kc ^ (mn & 7))) * 8;
}
__device__ __forceinline__ int off(int blk, int mn, int kc) { return off_n(ROWS, blk, mn, kc); }

// MN-major variant: the operand is stored [k][mn] with mn contiguous, which is what a tile staged
// straight from a row-major source already is.  Same 128-byte swizzle, but the chunk index that gets
// XOR-ed is the mn chunk and the row is k.
__device__ __forceinline__ int off_mn(int blk, int k, int chunk, int K) {
  return ((blk * K + k) * 8 + (chunk ^ (k & 7))) * 8;
}

__device__ __forceinline__ uint64_t desc_mn(const void* p, uint32_t lbo_u128) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(lbo_u128 & 0x3FFF) << 16);  // to the next 64-wide mn block
  d |= (static_cast<uint64_t>(SBO) << 32);
  d |= (static_cast<uint64_t>(1) << 62);                  // SWIZZLE_128B
  return d;
}

__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);                  // leading offset is 1 for a swizzled tile
  d |= (static_cast<uint64_t>(SBO) << 32);
  d |= (static_cast<uint64_t>(1) << 62);                  // SWIZZLE_128B
  return d;
}

__device__ __forceinline__ void mma_m64n128k16(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %64, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31,\n"
      " %32, %33, %34, %35, %36, %37, %38, %39,\n"
      " %40, %41, %42, %43, %44, %45, %46, %47,\n"
      " %48, %49, %50, %51, %52, %53, %54, %55,\n"
      " %56, %57, %58, %59, %60, %61, %62, %63},\n"
      " %65, %66, p, 1, 1, 0, 0;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31]),
        "+f"(d[32]),
        "+f"(d[33]),
        "+f"(d[34]),
        "+f"(d[35]),
        "+f"(d[36]),
        "+f"(d[37]),
        "+f"(d[38]),
        "+f"(d[39]),
        "+f"(d[40]),
        "+f"(d[41]),
        "+f"(d[42]),
        "+f"(d[43]),
        "+f"(d[44]),
        "+f"(d[45]),
        "+f"(d[46]),
        "+f"(d[47]),
        "+f"(d[48]),
        "+f"(d[49]),
        "+f"(d[50]),
        "+f"(d[51]),
        "+f"(d[52]),
        "+f"(d[53]),
        "+f"(d[54]),
        "+f"(d[55]),
        "+f"(d[56]),
        "+f"(d[57]),
        "+f"(d[58]),
        "+f"(d[59]),
        "+f"(d[60]),
        "+f"(d[61]),
        "+f"(d[62]),
        "+f"(d[63])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}
__device__ __forceinline__ void mma_m64n128k16_mn(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %64, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31,\n"
      " %32, %33, %34, %35, %36, %37, %38, %39,\n"
      " %40, %41, %42, %43, %44, %45, %46, %47,\n"
      " %48, %49, %50, %51, %52, %53, %54, %55,\n"
      " %56, %57, %58, %59, %60, %61, %62, %63},\n"
      " %65, %66, p, 1, 1, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31]),
        "+f"(d[32]),
        "+f"(d[33]),
        "+f"(d[34]),
        "+f"(d[35]),
        "+f"(d[36]),
        "+f"(d[37]),
        "+f"(d[38]),
        "+f"(d[39]),
        "+f"(d[40]),
        "+f"(d[41]),
        "+f"(d[42]),
        "+f"(d[43]),
        "+f"(d[44]),
        "+f"(d[45]),
        "+f"(d[46]),
        "+f"(d[47]),
        "+f"(d[48]),
        "+f"(d[49]),
        "+f"(d[50]),
        "+f"(d[51]),
        "+f"(d[52]),
        "+f"(d[53]),
        "+f"(d[54]),
        "+f"(d[55]),
        "+f"(d[56]),
        "+f"(d[57]),
        "+f"(d[58]),
        "+f"(d[59]),
        "+f"(d[60]),
        "+f"(d[61]),
        "+f"(d[62]),
        "+f"(d[63])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}
// N = 64 variant.
__device__ __forceinline__ void mma_m64n64k16(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %32, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31},\n"
      " %33, %34, p, 1, 1, 0, 0;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}

// N = 32 variant.  Sixteen accumulator registers, which is what lets the
// prologue run its projection in two halves without growing its register footprint.
__device__ __forceinline__ void mma_m64n32k16(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %16, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15},\n"
      " %17, %18, p, 1, 1, 0, 0;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}

// N = 64, MN-major operands variant.  Both operands are [k][mn] tiles -- the shape a
// row-major staging loop already produces, so neither the rows nor the weights need a transpose.
__device__ __forceinline__ void mma_m64n64k16_mn(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %32, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31},\n"
      " %33, %34, p, 1, 1, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}

// N = 72, MN-major operands variant.  One n tile past 64: the prologue hangs a
// mask column on its dW GEMM so the masked column sums come out of the accumulator.
__device__ __forceinline__ void mma_m64n72k16_mn(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %36, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31,\n"
      " %32, %33, %34, %35},\n"
      " %37, %38, p, 1, 1, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31]),
        "+f"(d[32]),
        "+f"(d[33]),
        "+f"(d[34]),
        "+f"(d[35])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}


// N = 136, MN-major operands variant: the d_msa = 128 prologue backward (128 channels + the mask column's n tile).
__device__ __forceinline__ void mma_m64n136k16_mn(uint64_t da, uint64_t db, float* d, int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %68, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n136k16.f32.bf16.bf16 {\n"
      " %0, %1, %2, %3, %4, %5, %6, %7,\n"
      " %8, %9, %10, %11, %12, %13, %14, %15,\n"
      " %16, %17, %18, %19, %20, %21, %22, %23,\n"
      " %24, %25, %26, %27, %28, %29, %30, %31,\n"
      " %32, %33, %34, %35, %36, %37, %38, %39,\n"
      " %40, %41, %42, %43, %44, %45, %46, %47,\n"
      " %48, %49, %50, %51, %52, %53, %54, %55,\n"
      " %56, %57, %58, %59, %60, %61, %62, %63,\n"
      " %64, %65, %66, %67},\n"
      " %69, %70, p, 1, 1, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15]),
        "+f"(d[16]),
        "+f"(d[17]),
        "+f"(d[18]),
        "+f"(d[19]),
        "+f"(d[20]),
        "+f"(d[21]),
        "+f"(d[22]),
        "+f"(d[23]),
        "+f"(d[24]),
        "+f"(d[25]),
        "+f"(d[26]),
        "+f"(d[27]),
        "+f"(d[28]),
        "+f"(d[29]),
        "+f"(d[30]),
        "+f"(d[31]),
        "+f"(d[32]),
        "+f"(d[33]),
        "+f"(d[34]),
        "+f"(d[35]),
        "+f"(d[36]),
        "+f"(d[37]),
        "+f"(d[38]),
        "+f"(d[39]),
        "+f"(d[40]),
        "+f"(d[41]),
        "+f"(d[42]),
        "+f"(d[43]),
        "+f"(d[44]),
        "+f"(d[45]),
        "+f"(d[46]),
        "+f"(d[47]),
        "+f"(d[48]),
        "+f"(d[49]),
        "+f"(d[50]),
        "+f"(d[51]),
        "+f"(d[52]),
        "+f"(d[53]),
        "+f"(d[54]),
        "+f"(d[55]),
        "+f"(d[56]),
        "+f"(d[57]),
        "+f"(d[58]),
        "+f"(d[59]),
        "+f"(d[60]),
        "+f"(d[61]),
        "+f"(d[62]),
        "+f"(d[63]),
        "+f"(d[64]),
        "+f"(d[65]),
        "+f"(d[66]),
        "+f"(d[67])
      : "r"(accumulate), "l"(da), "l"(db)
      : "memory");
}

__device__ __forceinline__ void fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
// wgmma reads shared memory through the async proxy.  A tile filled by cp.async is already visible
// there; one filled by ordinary shared stores is NOT, and the multiply silently reads stale memory
// (it cost a day's worth of NaN in the prologue).  All three kernels here stage with cp.async, so
// none of them needs this -- but any kernel that does not, does.
__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wait()   { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
}  // namespace wg

// One 64-MSA-row tile per token. LN/affine rounding, the two projections,
// masking, and grouped output layout remain one fusion boundary. WGMMA reads
// the normalized rows and stacked weights from shared memory; the output is
// transposed there so each global store spans contiguous MSA rows.
__global__ __launch_bounds__(128) void opm_prologue_wgmma_native(
    const __nv_bfloat16* m,const bool* mask,const float* gamma,const float* beta,
    const __nv_bfloat16* wa,const __nv_bfloat16* wb,__nv_bfloat16* a,
    __nv_bfloat16* b,__nv_bfloat16* mask16,float* stats,int S,int N,float eps) {
  __shared__ __align__(1024) __nv_bfloat16 sx[64*64],sw[64*64],so[64*64];
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,n=blockIdx.x,s0=blockIdx.y*64;
  for(int i=tid;i<4096;i+=128) {
    int c=i/64,k=i%64;
    sw[wg::off_n(64,0,c,k/8)+k%8]=(c<32?wa:wb)[(c%32)*64+k];
  }
  for(int rr=0;rr<16;++rr) {
    int r=warp*16+rr;long row=(long)(s0+r)*N+n;
    float x0=__bfloat162float(m[row*64+lane]);
    float x1=__bfloat162float(m[row*64+lane+32]);
    float mean=opm_warp_sum(x0+x1)*(1.f/64);
    x0-=mean;x1-=mean;
    float rs=rsqrtf(opm_warp_sum(x0*x0+x1*x1)*(1.f/64)+eps);
    sx[wg::off_n(64,0,r,lane/8)+lane%8]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(x0*rs,gamma[lane]),beta[lane]));
    sx[wg::off_n(64,0,r,(lane+32)/8)+lane%8]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(x1*rs,gamma[lane+32]),beta[lane+32]));
    if(lane==0) {
      mask16[row]=__float2bfloat16_rn(mask[row]?1.f:0.f);
      if(stats){stats[row*2]=mean;stats[row*2+1]=rs;}
    }
  }
  wg::proxy_fence();__syncthreads();wg::fence();
  float acc[32]={};
  #pragma unroll
  for(int ks=0;ks<4;++ks)
    wg::mma_m64n64k16(wg::desc(sx+wg::off_n(64,0,0,ks*2)),
                       wg::desc(sw+wg::off_n(64,0,0,ks*2)),acc,ks!=0);
  wg::commit();wg::wait();
  #pragma unroll
  for(int t=0;t<8;++t) {
    int c=t*8+(lane%4)*2,r=warp*16+lane/4;
    *reinterpret_cast<__nv_bfloat162*>(so+r*64+c)=__floats2bfloat162_rn(acc[t*4],acc[t*4+1]);
    *reinterpret_cast<__nv_bfloat162*>(so+(r+8)*64+c)=__floats2bfloat162_rn(acc[t*4+2],acc[t*4+3]);
  }
  __syncthreads();
  for(int t=tid;t<4096;t+=128) {
    int c=t/64,r=t%64;
    (c<32?a:b)[((long)n*32+c%32)*S+s0+r]=
        mask[(long)(s0+r)*N+n]?so[r*64+c]:__float2bfloat16_rn(0.f);
  }
}

// ----------------------------------------------------------------------------------------------
namespace opm_pro_tma {
constexpr int BS=128, NST=3, NT=512, XT=8192, TT=8192, MT=2048;
constexpr int SMEM=(NST*XT+2*TT+4096)*2+NST*MT+512+128+1024;
__device__ __forceinline__ uint32_t sa(const void* p){return uint32_t(__cvta_generic_to_shared(p));}
__device__ __forceinline__ void init(uint64_t* p,int n){asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(sa(p)),"r"(n):"memory");}
__device__ __forceinline__ void expect(uint64_t* p,int n){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(sa(p)),"r"(n):"memory");}
__device__ __forceinline__ void arrive(uint64_t* p){asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sa(p)):"memory");}
__device__ __forceinline__ void wait(uint64_t* p,int phase){asm volatile("{.reg .pred p; W_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W_%=;}"::"r"(sa(p)),"r"(phase):"memory");}
__device__ __forceinline__ void ld2(const void* map,void* dst,uint64_t* b,int x,int y){asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"::"r"(sa(dst)),"l"(map),"r"(sa(b)),"r"(x),"r"(y):"memory");}
__device__ __forceinline__ void ld3(const void* map,void* dst,uint64_t* b,int x,int y,int z){asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5}],[%2];"::"r"(sa(dst)),"l"(map),"r"(sa(b)),"r"(x),"r"(y),"r"(z):"memory");}
__device__ __forceinline__ void st2(const void* map,void* src,int x,int y){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(sa(src)),"r"(x),"r"(y):"memory");}
__device__ __forceinline__ void sync(){asm volatile("bar.sync 1,128;":::"memory");}
template<int N> __device__ __forceinline__ void wait_read(){asm volatile("cp.async.bulk.wait_group.read %0;"::"n"(N):"memory");}
}
__global__ __launch_bounds__(512,1) void opm_prologue_tma_hopper(
 const __grid_constant__ CUtensorMap xm,const __grid_constant__ CUtensorMap mm,
 const __grid_constant__ CUtensorMap wam,const __grid_constant__ CUtensorMap wbm,
 const __grid_constant__ CUtensorMap am,const __grid_constant__ CUtensorMap bm,
 const float* gamma,const float* beta,__nv_bfloat16* mask16,float* stats,unsigned* bits,
 int S,int N,int tiles,float eps){
 using namespace opm_pro_tma;
 extern __shared__ __align__(1024) unsigned char raw[];
 __nv_bfloat16* sx=reinterpret_cast<__nv_bfloat16*>(raw);
 __nv_bfloat16* st=sx+NST*XT;__nv_bfloat16* sw=st+2*TT;
 unsigned char* sm=reinterpret_cast<unsigned char*>(sw+4096);
 float* sl=reinterpret_cast<float*>(sm+NST*MT);
 uint64_t* bars=reinterpret_cast<uint64_t*>(sl+128);
 uint64_t* xf=bars;uint64_t* yf=xf+NST;uint64_t* xe=yf+NST;uint64_t* wf=xe+NST;
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,nsb=S/BS;
 if(tid==0){
  for(int k=0;k<NST;++k){init(xf+k,1);init(yf+k,8);init(xe+k,1);}
  init(wf,1);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 if(tid<64){sl[tid]=gamma[tid];sl[64+tid]=beta[tid];}
 __syncthreads();
 if(warp<4){
  if(tid==0){
   expect(wf,8192);ld2(&wam,sw,wf,0,0);ld2(&wbm,sw+2048,wf,0,0);
   int g=0;
   for(int t=blockIdx.x;t<tiles;t+=gridDim.x,++g){
    int slot=g%NST,i=t/nsb,s0=(t%nsb)*BS;
    if(g>=NST)wait(xe+slot,(g/NST-1)&1);
    expect(xf+slot,XT*2+MT);
    ld3(&xm,sx+slot*XT,xf+slot,0,i,s0);
    ld2(&mm,sm+slot*MT,xf+slot,i&~15,s0);
   }
  }
 }else if(warp<12){
  // The first prototype's inc-to-80 trapped when the compiler's initial
  // allotment was already larger. Keep the compiler's static allocation.
  const int r=(warp-4)*16+(lane&15),hf=lane>>4;int g=0;
  for(int t=blockIdx.x;t<tiles;t+=gridDim.x,++g){
   int slot=g%NST,i=t/nsb,s0=(t%nsb)*BS;
   wait(xf+slot,(g/NST)&1);
   __nv_bfloat16* row=sx+slot*XT+r*64;
   float vals[32];float sum0=0.f,sum1=0.f;
   #pragma unroll
   for(int k=0;k<4;++k){
    uint4 v=*reinterpret_cast<uint4*>(row+(((hf*4+k)^(r&7))*8));
    uint32_t q[4]={v.x,v.y,v.z,v.w};
    #pragma unroll
    for(int j=0;j<4;++j){
     float2 f=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q[j]));
     vals[k*8+j*2]=f.x;vals[k*8+j*2+1]=f.y;
     sum0+=f.x;sum1+=f.y;
    }
   }
   float sum=sum0+sum1;sum+=__shfl_xor_sync(0xffffffff,sum,16);
   float mean=sum*(1.f/64),v0=0.f,v1=0.f;
   #pragma unroll
   for(int k=0;k<16;++k){vals[k*2]-=mean;vals[k*2+1]-=mean;v0=fmaf(vals[k*2],vals[k*2],v0);v1=fmaf(vals[k*2+1],vals[k*2+1],v1);}
   float var=v0+v1;var+=__shfl_xor_sync(0xffffffff,var,16);
   float rs=1.f/sqrtf(var*(1.f/64)+eps);
   bool keep=sm[slot*MT+r*16+(i&15)]!=0;uint32_t keep32=keep?0xffffffffu:0u;
   #pragma unroll
   for(int k=0;k<4;++k){
    uint32_t q[4];
    #pragma unroll
    for(int j=0;j<4;++j){
     int c=hf*32+k*8+j*2;
     float y0=__fadd_rn(__fmul_rn(vals[k*8+j*2]*rs,sl[c]),sl[64+c]);
     float y1=__fadd_rn(__fmul_rn(vals[k*8+j*2+1]*rs,sl[c+1]),sl[65+c]);
     __nv_bfloat162 z=__floats2bfloat162_rn(y0,y1);q[j]=*reinterpret_cast<uint32_t*>(&z)&keep32;
    }
    *reinterpret_cast<uint4*>(row+(((hf*4+k)^(r&7))*8))=make_uint4(q[0],q[1],q[2],q[3]);
   }
   if(hf==0){
    long ix=(long)(s0+r)*N+i;mask16[ix]=__float2bfloat16_rn(keep?1.f:0.f);
    if(stats){stats[ix*2]=mean;stats[ix*2+1]=rs;}
   }
   unsigned ballot=__ballot_sync(0xffffffff,keep);
   if(lane==0)reinterpret_cast<uint16_t*>(bits)[((long)i*S+s0+r)/16]=uint16_t(ballot);
   wg::proxy_fence();__syncwarp();if(lane==0)arrive(yf+slot);
  }
 }else{
  wait(wf,0);const int cw=warp-12;int g=0;
  for(int t=blockIdx.x;t<tiles;t+=gridDim.x,++g){
   int slot=g%NST,b=g%2,i=t/nsb,s0=(t%nsb)*BS;
   wait(yf+slot,(g/NST)&1);
   float acc[64]={};wg::fence();
   #pragma unroll
   for(int ks=0;ks<4;++ks)wg::mma_m64n128k16(wg::desc(sw+ks*16),wg::desc(sx+slot*XT+ks*16),acc,ks!=0);
   wg::commit();wg::wait();sync();
   if(lane==0&&cw==0)arrive(xe+slot);
   if(lane==0&&cw==0&&g>=2)wait_read<1>();sync();
   #pragma unroll
   for(int k=0;k<16;++k){
    int col=k*8+(lane%4)*2,ch=cw*16+lane/4,h=col/64,cc=col%64;
    __nv_bfloat16* out=st+b*TT+h*4096;
    *reinterpret_cast<__nv_bfloat162*>(out+wg::off_n(64,0,ch,cc/8)+cc%8)=__floats2bfloat162_rn(acc[k*4],acc[k*4+1]);
    *reinterpret_cast<__nv_bfloat162*>(out+wg::off_n(64,0,ch+8,cc/8)+cc%8)=__floats2bfloat162_rn(acc[k*4+2],acc[k*4+3]);
   }
   wg::proxy_fence();sync();
   if(lane==0&&cw==0){
    for(int h=0;h<2;++h){st2(&am,st+b*TT+h*4096,s0+h*64,i*32);st2(&bm,st+b*TT+h*4096+2048,s0+h*64,i*32);}
    asm volatile("cp.async.bulk.commit_group;":::"memory");
   }
  }
  if(lane==0&&cw==0)wait_read<0>();sync();
 }
 __syncthreads();
}

// ----------------------------------------------------------------------------------------------
// The persistent TMA prologue with the row width CMT (d_msa 64 or 128: the wide OPM shapes of Protenix-v2 / ESMFold2 /
// OpenDDE have d_msa 128).  The pipeline is the d_msa = 64 kernel above: one warp issues the TMA loads, eight warps
// LayerNorm the 128 MSA rows of a tile in place (a lane pair per row), four warps (one warpgroup) run the stacked
// [Wa; Wb] projection on wgmma and store the grouped layouts by TMA.  Width CMT only changes the x tile (CMT / 64
// swizzled [128 s][64] blocks), the LayerNorm lane work (CMT / 2 values per lane) and the projection depth (CMT / 16
// wgmma steps over CMT / 64 weight blocks).
namespace opm_pro_wide {
template <int CMT>
struct Cfg {
  static constexpr int KB = CMT / 64, BS = 128, NST = 3, XT = BS * CMT, TT = 8192, MT = 2048, WT = 64 * CMT;
  static constexpr int SMEM = (NST * XT + 2 * TT + WT) * 2 + NST * MT + 2 * CMT * 4 + 128 + 1024;
};
}
template <int CMT>
__global__ __launch_bounds__(512, 1) void opm_prologue_tma_wide(
    const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap mm,
    const __grid_constant__ CUtensorMap wam, const __grid_constant__ CUtensorMap wbm,
    const __grid_constant__ CUtensorMap am, const __grid_constant__ CUtensorMap bm,
    const float* gamma, const float* beta, __nv_bfloat16* mask16, float* stats, unsigned* bits,
    int S, int N, int tiles, float eps) {
  using namespace opm_pro_tma;
  using C = opm_pro_wide::Cfg<CMT>;
  constexpr int KB = C::KB, NST = C::NST, XT = C::XT, TT = C::TT, MT = C::MT;
  extern __shared__ __align__(1024) unsigned char raw_[];
  unsigned char* raw = raw_ + ((1024u - (sa(raw_) & 1023u)) & 1023u);
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(raw);
  __nv_bfloat16* st = sx + NST * XT;
  __nv_bfloat16* sw = st + 2 * TT;
  unsigned char* sm = reinterpret_cast<unsigned char*>(sw + C::WT);
  float* sl = reinterpret_cast<float*>(sm + NST * MT);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sl + 2 * CMT);
  uint64_t* xf = bars; uint64_t* yf = xf + NST; uint64_t* xe = yf + NST; uint64_t* wf = xe + NST;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, nsb = S / C::BS;
  if (tid == 0) {
    for (int k = 0; k < NST; ++k) { init(xf + k, 1); init(yf + k, 8); init(xe + k, 1); }
    init(wf, 1); asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  for (int c = tid; c < CMT; c += 512) { sl[c] = gamma[c]; sl[CMT + c] = beta[c]; }
  __syncthreads();
  if (warp < 4) {
    if (tid == 0) {
      expect(wf, KB * 8192);
      for (int kb = 0; kb < KB; ++kb) {
        ld2(&wam, sw + kb * 4096, wf, kb * 64, 0);
        ld2(&wbm, sw + kb * 4096 + 2048, wf, kb * 64, 0);
      }
      int g = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++g) {
        int slot = g % NST, i = t / nsb, s0 = (t % nsb) * C::BS;
        if (g >= NST) wait(xe + slot, (g / NST - 1) & 1);
        expect(xf + slot, XT * 2 + MT);
        for (int kb = 0; kb < KB; ++kb) ld3(&xm, sx + slot * XT + kb * 8192, xf + slot, kb * 64, i, s0);
        ld2(&mm, sm + slot * MT, xf + slot, i & ~15, s0);
      }
    }
  } else if (warp < 12) {
    const int r = (warp - 4) * 16 + (lane & 15), hf = lane >> 4;
    int g = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++g) {
      int slot = g % NST, i = t / nsb, s0 = (t % nsb) * C::BS;
      wait(xf + slot, (g / NST) & 1);
      __nv_bfloat16* row = sx + slot * XT + r * 64;                        // + kb * 8192: the kb-th 64-column block
      float vals[CMT / 2];
      float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
      for (int kb = 0; kb < KB; ++kb) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          uint4 v = *reinterpret_cast<uint4*>(row + kb * 8192 + (((hf * 4 + k) ^ (r & 7)) * 8));
          uint32_t q[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q[j]));
            vals[(kb * 4 + k) * 8 + j * 2] = f.x; vals[(kb * 4 + k) * 8 + j * 2 + 1] = f.y;
            sum0 += f.x; sum1 += f.y;
          }
        }
      }
      float sum = sum0 + sum1; sum += __shfl_xor_sync(0xffffffff, sum, 16);
      float mean = sum * (1.f / CMT), v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int k = 0; k < CMT / 4; ++k) {
        vals[k * 2] -= mean; vals[k * 2 + 1] -= mean;
        v0 = fmaf(vals[k * 2], vals[k * 2], v0); v1 = fmaf(vals[k * 2 + 1], vals[k * 2 + 1], v1);
      }
      float var = v0 + v1; var += __shfl_xor_sync(0xffffffff, var, 16);
      float rs = 1.f / sqrtf(var * (1.f / CMT) + eps);
      bool keep = sm[slot * MT + r * 16 + (i & 15)] != 0; uint32_t keep32 = keep ? 0xffffffffu : 0u;
#pragma unroll
      for (int kb = 0; kb < KB; ++kb) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          uint32_t q[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            int c = kb * 64 + hf * 32 + k * 8 + j * 2;
            float y0 = __fadd_rn(__fmul_rn(vals[(kb * 4 + k) * 8 + j * 2] * rs, sl[c]), sl[CMT + c]);
            float y1 = __fadd_rn(__fmul_rn(vals[(kb * 4 + k) * 8 + j * 2 + 1] * rs, sl[c + 1]), sl[CMT + c + 1]);
            __nv_bfloat162 z = __floats2bfloat162_rn(y0, y1); q[j] = *reinterpret_cast<uint32_t*>(&z) & keep32;
          }
          *reinterpret_cast<uint4*>(row + kb * 8192 + (((hf * 4 + k) ^ (r & 7)) * 8)) = make_uint4(q[0], q[1], q[2], q[3]);
        }
      }
      if (hf == 0) {
        long ix = (long)(s0 + r) * N + i; mask16[ix] = __float2bfloat16_rn(keep ? 1.f : 0.f);
        if (stats) { stats[ix * 2] = mean; stats[ix * 2 + 1] = rs; }
      }
      unsigned ballot = __ballot_sync(0xffffffff, keep);
      if (lane == 0) reinterpret_cast<uint16_t*>(bits)[((long)i * S + s0 + r) / 16] = uint16_t(ballot);
      wg::proxy_fence(); __syncwarp(); if (lane == 0) arrive(yf + slot);
    }
  } else {
    wait(wf, 0); const int cw = warp - 12; int g = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++g) {
      int slot = g % NST, b = g % 2, i = t / nsb, s0 = (t % nsb) * C::BS;
      wait(yf + slot, (g / NST) & 1);
      float acc[64] = {}; wg::fence();
#pragma unroll
      for (int ks = 0; ks < CMT / 16; ++ks)
        wg::mma_m64n128k16(wg::desc(sw + (ks / 4) * 4096 + (ks % 4) * 16), wg::desc(sx + slot * XT + (ks / 4) * 8192 + (ks % 4) * 16), acc, ks != 0);
      wg::commit(); wg::wait(); sync();
      if (lane == 0 && cw == 0) arrive(xe + slot);
      if (lane == 0 && cw == 0 && g >= 2) wait_read<1>();
      sync();
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        int col = k * 8 + (lane % 4) * 2, ch = cw * 16 + lane / 4, h = col / 64, cc = col % 64;
        __nv_bfloat16* out = st + b * TT + h * 4096;
        *reinterpret_cast<__nv_bfloat162*>(out + wg::off_n(64, 0, ch, cc / 8) + cc % 8) = __floats2bfloat162_rn(acc[k * 4], acc[k * 4 + 1]);
        *reinterpret_cast<__nv_bfloat162*>(out + wg::off_n(64, 0, ch + 8, cc / 8) + cc % 8) = __floats2bfloat162_rn(acc[k * 4 + 2], acc[k * 4 + 3]);
      }
      wg::proxy_fence(); sync();
      if (lane == 0 && cw == 0) {
        for (int h = 0; h < 2; ++h) {
          st2(&am, st + b * TT + h * 4096, s0 + h * 64, i * 32);
          st2(&bm, st + b * TT + h * 4096 + 2048, s0 + h * 64, i * 32);
        }
        asm volatile("cp.async.bulk.commit_group;":::"memory");
      }
    }
    if (lane == 0 && cw == 0) wait_read<0>();
    sync();
  }
  __syncthreads();
}

// TMA (tensor memory accelerator).  A `cp.async.bulk.tensor` store hands the hardware a descriptor
// of the GLOBAL tensor plus one coordinate, and it walks the box itself: no per-thread addresses,
// no shared loads through the LSU, and the store is asynchronous, so it overlaps the next tile's
// math instead of sitting in front of it.  What makes it worth the descriptor plumbing here is that
// the descriptor can describe a 4-D box, and the grouped layout dO[(i,c),(j,e)] IS a 4-D box --
// the scatter the kernel used to do by hand is exactly one TMA store.
namespace tma {
// Same 128-byte XOR swizzle wgmma uses, which is what lets the accumulator reach shared without
// bank conflicts: rows of 64 bf16, the 16-byte chunk index XORed by the row index mod 8.
__device__ __forceinline__ int sw128(int row, int col64) {
  return row * 64 + (((col64 >> 3) ^ (row & 7)) << 3) + (col64 & 7);
}
__device__ __forceinline__ void store_4d(const void* map, uint32_t src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
               " [%0, {%2, %3, %4, %5}], [%1];\n"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
// .read is the cheap wait: it retires as soon as the descriptor has finished READING shared memory,
// which is all that is needed before refilling the tile.  Plain wait_group waits for the write to
// land in L2 and would serialise the store against the next tile's math.
__device__ __forceinline__ void wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
__device__ __forceinline__ void wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
// stmatrix is ldmatrix run backwards, and its register layout is exactly the mma accumulator's, so
// four 8x8 tiles of a converted accumulator leave in one instruction instead of sixteen 4-byte
// stores.  Same wavefronts, a quarter of the instructions -- and this kernel issues 2.4M of them.
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void stmatrix_x4(uint32_t dst, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
               :: "r"(dst), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
}  // namespace tma

#if defined(MINIWORLD_OPM_EPILOGUE_TMA_HALF)
constexpr int WARPS = 4;
#elif defined(MINIWORLD_OPM_EPILOGUE_TMA_WIDE)
constexpr int WARPS = 16;
#else
constexpr int WARPS = 8;            // two warpgroups of 64 rows each
#endif
constexpr int THREADS = WARPS * 32;
constexpr int ROWT = NP / 16;      // 4 mma row tiles
constexpr int NTILE = ZW / 8;      // 16 mma n tiles
// A warp takes 4 row tiles x 4 n tiles = 16 of the 128 tile pairs, not 2 x 4 = 8 of 64.  Shared
// bytes per mma are 512/NPW + 256/RPW, so this is 192 instead of 256, and the CTA covering 128 pairs
// instead of 64 halves how many times the weight is staged across the grid.
constexpr int RPW = 4;             // row tiles per warp
constexpr int NPW = 8;             // (unused under wgmma; kept for the accumulator width)
constexpr int KSTEP = KC / 16;     // 8 k steps per chunk
constexpr int VEC = NP * KC / 8 / THREADS;
constexpr int WSL = ZW * KC;                 // one chunk of Wo, [ZW rows][KC], swizzled
// Pipeline depth.  Three, not two, and it is free: this kernel is capped at two blocks per SM by its
// 102 registers (the m64n128 accumulator is 64 of them), so shared memory is not what limits it and a
// third stage costs nothing in occupancy.  What it buys is one __syncthreads per chunk instead of two
// -- at depth two the buffer a chunk just read is refilled by the very next chunk, so the wgmma needs
// a barrier behind it as well as in front -- and barrier was 39% of this kernel's stall samples.
constexpr int NSTAGE = 3;
constexpr int SMEM_STAGE = NSTAGE * NP * KCP * 2 + NSTAGE * WSL * 2;
// The accumulator tile is padded for the same reason as KCP: an unpadded 128-float row is exactly
// 32 banks, so every row of it started on bank 0.
constexpr int ACCP = ZW + 4;
constexpr int SMEM_BYTES = SMEM_STAGE > NP * ACCP * 4 ? SMEM_STAGE : NP * ACCP * 4;

// THREADS is a multiple of every field below, so a thread's (e, j, c) slot never changes: only the
// i tile moves between its VEC copies, and only the k chunk moves between passes.  Recomputing the
// row-pitch multiply per copy instead put IMAD at a quarter of the kernel's instructions.
// element offset of one 16-byte chunk of a staged row
__device__ __forceinline__ int sp_off(int row, int chunk) { return row * KC + ((chunk ^ (row & 7)) << 3); }

// A thread's (j, e) slot in the tile never moves; its VEC copies walk the i tokens and the c values.
// BJ is a multiple of 8, so the row's swizzle does not move with the i token either.
__device__ __forceinline__ void stage(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* dst,
                                      int jl, int chunk, long M) {
#pragma unroll
  for (int r = 0; r < VEC; ++r)          // BJ is a multiple of 8, so the row's swizzle is jl's alone
    __pipeline_memcpy_async(dst + sp_off(r * BJ + jl, chunk), src + (size_t)r * CH * M, 16);
  __pipeline_commit();
}

#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)

namespace opm_epi_tma {
__device__ __forceinline__ uint64_t desc64(const void* p) {
 const uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(p));
 return uint64_t((a>>4)&0x3fff) | (uint64_t(1)<<16) | (uint64_t(32)<<32) | (uint64_t(2)<<62);
}
__device__ __forceinline__ void ld4(const void* map,void* dst,uint64_t* b,int x,int y,int z,int w) {
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5,%6}],[%2];"
 ::"r"(opm_pro_tma::sa(dst)),"l"(map),"r"(opm_pro_tma::sa(b)),"r"(x),"r"(y),"r"(z),"r"(w):"memory");
}
}

#endif
__global__ __launch_bounds__(THREADS) void opm_epilogue_kernel(
    const __nv_bfloat16* __restrict__ O,      // [N*CH, N*CH] grouped GEMM output
    const float* __restrict__ NORM,           // [N, N]  mask counts in fp32, already clamped >= 1
    const __nv_bfloat16* __restrict__ BF,     // [K/16][CZ/8][32][4] pre-swizzled projection weight
    const float* __restrict__ BIAS,           // [CZ]  bf16-rounded bias, held in fp32
    const __nv_bfloat16* __restrict__ RESIDUAL, // optional [1, NI, NJ, CZ]
    int NI, int NJ, long M, int AFTER,  // AFTER: divide after the projection AND its bias (ESMFold2), not before it
    // NJ is a BLOCK of j columns when the caller bounds the workspace
    // z[i, j, :] as a 4-D box: (64 z) x j x (2 halves of z) x i.  j sits INSIDE the z half on
    // purpose: it is what makes the accumulator's eight rows land on eight distinct swizzle chunks,
    // so the write is conflict-free.  The strides are not monotonic, and TMA does not mind.
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)
    const __grid_constant__ CUtensorMap OUT_MAP,
    const __grid_constant__ CUtensorMap O_MAP,
    const __grid_constant__ CUtensorMap W_MAP) {
#else
    const __grid_constant__ CUtensorMap OUT_MAP) {
#endif
  extern __shared__ char smem[];
  // both wgmma's swizzle and TMA's are functions of the absolute shared address: 1 KiB alignment
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(smem + ((1024u - (sbase & 1023u)) & 1023u));
  // The weight's B fragments used to go straight from global into the mma, one LDG.64 per two mmas.
  // All 256 KiB of them sit in L2 because every CTA reads them, but an L2 hit is still ~250 cycles in
  // the middle of the dependency chain: the HMMAs held 79% of this kernel's stall samples, all of it
  // long scoreboard.  One chunk is staged per K pass instead.
  __nv_bfloat16* sW = sP + NSTAGE * NP * KCP;                        // [NSTAGE][CZ][KC], swizzled
  float* sAcc = reinterpret_cast<float*>(smem);                      // [NP][CZ] after the K loop
  __shared__ float sInv[NP];
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)
  __shared__ uint64_t epi_full[NSTAGE];
#endif
  __shared__ float sBias[ZW];                        // 128 numbers, otherwise re-read from global per element

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rg = (warp & 1) * RPW;          // this warp's first row tile
  const int ng = (warp >> 1) * NPW;         // this warp's first n tile
  const int i0 = blockIdx.y * BI;
  const int zs = blockIdx.x % NZS;                   // c_z slice: the fastest grid axis, so the slices that
  const int j0 = (blockIdx.x / NZS) * BJ;            // share an O tile run together and the second read hits L2
  const int z0 = zs * ZW;

  if (tid < NP) {
    const int il = tid / BJ, jl = tid - il * BJ;
    sInv[tid] = 1.0f / NORM[(i0 + il) * NJ + (j0 + jl)];
  }
  for (int v = tid; v < ZW; v += THREADS) sBias[v] = BIAS[z0 + v];

  float acc[64];                                       // one warpgroup's 64x128 tile
  const int wgi = tid >> 7, wl = (tid >> 5) & 3;

#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)

  if (tid==0) {
    for(int k=0;k<NSTAGE;++k)opm_pro_tma::init(epi_full+k,1);
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  auto fill = [&](int c) {
    const int slot=c%NSTAGE;
    opm_pro_tma::expect(epi_full+slot,(NP*KC+WSL)*2);
    opm_epi_tma::ld4(&O_MAP,sP+slot*NP*KC,epi_full+slot,0,j0,i0,c*2);
    opm_pro_tma::ld2(&W_MAP,sW+slot*WSL,epi_full+slot,c*KC,0);
#else
  // this thread's fixed slot in the tile, and the two streams walked forward one chunk at a time
  const int k8 = tid & 3, jl = (tid >> 2) & (BJ - 1), cl = (tid >> 7) & (KC / CH - 1);
  const int chunk0 = cl * (CH / 8) + k8;
  const __nv_bfloat16* osrc = O + (size_t)(i0 * CH + cl) * M + (size_t)(j0 + jl) * CH + k8 * 8;
  const __nv_bfloat16* wsrc = BF + (size_t)z0 * (CH * CH);
  // the weight slice rides in the same cp.async group as the tile it belongs to, so the pipeline
  // depth the O staging already had covers it too
  auto weights = [&](const __nv_bfloat16* src, __nv_bfloat16* dst) {
#pragma unroll
    for (int q = 0; q < WSL / 8 / THREADS; ++q) {
      const int v = q * THREADS + tid, kc = v & (KC / 8 - 1), n = v / (KC / 8);
      __pipeline_memcpy_async(dst + sp_off(n, kc), src + (size_t)n * (CH * CH) + kc * 8, 16);
    }
#endif
  };
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)
  if(tid==0) for(int p=0;p<NSTAGE-1;++p)fill(p);
  #pragma unroll 1
  for(int ck=0;ck<NCHUNK;++ck) {
    opm_pro_tma::wait(epi_full+ck%NSTAGE,(ck/NSTAGE)&1);
    const __nv_bfloat16* p=sP+(ck%NSTAGE)*NP*KC;
    const __nv_bfloat16* w=sW+(ck%NSTAGE)*WSL;
    wg::fence();
    #pragma unroll
    for(int ks=0;ks<KSTEP;++ks)
      wg::mma_m64n128k16(
        opm_epi_tma::desc64(p+(ks>>1)*NP*32+wgi*64*32+(ks&1)*16),
        wg::desc(w+sp_off(0,ks*2)),acc,(ck==0&&ks==0)?0:1);
    wg::commit();
    asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
    // Both groups finish reading ck-1 before its ring slot is overwritten.
#else
  auto fill = [&](int c) {
    weights(wsrc + (size_t)c * KC, sW + (c % NSTAGE) * WSL);
    stage(osrc + (size_t)c * (KC / CH) * M, sP + (c % NSTAGE) * NP * KCP, jl, chunk0, M);
  };
#pragma unroll
  for (int p = 0; p < NSTAGE - 1; ++p) fill(p);
#pragma unroll 1
  for (int ck = 0; ck < NCHUNK; ++ck) {
    // leave in flight every group but this chunk's
    const int left = NCHUNK - 1 - ck;
    __pipeline_wait_prior(left < NSTAGE - 2 ? left : NSTAGE - 2);
    // The one barrier of the chunk.  Behind it sits the previous chunk's wgmma, finished, which is
    // what makes it safe to refill that chunk's buffer on the line after.
#endif
    __syncthreads();
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)
    if(tid==0 && ck+NSTAGE-1<NCHUNK)fill(ck+NSTAGE-1);
#else
    if (ck + NSTAGE - 1 < NCHUNK) fill(ck + NSTAGE - 1);
    const __nv_bfloat16* p = sP + (ck % NSTAGE) * NP * KCP;
    const __nv_bfloat16* w = sW + (ck % NSTAGE) * WSL;
    wg::fence();
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks)
      wg::mma_m64n128k16(wg::desc(p + sp_off(wgi * 64, ks * 2)),
                         wg::desc(w + sp_off(0, ks * 2)), acc, (ck == 0 && ks == 0) ? 0 : 1);
    wg::commit();
    wg::wait();
#endif
  }
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)
  wg::wait();
#else

#endif
  __syncthreads();
  // / num_mask and + bias in fp32 -> bf16, one rounding, straight from the accumulator into the TMA
  // box over the dead staging buffers.  It used to go out as fp32 through a padded [NP][CZ] tile and
  // come back in for the conversion: 64 KiB of shared traffic per CTA and 1.35M conflicted
  // wavefronts, replaced by a 16 KiB conflict-free write and one asynchronous store instruction.
  __nv_bfloat16* sO = sP;
  static_assert(NP * ZW <= 2 * NP * KCP, "the output box must fit the staging buffers it reuses");
  {
    const int p0 = wgi * 64 + wl * 16 + (lane >> 2);
    const float inv0 = sInv[p0], inv1 = sInv[p0 + 8];
    const int il = p0 / BJ, jl = p0 - il * BJ;              // p0 and p0 + 8 share il: BJ is 32
#pragma unroll
    for (int nt = 0; nt < ZW / 8; ++nt) {
      const int n = nt * 8 + (lane & 3) * 2;
      const float2 b = *reinterpret_cast<const float2*>(&sBias[n]);
      const int row = (il * 2 + (n >> 6)) * BJ + jl;      // rows p and p + 8 are 8 lines apart
      __nv_bfloat162 v0, v1;
      if (AFTER) {
        v0 = __float22bfloat162_rn(make_float2((acc[nt * 4 + 0] + b.x) * inv0, (acc[nt * 4 + 1] + b.y) * inv0));
        v1 = __float22bfloat162_rn(make_float2((acc[nt * 4 + 2] + b.x) * inv1, (acc[nt * 4 + 3] + b.y) * inv1));
      } else {
        v0 = __float22bfloat162_rn(make_float2(fmaf(acc[nt * 4 + 0], inv0, b.x), fmaf(acc[nt * 4 + 1], inv0, b.y)));
        v1 = __float22bfloat162_rn(make_float2(fmaf(acc[nt * 4 + 2], inv1, b.x), fmaf(acc[nt * 4 + 3], inv1, b.y)));
      }
      if (RESIDUAL != nullptr) {
        // Round the update before adding: identical to bf16 OPM + bf16 residual.
        const auto* rp = RESIDUAL + ((size_t)(i0 + il) * NJ + j0 + jl) * CZ + z0 + n;
        v0 = __hadd2(v0, *reinterpret_cast<const __nv_bfloat162*>(rp));
        v1 = __hadd2(v1, *reinterpret_cast<const __nv_bfloat162*>(rp + 8 * CZ));
      }
      *reinterpret_cast<__nv_bfloat162*>(sO + tma::sw128(row, n & 63)) = v0;
      *reinterpret_cast<__nv_bfloat162*>(sO + tma::sw128(row + 8, n & 63)) = v1;
    }
  }
  wg::proxy_fence();
  __syncthreads();
  if (tid == 0) {
    tma::store_4d(&OUT_MAP, static_cast<uint32_t>(__cvta_generic_to_shared(sO)), 0, j0, zs * 2, i0);
    tma::commit();
    tma::wait_all();
  }
}

// ---------------------------------------------------------------------------------------------
// BACKWARD: the same map run the other way.  dO[(i,c),(j,e)] = (sum_z dz[i,j,z] * Wo[z,(c,e)]) / n[i,j]
// straight into the GROUPED layout, so the [N,N,c_hidden^2] permute the naive chain needs (302 MB
// written, read and written again at L=384/S=1024) never happens: this reads dz and writes dO once.
// Same tiling as the forward epilogue with reads and writes swapped, and the same pre-swizzled
// constant weight -- here Wo itself rather than its transpose, since z is now the contraction.
constexpr int KZ = CZ;             // the contraction is over c_z
constexpr int NCH = CH * CH;       // 1024 outputs per pair
// dgrad's own tile, separate from the forward epilogue's (whose warp map is pinned to 4x16).
// Every CTA applies all 1024 output columns and so re-reads the whole pre-swizzled weight: the load
// sectors are ~33x the data the kernel actually needs. Halving the CTA count (DG_BI = 8) halves that
// re-read and measured SLOWER, 0.437 -> 0.468 ms -- the weight lives in L1/L2 and the kernel is bound
// by latency, not by that traffic. Left at 4.
// Four warpgroups and 256 pairs per CTA would halve the CTA count and with it the 288 MB of Wo^T
// that gets re-read out of L2 (this kernel moves 794 MB through L2 against 356 MB through DRAM).
// It measured WORSE, 179 -> 198 us: the block doubles to 113 KiB, so only two fit, and the kernel
// wants blocks -- it is bound by the latency of the weight load, not by its bandwidth.
constexpr int DG_WARPS = 8;             // 32 pairs per warp measured worse here (224 -> 240 us):
                                        // it halves L1 traffic but leaves 8 warps per SM, and this
                                        // kernel writes 318 MB and needs the memory parallelism
constexpr int DG_THREADS = DG_WARPS * 32;
constexpr int DG_BI = 8;
constexpr int DG_BJ = 16;               // its own j block: the forward epilogue widened BJ to 32
constexpr int DG_NP = DG_BI * DG_BJ;
constexpr int DG_ROWT = DG_NP / 16;
// Output columns per pass.  64, not 128: it halves the wgmma accumulator to 32 registers and the
// staged weight to 16 KiB, which is what finally gets three blocks on an SM instead of two -- the
// kernel was capped by registers AND shared at 128.  It costs a third more shared traffic (the dz
// tile is read by sixteen passes instead of eight) and buys 24 warps per SM instead of 16.
constexpr int CHUNK = 64;
// A warp's share of the tile.  At 4 m x 2 n each mma cost 512/2 + 256/4 = 320 bytes through L1;
// at 2 m x 4 n it costs 256, for the same 32 accumulator registers.  L1 is what this kernel is
// short of (83.8% of peak against 43.5% of DRAM), so the shape matters more than the count.
constexpr int DG_MW = 4;                     // m tiles per warp
constexpr int DG_NW = (CHUNK / 8) / (DG_WARPS / (DG_ROWT / DG_MW));   // 4 n tiles per warp
static_assert((DG_ROWT / DG_MW) * ((CHUNK / 8) / DG_NW) == DG_WARPS, "the warp grid must tile the block");


// element offset of one 16-byte chunk within a [rows][WIDTH] staged tile
template <int WIDTH>
__device__ __forceinline__ int sw_off(int row, int chunk) {
  return row * WIDTH + ((chunk ^ (row & 7)) << 3);
}

// (pair, output channel) -> element offset in the swizzled TMA box [i][c][j/2][64]
__device__ __forceinline__ int dg_box(int p, int n) {
  const int jl = p & (DG_BJ - 1), il = p / DG_BJ;
  const int col = (jl & 1) * CH + (n & (CH - 1));          // 64-wide row = two j of 32 channels
  return tma::sw128((il * (CHUNK / CH) + (n >> 5)) * (DG_BJ / 2) + (jl >> 1), col);
}

__global__ __launch_bounds__(DG_THREADS) void opm_dgrad_kernel(
    const __nv_bfloat16* __restrict__ DZ,     // [NI, NJ, CZ]
    const float* __restrict__ NORM,           // [NI, NJ]
    const __nv_bfloat16* __restrict__ WOT,    // [NCH, CZ] Wo transposed: wgmma wants k contiguous
    __nv_bfloat16* __restrict__ DZP,          // [NI, NJ, CZ] dz/n, the same value staged below
    float* __restrict__ DBO,                  // [gridDim, CZ] partial sums of dz over (i, j)
    int NI, int NJ, long M, int AFTER,       // AFTER: dbo sums dz / n (the bias was divided too)
    // dO[(i,c),(j,e)] as a 4-D box: (64 columns = two whole j) x (j/2) x c x i.  The kernel never
    // computes an address into it; it hands TMA the four coordinates of the tile.
    const __grid_constant__ CUtensorMap DO_MAP) {
  extern __shared__ char smem[];
  // wgmma's swizzle is a function of the absolute shared address, so the operand tiles start on a
  // 1 KiB boundary; the launch allocates one extra KiB for the alignment.
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem + ((1024u - (sbase & 1023u)) & 1023u));
  // KZ >= 384 (one CTA per SM either way): the Wo^T chunk is double buffered, so the next chunk's weights stream in under this one's wgmma
  constexpr int WBUF = KZ >= 384 ? 2 : 1;
  constexpr int WSZ = (KZ / 64) * CHUNK * 64;
  __nv_bfloat16* sW = sA + (KZ / 64) * DG_NP * 64;                     // dz/n then Wo^T, both swizzled
  // The store-out tile, in TMA box order: [i][c][j/2][64] with the 128-byte XOR swizzle.  It used to
  // be a padded [DG_NP][CHUNK+8] tile read back by hand, and that read was 2.9M of the kernel's 18.3M
  // shared wavefronts -- 4-way conflicted, and unfixably so: a padding that makes the READ clean
  // (CHUNK+32) turns the accumulator WRITE 4-way, because the two want row strides 64 B apart mod 128
  // and 16 B apart mod 128 respectively.  The swizzle satisfies both, and TMA does the read.
  __nv_bfloat16* sO = sW + WBUF * WSZ;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int i0 = blockIdx.y * DG_BI, j0 = blockIdx.x * DG_BJ;
  // dbias_o = sum over every (i, j) of dz.  Folding it in here costs eight registers; torch's own
  // reduce over the two outer dims of a [N, N, c_z] bf16 tensor took 146 us to read 38 MB.
  // A thread's z slot never changes: the staging walks RPP rows per pass with SLOTS threads a row (for
  // KZ = 384 that is 48 slots x 5 rows, 240 of the 256 threads), so the eight running sums stay in registers.
  constexpr int SLOTS = KZ / 8, RPP = DG_THREADS / SLOTS;
  static_assert(RPP >= 1, "a dz row must fit in one pass");
  float bo[8];
#pragma unroll
  for (int t = 0; t < 8; ++t) bo[t] = 0.f;

  // stage the (i,j) tile of dz, divided by the mask count once.  A wide row (KZ > 128) takes UNR rows per round, loads first: the
  // loop was one dependent global round trip per row, the whole of the staging time at one CTA per SM.
  if (tid < RPP * SLOTS) {
    const int z8 = (tid % SLOTS) * 8;
    constexpr int UNR = KZ >= 384 ? 8 : (KZ > 128 ? 16 : 1);   // rows of dz in flight per thread (measured: 256 -> 16, 384 -> 8 on L768 S1024)
    for (int pb = tid / SLOTS; pb < DG_NP; pb += RPP * UNR) {
      uint4 raw[UNR];
      float invs[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int p = pb + u * RPP;
        if (p < DG_NP) {
          const int il = p / DG_BJ, jl = p - il * DG_BJ;
          invs[u] = 1.0f / NORM[(i0 + il) * NJ + (j0 + jl)];
          raw[u] = *reinterpret_cast<const uint4*>(DZ + (size_t)((i0 + il) * NJ + (j0 + jl)) * CZ + z8);
        }
      }
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int p = pb + u * RPP;
        if (p < DG_NP) {
          const int il = p / DG_BJ, jl = p - il * DG_BJ;
          const float inv = invs[u];
          uint4 val = raw[u];
          __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&val);
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            float2 f = __bfloat1622float2(h[t]);
            if (!AFTER) { bo[2 * t] += f.x; bo[2 * t + 1] += f.y; }      // the raw dz, before the mask count divides it
            f.x *= inv; f.y *= inv;
            if (AFTER) { bo[2 * t] += f.x; bo[2 * t + 1] += f.y; }       // ESMFold2: the bias went through the division
            h[t] = __float22bfloat162_rn(f);
          }
          *reinterpret_cast<uint4*>(sA + wg::off(z8 >> 6, p, (z8 >> 3) & 7)) = val;
          // dWo wants the same dz/n.  Recomputing it in torch cost an fp32 round trip of the whole
          // tensor -- 150 MB -- to redo a division this loop has already done.
          *reinterpret_cast<uint4*>(DZP + (size_t)((i0 + il) * NJ + (j0 + jl)) * CZ + z8) = val;
        }
      }
    }
  }
  {
    // Fold the per-thread sums through the output tile, which is not live yet.  An atomicAdd into
    // shared here looked cheaper but nvcc lowers a shared atomic with likely-colliding addresses into
    // a warp-aggregation loop: eight of them cost 24M instructions, a third of the whole kernel.
    // Every (group, z) slot below has exactly one writer, so a plain store and a tree do instead.
    float* sbo = reinterpret_cast<float*>(sO);
    constexpr int GRP = RPP;
    static_assert(GRP * CZ <= DG_NP * CHUNK / 2, "the dz sums must fit in the output tile");
    if (tid < RPP * SLOTS) {
      const int z8 = (tid % SLOTS) * 8, grp = tid / SLOTS;
      *reinterpret_cast<float4*>(sbo + grp * CZ + z8) = make_float4(bo[0], bo[1], bo[2], bo[3]);
      *reinterpret_cast<float4*>(sbo + grp * CZ + z8 + 4) = make_float4(bo[4], bo[5], bo[6], bo[7]);
    }
    __syncthreads();
    for (int z = tid; z < CZ; z += DG_THREADS) {
      float s = 0.f;
#pragma unroll
      for (int g = 0; g < GRP; ++g) s += sbo[g * CZ + z];
      DBO[(size_t)(blockIdx.y * gridDim.x + blockIdx.x) * CZ + z] = s;
    }
  }
  __syncthreads();

  const int wgi = tid >> 7, wl = (tid >> 5) & 3;        // warpgroup, and the warp inside it
  auto load_w = [&](int ck, __nv_bfloat16* dst) {
    // this ck slice of Wo^T: CHUNK rows of CZ, straight into the swizzled layout
    for (int q = tid; q < CHUNK * KZ / 8; q += DG_THREADS) {
      const int kc = q % (KZ / 8), n = q / (KZ / 8);
      __pipeline_memcpy_async(dst + wg::off_n(CHUNK, kc >> 3, n, kc & 7),
                              WOT + (size_t)(ck * CHUNK + n) * CZ + kc * 8, 16);
    }
    __pipeline_commit();
  };
  if (WBUF == 2) load_w(0, sW);
#pragma unroll 1
  for (int ck = 0; ck < NCH / CHUNK; ++ck) {
    __nv_bfloat16* swc = sW;
    float acc[32];
    if (WBUF == 2) {
      // the other buffer was last read by chunk ck - 1, whose wgmma every warp has retired behind the barrier at its end
      if (ck + 1 < NCH / CHUNK) load_w(ck + 1, sW + ((ck + 1) & 1) * WSZ); else __pipeline_commit();
      swc = sW + (ck & 1) * WSZ;
      __pipeline_wait_prior(1);
    } else {
      load_w(ck, sW);
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    wg::fence();
#pragma unroll
    for (int ks = 0; ks < KZ / 16; ++ks) {
      const int blk = ks >> 2, kk = (ks & 3) * 2;
      wg::mma_m64n64k16(wg::desc(sA + wg::off(blk, wgi * 64, kk)),
                        wg::desc(swc + wg::off_n(CHUNK, blk, 0, kk)), acc, ks == 0 ? 0 : 1);
    }
    wg::commit();
    wg::wait();
    // The wait belongs HERE, not at the top of the loop: the previous tile's TMA store has to be done
    // READING sO before this one refills it, and leaving it until after the wgmma is what lets the
    // store overlap the math.  Hoisting it up to share the staging barrier saves a __syncthreads per
    // chunk and costs the overlap: 155 -> 175 us.  The store's WRITE stays in flight either way.
    if (tid == 0) tma::wait_read();
    __syncthreads();
    // The accumulator goes straight into the swizzled box: a warp's eight rows land on eight
    // different XOR chunks, so the 32 lanes hit 32 distinct banks.  stmatrix.x4 does the same four
    // 8x8 tiles in one instruction (2.4M shared stores become 0.6M) and measured no better -- the
    // wavefront count is identical and the four addresses it needs cost what the issue slots save.
    static_assert(CHUNK / CH == 2, "the box below assumes two c values per chunk");
#pragma unroll
    for (int nt = 0; nt < CHUNK / 8; ++nt) {
      const int n = nt * 8 + (lane & 3) * 2, p = wgi * 64 + wl * 16 + (lane >> 2);
      *reinterpret_cast<__nv_bfloat162*>(sO + dg_box(p, n)) =
          __float22bfloat162_rn(make_float2(acc[nt * 4 + 0], acc[nt * 4 + 1]));
      *reinterpret_cast<__nv_bfloat162*>(sO + dg_box(p + 8, n)) =
          __float22bfloat162_rn(make_float2(acc[nt * 4 + 2], acc[nt * 4 + 3]));
    }
    // TMA reads shared through the async proxy: ordinary stores are invisible to it without this
    wg::proxy_fence();
    __syncthreads();
    if (tid == 0) {
      tma::store_4d(&DO_MAP, static_cast<uint32_t>(__cvta_generic_to_shared(sO)),
                    0, j0 * CH / 64, ck * (CHUNK / CH), i0);
      tma::commit();
    }
  }
  if (tid == 0) tma::wait_all();     // the last stores must land before the kernel retires
}
}  // namespace


// ---------------------------------------------------------------------------------------------
// BACKWARD prologue, fused.  The mirror of the forward prologue: everything between dA/dB and dm is
// one pass.  Read dA[(i,c),s], dB[(j,e),s], m; recompute y = LN(m) in registers; apply the mask;
// form dy = da.Wa + db.Wb; run the LayerNorm backward; write dm.  y, da and db leave as byproducts
// for the two weight-gradient GEMMs (cuBLAS is good at their K = S*N shape), and dgamma/dbeta come
// out as per-CTA partials -- no atomics, so the reduction is deterministic.
//
// It replaces a chain of torch ops that moved the same 50 MB tensors five or six times.
namespace {
constexpr int PB_BI = 4;                 // i tokens per CTA
constexpr int PB_BS = 32;                // MSA rows per CTA
constexpr int PB_ROWS = PB_BI * PB_BS;   // 128 (s,i) rows
constexpr int PB_THREADS = 256;          // 8 warps = two warpgroups
constexpr int WARPS_PB = PB_THREADS / 32;
// Eight threads per row in the LayerNorm phase, not four.  It halves that phase's register
// footprint (8 channels held instead of 16), which is what pays for the wgmma accumulator below,
// and it makes a warp's global read of m one whole 128-byte row instead of two interleaved halves.
// For d_msa = 128 sixteen threads share a row, so a thread still holds eight channels.
constexpr int PB_CPT = 8;                // channels per thread
constexpr int PB_PART = CM_MSA / PB_CPT; // threads per row: 8 (d_msa 64) or 16 (128)
constexpr int PB_DAB = 2 * CH;           // 64: the stacked da | db columns, the K of the projection
static_assert(CM_MSA % 64 == 0, "d_msa must be a multiple of 64 (whole 128-byte swizzle blocks)");
// sdy carries the same 128-byte XOR swizzle as the operand tiles rather than an 8-element pad.  It
// is conflict-free for all three access patterns (accumulator columns, whole rows, column pairs) and
// 2 KiB smaller, which is exactly what the double-buffered staging tile below costs.
constexpr int PB_KS = PB_DAB / 16;       // 4 wgmma k steps, over the 64 stacked da|db channels
// s blocks per CTA.  At one, the grid was 3072 CTAs and dWa/dWb left 50 MB of fp32 partials behind --
// a quarter of the kernel's traffic, and seven separate reduction launches to fold them back up.
constexpr int PB_SB = 8;
// dWa, dWb, dgamma and dbeta share one partial buffer, one row per CTA, so folding them up is one
// reduction launch instead of four -- the backward was spending more time launching reductions than
// running them.
// One partial row per CTA: the raw dW GEMM result R = [da db]^T (mask .* xh), 64 x 64, and the two
// masked column sums ssa = sum_r mask_r [da db].  dgamma and dbeta are NOT produced here any more:
// dout is mask .* ([da db] W), so sum_r dout .* xh = sum_k W[k][:] .* R[k][:] and sum_r dout =
// W^T ssa -- both fall out of R and ssa on the host, 64 x 64 elementwise, and the two column-sum
// phases that used to compute them (a barrier each, 7% of the instructions, and a product tile
// written back to shared once per row) are gone.
constexpr int PB_PW = 2 * CH * CM_MSA + 2 * CH;
constexpr int PB_RED = CM_MSA + 12;      // padded stride of the cross-warpgroup reduction buffer: 72 columns, 8 rows per bank cycle

__global__ __launch_bounds__(PB_THREADS, (CM_MSA == 64 ? 3 : 2)) void opm_prologue_bwd_kernel(   // asking for four
                                          // instead spills 48 more bytes per thread and loses:
                                          // 0.143 -> 0.165 ms   // three blocks per SM (asking
    const __nv_bfloat16* __restrict__ DA,     // [S, N*CH]: a row's channels lie next to each other
    const __nv_bfloat16* __restrict__ DB,
    const __nv_bfloat16* __restrict__ M,      // [S, N, CM]
    const float* __restrict__ STATS,          // [S, N, 2]: the forward's (mean, rstd) per row
    const __nv_bfloat16* __restrict__ MASK,   // [S, N]
    const float* __restrict__ GAMMA, const float* __restrict__ BETA, float eps,
    const __nv_bfloat16* __restrict__ BWP,    // [CM][2*CH]: (gamma .* Wa)^T | (gamma .* Wb)^T
    __nv_bfloat16* __restrict__ DM,
    float* __restrict__ DWA, float* __restrict__ DWB,                              // [blocks, CH, CM]
    int N, int S) {
  // Both GEMMs are wgmma now, so every tile it reads has to be in the 128-byte XOR swizzle and start
  // on a 1 KiB boundary; the launch allocates one extra KiB for the alignment.  What the swizzle buys
  // beyond the descriptors is that da and db can share ONE tile -- stacked as 64 columns of a 128-row
  // block -- which is what turns two m16n8k16 GEMMs into one m64 GEMM in each direction.
  extern __shared__ char pb_smem[];
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(pb_smem));
  __nv_bfloat16* sdab = reinterpret_cast<__nv_bfloat16*>(pb_smem + ((1024u - (sbase & 1023u)) & 1023u));
  // Layout: sdab (two s blocks of da|db) | sw | sdy | mk.  sdy holds the projection output dout, and the
  // LayerNorm phase turns each of its elements into xh IN PLACE (one thread owns an element: it reads dout,
  // writes xh), so `sy` is the same tile.  The dW GEMM's B operand is then [xh block 0 .. block CM/64-1 | mk],
  // evenly spaced one 16 KiB block apart -- which is what the single leading-dimension offset can express.
  __nv_bfloat16* sw = sdab + 2 * PB_ROWS * PB_DAB;     // the stacked weight, [CM n][64 k], swizzled
  __nv_bfloat16* sdy = sw + CM_MSA * PB_DAB;           // dout, then xh: CM/64 blocks of [128 rows][64], swizzled
  __nv_bfloat16* sy = sdy;
  __nv_bfloat16* smkt = sdy + (CM_MSA / 64) * PB_ROWS * 64;   // the mask column's block (only its first 8 columns are read)
  // gamma and beta appear nowhere in the inner loop any more.  gamma is folded into Wa and Wb on the
  // host, so the projection hands back g = dout * gamma directly; and y is not materialised at all --
  // dWa = gamma .* (da^T xh) + beta (x) sum_r da, so the GEMM runs against xh and the two scalars are
  // applied once, to the 32x64 result.  Reading them per element cost 8-way conflicted LDS.128 and was
  // the single largest consumer of this kernel's shared bandwidth; four ways of keeping them closer
  // (registers, volatile, PB_PART=8, constant memory) all measured worse than the conflicts.
  // the tile's mask, as the bf16 it arrives in: four tokens of one s row are 8 contiguous bytes, so
  // it rides in on cp.async with the rest of the staging instead of through a register (whose load
  // latency was exposed at the store: 4.5% of the stall samples)
  __shared__ __align__(16) __nv_bfloat16 smk[2][PB_ROWS];

  const int tid = threadIdx.x, lane = tid & 31;
  const int wgi = tid >> 7, wl = (tid >> 5) & 3;       // warpgroup, and the warp inside it
  const int i0 = blockIdx.x * PB_BI;

  // The projection weight is constant for the whole kernel, so it is staged ONCE, not once per s
  // block: eight passes were re-reading the same 8 KiB from global and re-writing it to shared.
  for (int v = tid; v < CM_MSA * PB_DAB / 8; v += PB_THREADS) {
    const int n = v >> 3, kc = v & 7;
    *reinterpret_cast<uint4*>(sw + wg::off(0, n, kc)) =
        *reinterpret_cast<const uint4*>(BWP + n * PB_DAB + kc * 8);
  }
  wg::proxy_fence();     // wgmma reads sw through the async proxy

  // dWa and dWb are one m64 x n64 wgmma tile: rows 0-31 are dWa, rows 32-63 dWb, columns the 64
  // channels.  The two warpgroups split the 128 k rows between them and their partial sums are added
  // once, at the end -- splitting n instead would need a 32-wide operand tile, which the 128-byte
  // swizzle cannot express.
  constexpr int PB_NW = CM_MSA / 2 + 4;                // wacc registers: m64 x n(CM + 8), the mask column's n tile on top
  float wacc[PB_NW];
#pragma unroll
  for (int t = 0; t < PB_NW; ++t) wacc[t] = 0.f;
  // the MN blocks of the dW GEMM's B operand (xh blocks, then the mask block) are one 128-row block apart
  constexpr uint32_t PB_LBO = (uint32_t)(PB_ROWS * 64 * 2 / 16);

  // dA/dB arrive as [s, (i,c)], so a row's 32 channels are contiguous on both sides and the copy is
  // four 16-byte moves.  Taking them as [(i,c), s] instead made this a transpose: sixteen scalar
  // two-byte shared stores per thread, eight-way bank conflicted, 15% of the kernel's instructions
  // but 28% of its stalls.  cuBLAS charges 8 us for the orientation.
  //
  // The mask no longer rides along here.  It is a per-ROW scalar, so it can be applied to either side
  // of both contractions -- dy = mask .* (da Wa + db Wb) and da^T (mask .* xh) == (mask .* da)^T xh --
  // and moving it onto the two outputs is what lets this be cp.async: global straight to shared, no
  // register round trip and no shared stores.  Two buffers deep, so the next s block's 32 KiB is in
  // flight across the whole of this one's projection, LayerNorm and dW GEMM: the exposed latency in
  // front of the projection was 32% of this kernel's stall samples.
  // A thread's two 16-byte slots never move -- only the s block does -- so every address below is a
  // per-thread constant plus t0 * (N * CH).  Recomputing them per copy was 11% of the kernel's
  // instructions (64-bit multiplies and two divisions per slot, eight s blocks, two slots each).
  static_assert(PB_ROWS * CH / 8 == 2 * PB_THREADS, "two 16-byte slots per thread");
  const int st_r = tid >> 2, st_c0 = (tid & 3) * 8;                  // rows st_r and st_r + 64
  const long st_g = (long)(st_r / PB_BI) * N * CH + (long)(i0 + (st_r & (PB_BI - 1))) * CH + st_c0;
  constexpr long ST_G1 = (long)(PB_THREADS / 4 / PB_BI);             // 16 s rows further, in units of N*CH
  const int st_sa = wg::off(0, st_r, st_c0 >> 3), st_sb = wg::off(0, st_r, (CH + st_c0) >> 3);
  constexpr int ST_S1 = (PB_THREADS / 4) * 64;                       // row st_r + 64: same swizzle phase
  static_assert(((PB_THREADS / 4) & 7) == 0, "the second slot must share the first one's swizzle");
  static_assert(PB_BI == 4, "one 8-byte mask copy per s row");
  auto stage = [&](int sb, int buf) {
    const int t0 = (blockIdx.y * PB_SB + sb) * PB_BS;
    if (tid < PB_BS) __pipeline_memcpy_async(&smk[buf][tid * PB_BI], MASK + (long)(t0 + tid) * N + i0, 8);
    __nv_bfloat16* dst = sdab + buf * (PB_ROWS * PB_DAB);
    const long g0 = (long)t0 * N * CH + st_g, g1 = g0 + ST_G1 * N * CH;
    __pipeline_memcpy_async(dst + st_sa, DA + g0, 16);
    __pipeline_memcpy_async(dst + st_sb, DB + g0, 16);
    __pipeline_memcpy_async(dst + st_sa + ST_S1, DA + g1, 16);
    __pipeline_memcpy_async(dst + st_sb + ST_S1, DB + g1, 16);
    __pipeline_commit();
  };
  stage(0, 0);

  for (int sbk = 0; sbk < PB_SB; ++sbk) {
  const int s0 = (blockIdx.y * PB_SB + sbk) * PB_BS;
  const int buf = sbk & 1;
  __nv_bfloat16* sdb_ = sdab + buf * (PB_ROWS * PB_DAB);
  const __nv_bfloat16* mk_ = smk[buf];
  __pipeline_wait_prior(0);
  // also separates the previous pass's dW GEMM from the prefetch below, which refills the buffer it
  // was reading two iterations ago
  __syncthreads();
  if (sbk + 1 < PB_SB) stage(sbk + 1, buf ^ 1);

  // dy = [da db] . [Wa; Wb] as ONE m128 x n64 x k64 GEMM -- the stacked tile makes the two
  // projections a single contraction.  It runs in two halves of n so the accumulator stays at
  // sixteen registers: the dW accumulator above is live across it, and 32 + 32 would cost a block.
#pragma unroll
  for (int h = 0; h < CM_MSA / 32; ++h) {
    float acc[16];
    wg::fence();
#pragma unroll
    for (int ks = 0; ks < PB_KS; ++ks)
      wg::mma_m64n32k16(wg::desc(sdb_ + wg::off(0, wgi * 64, ks * 2)),
                        wg::desc(sw + wg::off(0, h * 32, ks * 2)), acc, ks == 0 ? 0 : 1);
    wg::commit();
    wg::wait();
    // back through shared memory so the LayerNorm phase sees a row's 64 channels contiguously
#pragma unroll
    const int rw = wgi * 64 + wl * 16 + (lane >> 2);
    const float m0 = __bfloat162float(mk_[rw]), m1 = __bfloat162float(mk_[rw + 8]);   // once per thread
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = h * 32 + nt * 8 + (lane & 3) * 2;
      *reinterpret_cast<__nv_bfloat162*>(sdy + wg::off(col >> 6, rw, (col >> 3) & 7) + (col & 7)) =
          __float22bfloat162_rn(make_float2(acc[nt * 4 + 0] * m0, acc[nt * 4 + 1] * m0));
      *reinterpret_cast<__nv_bfloat162*>(sdy + wg::off(col >> 6, rw + 8, (col >> 3) & 7) + (col & 7)) =
          __float22bfloat162_rn(make_float2(acc[nt * 4 + 2] * m1, acc[nt * 4 + 3] * m1));
    }
  }
  __syncthreads();


  // LayerNorm forward statistics + backward, eight threads per row over contiguous channels.  The
  // read of m for the NEXT pass goes out before this pass's arithmetic: four dependent passes with an
  // exposed global load each was most of what remained of this kernel's long-scoreboard stall.
  // A pass advances the row by 32, a multiple of 8, so a thread's swizzled chunk offset and its
  // (s, i) split never change: every address in the loop is a per-thread constant plus pass * stride.
  // This kernel issues 43M instructions to move 144 MB and 42% of them were integer arithmetic.
  constexpr int PB_PASSES = PB_ROWS / (PB_THREADS / PB_PART);
  constexpr int PB_RPP = PB_THREADS / PB_PART;         // 32 rows per pass
  const int part = (tid & (PB_PART - 1)) * PB_CPT;
  const int tr = tid / PB_PART;                       // this thread's row within the pass
  const int tile_off = wg::off(part >> 6, tr, (part >> 3) & 7);        // == wg::off(part>>6, r, ..) - pass*RPP*64
  const long grow0 = (long)(s0 + tr / PB_BI) * N + (i0 + (tr & (PB_BI - 1)));   // == row of pass 0
  constexpr long PB_GSTEP = (long)(PB_RPP / PB_BI);    // rows of m between passes, in units of N
  const __nv_bfloat16* mptr = M + grow0 * CM_MSA + part;
  const float* sptr = STATS + grow0 * 2;
  __nv_bfloat16* dptr = DM + grow0 * CM_MSA + part;
  uint4 v0 = *reinterpret_cast<const uint4*>(mptr);
  // The forward already computed each row's mean and rstd; re-deriving them here was 16 FLOPs, six
  // shuffle rounds and an rsqrt per pass -- a seventh of the LayerNorm phase, in a kernel that is
  // bound by instruction issue.  8 bytes per row instead, prefetched with m.
  float2 st0 = *reinterpret_cast<const float2*>(sptr);
  for (int pass = 0; pass < PB_PASSES; ++pass) {
    const int r = pass * PB_RPP + tr;
    const int toff = tile_off + pass * (PB_RPP * 64);
    uint4 vn = v0;
    float2 stn = st0;
    if (pass + 1 < PB_PASSES) {
      vn = *reinterpret_cast<const uint4*>(mptr + (pass + 1) * PB_GSTEP * N * CM_MSA);
      stn = *reinterpret_cast<const float2*>(sptr + (pass + 1) * PB_GSTEP * N * 2);
    }
    float x[PB_CPT], dy[PB_CPT];
    {
      const uint4 d0 = *reinterpret_cast<const uint4*>(sdy + toff);
      const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
      const __nv_bfloat16* e0 = reinterpret_cast<const __nv_bfloat16*>(&d0);
#pragma unroll
      for (int t = 0; t < 8; ++t) { x[t] = __bfloat162float(h0[t]); dy[t] = __bfloat162float(e0[t]); }
    }
    const float mean = st0.x, rstd = st0.y;
    float gs = 0.f, gx = 0.f;                          // x becomes xh in place, dy becomes g in place
#pragma unroll
    for (int t = 0; t < PB_CPT; ++t) x[t] = (x[t] - mean) * rstd;
    // dgamma wants sum_r dout * xh.  Put the product back where dout was -- each element is read by
    // exactly this thread and dout is not wanted again -- and the column sum after this phase takes it.
    // 16 bytes per store, not 4: a 4-byte store leaves only eight distinct banks across the warp
    // (the row stride and the channel offset are both multiples of four), so it was 4-way conflicted.
    // Both products are formed in packed bf16 from the values that are being rounded to bf16 anyway:
    // 8 packs + 8 HMUL2 instead of 16 FMUL + 8 packs.  The dgamma product goes back where dout was
    // (each element is read by exactly this thread and dout is not wanted again); xh goes out in the
    // wgmma swizzle, because the dW GEMM reads it as an operand tile, and carries the mask:
    // da^T (mask .* xh) is the same product as (mask .* da)^T xh.
    // Both products are formed in packed bf16 from values that are being rounded to bf16 anyway:
    // 8 packs + 8 HMUL2 instead of 16 FMUL + 8 packs.  The dgamma product goes back where dout was
    // (each element is read by exactly this thread and dout is not wanted again); xh goes out in the
    // wgmma swizzle, because the dW GEMM reads it as an operand tile, and carries the mask:
    // da^T (mask .* xh) is the same product as (mask .* da)^T xh.
    {
      const __nv_bfloat162 mk2 = __bfloat162bfloat162(mk_[r]);
      uint4 vy;
      __nv_bfloat162* hy = reinterpret_cast<__nv_bfloat162*>(&vy);
#pragma unroll
      for (int u = 0; u < 4; ++u)
        hy[u] = __hmul2(__float22bfloat162_rn(make_float2(x[u * 2], x[u * 2 + 1])), mk2);
      *reinterpret_cast<uint4*>(sy + toff) = vy;
    }
#pragma unroll
    for (int t = 0; t < PB_CPT; ++t) { gs += dy[t]; gx += dy[t] * x[t]; }
#pragma unroll
    for (int off = 1; off < PB_PART; off <<= 1) {
      gs += __shfl_xor_sync(0xffffffff, gs, off);
      gx += __shfl_xor_sync(0xffffffff, gx, off);
    }
    // dm = rstd * (g - mean(g) - xh * mean(g xh)) as two FMAs per element: fma(-xh, rstd*gx, fma(g, rstd, -rstd*gs))
    const float ra = -rstd * gs * (1.f / CM_MSA), rb = rstd * gx * (1.f / CM_MSA);
    {
      uint4 v;
      __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int tt = u * 2;
        h[u] = __float22bfloat162_rn(make_float2(fmaf(-x[tt], rb, fmaf(dy[tt], rstd, ra)),
                                                 fmaf(-x[tt + 1], rb, fmaf(dy[tt + 1], rstd, ra))));
      }
      *reinterpret_cast<uint4*>(dptr + pass * PB_GSTEP * N * CM_MSA) = v;
    }
    v0 = vn; st0 = stn;
  }
  __syncthreads();

  // The masked column sums sum_r mask_r [da db][r][k] -- the beta half of dWa/dWb, and dbeta on the
  // host -- come out of the dW GEMM itself: its B operand grows a ninth n tile whose first column is
  // the mask, in a second 64-wide MN block that lives in the dead dout tile.  wgmma reads only that
  // tile's first 8 columns, so one 16-byte store per row (mask, 0 x 7) is the whole cost; it replaces
  // 32 masked scalar loads per thread on half the block.
  if (tid < PB_ROWS) {
    *reinterpret_cast<uint4*>(smkt + wg::off_mn(0, tid, 0, PB_ROWS)) =
        make_uint4(static_cast<uint32_t>(__bfloat16_as_ushort(mk_[tid])), 0u, 0u, 0u);
  }
  // sy was written with ordinary stores; the wgmma below reads it through the async proxy
  wg::proxy_fence();
  __syncthreads();

  // [dWa; dWb] = [da db]^T . xh over this CTA's rows.  Both operands are [k][mn] tiles already, so
  // the MN-major descriptor takes them as they lie -- no ldmatrix.trans, no transposed staging.  The
  // only reason y and da/db used to reach HBM -- feeding two cuBLAS calls for 2 x 2048 numbers -- is
  // gone: 200 MB of the kernel's 250 MB of traffic was that round trip.
  {
    wg::fence();
#pragma unroll
    for (int ks = 0; ks < PB_ROWS / 2 / 16; ++ks) {
      const int k0 = wgi * (PB_ROWS / 2) + ks * 16;
      if constexpr (CM_MSA == 64)
        wg::mma_m64n72k16_mn(wg::desc_mn(sdb_ + wg::off_mn(0, k0, 0, PB_ROWS), 1),
                             wg::desc_mn(sy + wg::off_mn(0, k0, 0, PB_ROWS), PB_LBO), wacc, 1);
      else
        wg::mma_m64n136k16_mn(wg::desc_mn(sdb_ + wg::off_mn(0, k0, 0, PB_ROWS), 1),
                              wg::desc_mn(sy + wg::off_mn(0, k0, 0, PB_ROWS), PB_LBO), wacc, 1);
    }
    wg::commit();
    wg::wait();
  }
  }                                                    // end of the s loop

  // the two warpgroups hold partial sums over disjoint halves of the rows; fold them through the
  // staging tile, which is dead by now
  {
    float* red = reinterpret_cast<float*>(sdab);
    const int mrow = wl * 16 + (lane >> 2);
    __syncthreads();
    if (wgi == 1) {
#pragma unroll
      for (int nt = 0; nt < CM_MSA / 8 + 1; ++nt) {
        const int n = nt * 8 + (lane & 3) * 2;
        *reinterpret_cast<float2*>(red + mrow * PB_RED + n) = make_float2(wacc[nt * 4 + 0], wacc[nt * 4 + 1]);
        *reinterpret_cast<float2*>(red + (mrow + 8) * PB_RED + n) = make_float2(wacc[nt * 4 + 2], wacc[nt * 4 + 3]);
      }
    }
    __syncthreads();
    if (wgi == 0) {
      const long blk = (long)blockIdx.y * gridDim.x + blockIdx.x;
      float* p = (wl < 2 ? DWA : DWB) + blk * PB_PW;
      const int rw = (wl & 1) * 16 + (lane >> 2);
#pragma unroll
      for (int nt = 0; nt < CM_MSA / 8; ++nt) {
        const int n = nt * 8 + (lane & 3) * 2;
        const float2 a = *reinterpret_cast<const float2*>(red + mrow * PB_RED + n);
        const float2 b = *reinterpret_cast<const float2*>(red + (mrow + 8) * PB_RED + n);
        *reinterpret_cast<float2*>(p + (size_t)rw * CM_MSA + n) =
            make_float2(wacc[nt * 4 + 0] + a.x, wacc[nt * 4 + 1] + a.y);
        *reinterpret_cast<float2*>(p + (size_t)(rw + 8) * CM_MSA + n) =
            make_float2(wacc[nt * 4 + 2] + b.x, wacc[nt * 4 + 3] + b.y);
      }
      // the ninth tile's first column: ssa[k] for the stacked channel k = this accumulator row
      if ((lane & 3) == 0) {
        float* q = DWA + blk * PB_PW + 2 * CH * CM_MSA;
        q[mrow] = wacc[CM_MSA / 2] + red[mrow * PB_RED + CM_MSA];
        q[mrow + 8] = wacc[CM_MSA / 2 + 2] + red[(mrow + 8) * PB_RED + CM_MSA];
      }
    }
  }
}
}  // namespace (prologue backward)

// ---------------------------------------------------------------------------------------------
// dWo, without the permute.  dWo[z,(c,e)] = sum_{i,j} (dz/n)[i,j,z] * O[(i,c),(j,e)] reads O in the
// GROUPED layout it is already in, instead of materialising a [N,N,c_hidden^2] transpose of it
// (302 MB written and read again, 0.45 ms at L=384/S=1024) just to hand cuBLAS a matmul.
//
// It is a GEMM with M = c_z, N = c_hidden^2 and K = the N^2 pairs, so K is split across CTAs and the
// partials are summed afterwards -- no atomics, so the result does not depend on CTA order.
namespace {
constexpr int DW_JB = 64;                // j per k step: one (i, j-block) gives 4 x 512 contiguous O.
                                         // At 16 a step was two cp.async, two barriers and the whole
                                         // index recomputation for only 16 mmas, and address
                                         // arithmetic came to a quarter of the kernel's instructions.
constexpr int DW_NG = 128;               // (c,e) columns per CTA.  Halving it to 64 gives 8 tile
                                         // pairs per warp, 80 registers and three blocks per SM
                                         // instead of two, but the extra L1 traffic (42.7 -> 55.3%
                                         // of peak) cancels the occupancy: 219 -> 226 us.
constexpr int DW_IB = CZ > 128 ? 8 : 4;  // whole i rows per CTA: a k slice must not straddle two rows (8 when c_z is wide: halves the partial buffer)
constexpr int DW_WARPS = 8;
constexpr int DW_THREADS = DW_WARPS * 32;
constexpr int DW_ZT = ZW / 16;            // 8 mma m tiles down one c_z slice
constexpr int DW_NT = DW_NG / 8;         // 16 mma n tiles across the (c,e) columns
// A warp takes a 4 x 4 square of them rather than a 1 x 16 strip.  With one z tile a warp needed a
// fresh B fragment for every single mma, and the HMMAs spent 85% of their stall time on short
// scoreboard waiting for that ldmatrix; a square shares four A and four B loads across sixteen mmas
// for the same bytes, and halves the accumulator registers as well.
constexpr int DW_ZW = 4, DW_NW = 4;      // tiles per warp, DW_ZW * DW_NW warps cover the block
static_assert(DW_ZT / DW_ZW * (DW_NT / DW_NW) == DW_WARPS, "the warp grid must tile the block");

// Both source streams are walked with a running pointer.  Recomputing the two global addresses per
// cp.async -- each a 64-bit multiply by NJ or by the row pitch -- put IMAD and LEA at 42% of this
// kernel's instructions, more than three times the HMMA count.
//
// THREADS divides each staging count, so a thread's slot within the tile never changes: for dzp its
// four copies differ by a constant 16 j, for O by a constant one (c) row, and a step moves both
// pointers by a fixed stride.
struct dw_src {
  const __nv_bfloat16* a;                     // dzp, at this thread's (j, z) slot
  const __nv_bfloat16* b;                     // O, at this thread's (j, e) slot of the first c row
};

__device__ __forceinline__ void dw_stage(const dw_src& s, __nv_bfloat16* sA, __nv_bfloat16* sB,
                                         long M, int tid) {
  constexpr int AJ = DW_THREADS / (ZW / 8);          // j rows of dzp a thread's slot skips per copy
  const int az = (tid % (ZW / 8)) * 8, aj = tid / (ZW / 8);
  const int be = (tid % (CH / 8)) * 8, bj = (tid / (CH / 8)) % DW_JB;
#pragma unroll
  for (int k = 0; k < DW_JB * ZW / 8 / DW_THREADS; ++k)
    __pipeline_memcpy_async(sA + wg::off_mn(az >> 6, aj + k * AJ, (az >> 3) & 7, DW_JB),
                            s.a + (long)k * AJ * CZ, 16);
#pragma unroll
  for (int k = 0; k < DW_JB * DW_NG / 8 / DW_THREADS; ++k) {
    const int ce = k * CH + be;                      // this copy's (c,e) column in the block
    __pipeline_memcpy_async(sB + wg::off_mn(ce >> 6, bj, (ce >> 3) & 7, DW_JB), s.b + (long)k * M, 16);
  }
  __pipeline_commit();
}

// The k loop used to be: barrier, load, barrier, mma -- 192 barriers per CTA with nothing overlapping
// the loads. Two cp.async buffers let step k+1 arrive while step k multiplies.
__global__ __launch_bounds__(DW_THREADS) void opm_dwo_kernel(
    const __nv_bfloat16* __restrict__ DZP,    // [NI, NJ, CZ] already divided by the mask count
    const __nv_bfloat16* __restrict__ O,      // [NI*CH, NJ*CH]
    float* __restrict__ PART,                 // [ksplits, CZ, NCH]
    int NI, int NJ, long M, int ksplit) {
  constexpr int ATILE = (ZW / 64) * DW_JB * 64;      // one swizzled dzp tile, [z block][j][64 z]
  constexpr int BTILE = (DW_NG / 64) * DW_JB * 64;   // one swizzled O tile
  extern __shared__ __nv_bfloat16 dw_smem_raw[];
  // wgmma's swizzle reads absolute shared addresses, so the tiles start on a 1 KiB boundary
  const uint32_t dwbase = static_cast<uint32_t>(__cvta_generic_to_shared(dw_smem_raw));
  __nv_bfloat16* dw_smem = dw_smem_raw + (((1024u - (dwbase & 1023u)) & 1023u) >> 1);
  __nv_bfloat16* sA = dw_smem;                       // [2 buffers][ATILE]
  __nv_bfloat16* sB = sA + 2 * ATILE;                // [2 buffers][BTILE]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int zs = blockIdx.x % NZS;                   // c_z slice, fastest: the slices of one O tile run together
  const int z0 = zs * ZW;
  const int ng = (blockIdx.x / NZS) * DW_NG;
  const int i00 = blockIdx.y * DW_IB;
  const int per_row = NJ / DW_JB, steps = DW_IB * per_row;

  const int zb = (warp % (DW_ZT / DW_ZW)) * DW_ZW;    // this warp's first z tile
  const int nb = (warp / (DW_ZT / DW_ZW)) * DW_NW;    // ... and its first n tile
  const int wgi = tid >> 7, wl = (tid >> 5) & 3;      // warpgroup, and the warp inside it
  float acc[64];                                     // this warpgroup's 64 z x 128 (c,e) tile

  // this thread's slot in the first staged block, and the strides that walk it forward
  dw_src src;
  {
    const int az = (tid % (ZW / 8)) * 8, aj = tid / (ZW / 8);
    const int be = (tid % (CH / 8)) * 8, bj = (tid / (CH / 8)) % DW_JB, bc = (tid / (CH / 8)) / DW_JB;
    src.a = DZP + ((long)i00 * NJ + aj) * CZ + z0 + az;
    src.b = O + ((long)i00 * CH + ng / CH + bc) * M + (long)bj * CH + be;
  }
  const long a_step = (long)DW_JB * CZ, b_step = (long)DW_JB * CH;
  const long b_wrap = (long)CH * M - (long)(NJ - DW_JB) * CH;   // i advances, j returns to zero
  dw_stage(src, sA, sB, M, tid);
#pragma unroll 1
  for (int st = 0; st < steps; ++st) {
    const int buf = st & 1;
    if (st + 1 < steps) {
      const bool wrap = ((st + 1) % per_row) == 0;
      src.a += a_step;
      src.b += wrap ? b_wrap : b_step;
      dw_stage(src, sA + (1 - buf) * ATILE, sB + (1 - buf) * BTILE, M, tid);
    }
    __pipeline_wait_prior(st + 1 < steps ? 1 : 0);
    __syncthreads();
    const __nv_bfloat16* pa = sA + buf * ATILE;
    const __nv_bfloat16* pb = sB + buf * BTILE;
    wg::fence();
#pragma unroll
    for (int kk = 0; kk < DW_JB / 16; ++kk) {          // one staged j block is DW_JB/16 wgmma k steps
      wg::mma_m64n128k16_mn(
          wg::desc_mn(pa + wg::off_mn(wgi, kk * 16, 0, DW_JB), 1),
          wg::desc_mn(pb + wg::off_mn(0, kk * 16, 0, DW_JB), DW_JB * 8), acc,
          (st == 0 && kk == 0) ? 0 : 1);
    }
    wg::commit();
    wg::wait();
    __syncthreads();
  }

  float* out = PART + (long)blockIdx.y * CZ * NCH + (long)z0 * NCH + ng;
  {
    float* o0 = out + (size_t)(wgi * 64 + wl * 16 + (lane >> 2)) * NCH + (lane & 3) * 2;
    float* o8 = o0 + (size_t)8 * NCH;
#pragma unroll
    for (int nt = 0; nt < DW_NG / 8; ++nt) {
      *reinterpret_cast<float2*>(o0 + nt * 8) = make_float2(acc[nt * 4 + 0], acc[nt * 4 + 1]);
      *reinterpret_cast<float2*>(o8 + nt * 8) = make_float2(acc[nt * 4 + 2], acc[nt * 4 + 3]);
    }
  }
}
}  // namespace (dWo)

// ---------------------------------------------------------------------------------------------
// Sum a [splits, width] fp32 partial buffer down the split axis.  Every split-K step here ended in
// torch's dim-0 reduce, and on a tall, thin tensor it runs at a tenth of the bandwidth it should:
// 169 us to read the 50 MB of dWo partials, plus 67 us more for the prologue's four.  This reads
// each partial exactly once, coalesced, and sums in split order, so it stays deterministic.
namespace {
__global__ void opm_reduce_partials_chunked(const float* __restrict__ P, float* __restrict__ OUT,
                                            int splits, int chunk, int G, long width) {
  // Threads are laid out columns-first, so a narrow buffer spreads across the split axis instead of
  // leaving most of the block idle: at width 64 a column-only mapping gave 16 live threads and took
  // 57 us to read 786 KB.
  const int vcols = (int)(width >> 2);
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int v4 = (int)(t % vcols), g = (int)(t / vcols);
  if (g >= G) return;
  const int k0 = g * chunk, k1 = min(k0 + chunk, splits);
  const long v = (long)v4 * 4;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
  for (int k = k0; k < k1; ++k) {
    const float4 q = *reinterpret_cast<const float4*>(P + (long)k * width + v);
    s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
  }
  *reinterpret_cast<float4*>(OUT + (long)g * width + v) = s;
}

// Sums in split order, so the result does not depend on how the splits were chunked.  The split
// count is chosen to keep roughly 16k threads busy -- enough outstanding loads to cover DRAM
// latency -- while leaving the second pass few enough splits to finish in a couple of microseconds.
torch::Tensor reduce_partials(const torch::Tensor& part) {
  const int splits = (int)part.size(0);
  const long width = part.numel() / splits;
  TORCH_CHECK(width % 4 == 0, "partial width must be a multiple of four floats");
  const long vcols = width / 4;
  auto pass = [&](const torch::Tensor& src, torch::Tensor& dst, int s, int c, int g) {
    const long threads_needed = vcols * g;
    const int threads = 256, blocks = (int)((threads_needed + threads - 1) / threads);
    opm_reduce_partials_chunked<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), s, c, g, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  auto out_shape = part.sizes().slice(1).vec();
  long G = 16384 / (vcols > 0 ? vcols : 1);
  if (G > 64) G = 64;
  if (G > splits / 2) G = splits / 2;
  if (G < 1) G = 1;
  const int chunk = (int)((splits + G - 1) / G);
  G = (splits + chunk - 1) / chunk;                  // chunk rounding can drop a step
  if (G == 1) {
    auto out = torch::empty(out_shape, part.options());
    pass(part, out, splits, splits, 1);
    return out;
  }
  auto tmp = torch::empty({G, width}, part.options());
  pass(part, tmp, splits, chunk, (int)G);
  auto out = torch::empty(out_shape, part.options());
  pass(tmp, out, (int)G, (int)G, 1);
  return out;
}
}  // namespace

torch::Tensor opm_dwo(torch::Tensor dzp, torch::Tensor o, int64_t ni, int64_t nj) {
  TORCH_CHECK(dzp.is_contiguous() && o.is_contiguous(), "dzp and O must be contiguous");
  TORCH_CHECK(ni % DW_IB == 0, "the kernel splits k in whole groups of ", DW_IB, " i rows");
  TORCH_CHECK(nj % DW_JB == 0, "the kernel steps k in whole blocks of ", DW_JB, " j");
  const int ksplit = (int)(ni / DW_IB);
  // empty, not zeros: every CTA assigns (never accumulates into) its own slice, so zeroing 50 MB
  // first was 16 us of pure waste
  auto part = torch::empty({(long)ksplit, (long)CZ, (long)NCH}, dzp.options().dtype(torch::kFloat32));
  dim3 grid((NCH / DW_NG) * NZS, ksplit);
  const int dw_smem = 1024 + 2 * DW_JB * (ZW + DW_NG) * 2;   // two swizzled tiles, double buffered
  static bool dw_attr = false;                   // a wide k step pushes the tiles past 48 KiB
  if (!dw_attr) {
    cudaFuncSetAttribute(opm_dwo_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024);
    dw_attr = true;
  }
  opm_dwo_kernel<<<grid, DW_THREADS, dw_smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dzp.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(o.data_ptr<at::BFloat16>()),
      part.data_ptr<float>(), (int)ni, (int)nj, nj * CH, ksplit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return reduce_partials(part);
}

std::vector<torch::Tensor> opm_prologue_bwd(torch::Tensor dA, torch::Tensor dB, torch::Tensor m,
                                            torch::Tensor stats, torch::Tensor mask, torch::Tensor gamma, torch::Tensor beta,
                                            double eps, torch::Tensor bwp, int64_t N, int64_t S,
                                            bool native_finish = false,
                                            std::vector<torch::Tensor> raw_weights = {}) {
  TORCH_CHECK(dA.is_contiguous() && dB.is_contiguous() && m.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(stats.is_contiguous() && stats.scalar_type() == torch::kFloat32 && stats.numel() == S * N * 2,
              "stats: contiguous fp32 [S, N, 2] (mean, rstd) from the forward prologue");
  TORCH_CHECK(mask.is_contiguous() && mask.scalar_type() == torch::kBFloat16, "mask: contiguous bf16 [S, N]");
  TORCH_CHECK(N % PB_BI == 0 && S % (PB_BS * PB_SB) == 0,
              "the kernel tiles ", PB_BI, " tokens x ", PB_BS * PB_SB, " rows");
  auto opts = m.options();
  auto dm = torch::empty({1, S, N, (long)CM_MSA}, opts);
  dim3 grid(N / PB_BI, S / (PB_BS * PB_SB));
  TORCH_CHECK(bwp.numel() == CM_MSA * 2 * CH, "bwp: [CM, 2*CH] = (gamma*Wa)^T | (gamma*Wb)^T");
  // one extra KiB so the wgmma operand tiles can start on a 1 KiB boundary
  const int pb_smem = 1024 + (2 * PB_ROWS * PB_DAB + CM_MSA * PB_DAB + (CM_MSA / 64 + 1) * PB_ROWS * 64) * 2;   // sdab x2, sw, sdy (= sy), mask block
  static bool pb_attr = false;
  if (!pb_attr) {
    cudaFuncSetAttribute(opm_prologue_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024);
    pb_attr = true;
  }
  auto fopts = opts.dtype(torch::kFloat32);
  const long blocks = (long)grid.x * grid.y;
  auto part = torch::empty({blocks, (long)PB_PW}, fopts);
  float* pbase = part.data_ptr<float>();
  constexpr long W = CH * CM_MSA;
  opm_prologue_bwd_kernel<<<grid, PB_THREADS, pb_smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dA.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(dB.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(m.data_ptr<at::BFloat16>()),
      stats.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(mask.data_ptr<at::BFloat16>()),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), (float)eps,
      reinterpret_cast<const __nv_bfloat16*>(bwp.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(dm.data_ptr<at::BFloat16>()),
      pbase, pbase + W, (int)N, (int)S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto red = reduce_partials(part);
  if (native_finish) {
    TORCH_CHECK(raw_weights.size()==2 && raw_weights[0].scalar_type()==torch::kBFloat16
        && raw_weights[1].scalar_type()==torch::kBFloat16,"native finish needs raw BF16 projection weights");
    auto da=torch::empty({32,64},fopts), db=torch::empty({32,64},fopts);
    auto dg=torch::empty({64},fopts), de=torch::empty({64},fopts);
    opm_native_launch(opm_finish_native,dim3(8),dim3(256),
        red.data_ptr<float>(),gamma.data_ptr<float>(),beta.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(raw_weights[0].data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(raw_weights[1].data_ptr()),da.data_ptr<float>(),
        db.data_ptr<float>(),dg.data_ptr<float>(),de.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dm,da,db,dg,de};
  }
  // The kernel ran its GEMM against xh and saw g = dout * gamma, so gamma and beta are applied once
  // here instead of once per element inside it: dWa = gamma .* (da^T xh) + beta (x) sum_r da, and the
  // two LayerNorm gradients come back gamma-scaled.
  const long C = CH, M2 = CM_MSA;
  auto g2 = gamma.view({1, M2});
  auto R = red.slice(0, 0, 2 * W).view({2 * C, M2});             // raw [da db]^T (mask .* xh)
  auto sums = red.slice(0, 2 * W, PB_PW);                          // ssa: [da c | db c]
  auto dwa = R.slice(0, 0, C) * g2 + sums.slice(0, 0, C).view({C, 1}) * beta.view({1, M2});
  auto dwb = R.slice(0, C, 2 * C) * g2 + sums.slice(0, C, 2 * C).view({C, 1}) * beta.view({1, M2});
  // dout = mask .* ([da db] Wf) with Wf = gamma .* W (bwp holds Wf^T in bf16, exactly what the kernel
  // multiplied by), so  dgamma = (sum_r dout .* xh) / gamma = (Wf .* R).sum(k) / gamma  and
  // dbeta = (sum_r dout) / gamma = (Wf^T ssa) / gamma.
  auto wf = bwp.to(torch::kFloat32).t();                           // [2C k][M2]
  auto dgamma = (wf * R).sum(0) / gamma;
  auto dbeta = wf.t().matmul(sums) / gamma;
  return {dm, dwa, dwb, dgamma, dbeta};
}

// cuTensorMapEncodeTiled lives in the driver, not in the runtime: the extension resolves it once
// through the driver entry-point table rather than linking libcuda.
namespace {
PFN_cuTensorMapEncodeTiled tma_encode() {
  static PFN_cuTensorMapEncodeTiled fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled>(p);
  }
  return fn;
}
}  // namespace

std::vector<torch::Tensor> opm_dgrad(torch::Tensor dz, torch::Tensor norm, torch::Tensor bw, int64_t ni, int64_t nj, int64_t after) {
  TORCH_CHECK(dz.is_cuda() && dz.scalar_type() == torch::kBFloat16 && dz.is_contiguous(), "dz: contiguous cuda bf16");
  TORCH_CHECK(norm.scalar_type() == torch::kFloat32 && norm.is_contiguous(), "norm: contiguous fp32");
  TORCH_CHECK(bw.numel() == CZ * NCH, "bw: Wo transposed, [NCH, CZ]");
  TORCH_CHECK(ni % DG_BI == 0 && nj % DG_BJ == 0, "the kernel tiles ", DG_BI, "x", DG_BJ, " tokens");
  auto dO = torch::empty({ni * CH, nj * CH}, dz.options());
  auto dzp = torch::empty({ni, nj, (long)CZ}, dz.options());
  auto dbo_part = torch::empty({(ni / DG_BI) * (nj / DG_BJ), (long)CZ}, dz.options().dtype(torch::kFloat32));
  const int smem = 1024 + ((KZ / 64) * DG_NP * 64 + (KZ >= 384 ? 2 : 1) * (KZ / 64) * CHUNK * 64 + DG_NP * CHUNK) * 2;
  static bool attr = false;                      // 256-wide chunks push past the 48 KiB static limit
  if (!attr) {
    cudaFuncSetAttribute(opm_dgrad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    attr = true;
  }
  // dO is [ni*CH, nj*CH]; TMA sees it as the 4-D tensor it really is, innermost first:
  //   (64 columns = two whole j) x (nj*CH/64) x c x i
  // so the CTA's tile for one chunk -- DG_BI i, two c, DG_BJ j, all 32 e -- is a single box.
  const long M = nj * CH;
  TORCH_CHECK(M % 64 == 0, "TMA needs the row length in whole 128-byte groups");
  alignas(64) CUtensorMap dO_map{};
  {
    uint64_t gdim[4] = {64, (uint64_t)(M / 64), (uint64_t)CH, (uint64_t)ni};
    uint64_t gstride[3] = {128, (uint64_t)M * 2, (uint64_t)CH * M * 2};   // bytes, dims 1..3
    uint32_t bdim[4] = {64, DG_BJ / 2, CHUNK / CH, DG_BI};
    uint32_t estride[4] = {1, 1, 1, 1};
    CUresult r = tma_encode()(&dO_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                              dO.data_ptr(), gdim, gstride, bdim, estride,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  }
  dim3 grid(nj / DG_BJ, ni / DG_BI);
  opm_dgrad_kernel<<<grid, DG_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dz.data_ptr<at::BFloat16>()), norm.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(bw.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(dzp.data_ptr<at::BFloat16>()), dbo_part.data_ptr<float>(),
      (int)ni, (int)nj, M, (int)after, dO_map);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dzp, reduce_partials(dbo_part)};
}

torch::Tensor opm_epilogue(torch::Tensor O, torch::Tensor norm, torch::Tensor bf, torch::Tensor bias,
                           int64_t ni, int64_t nj, c10::optional<torch::Tensor> residual, int64_t after) {
  TORCH_CHECK(O.is_cuda() && O.scalar_type() == torch::kBFloat16 && O.is_contiguous(), "O: contiguous cuda bf16");
  TORCH_CHECK(norm.scalar_type() == torch::kFloat32 && norm.is_contiguous(), "norm: contiguous fp32");
  TORCH_CHECK(bf.scalar_type() == torch::kBFloat16 && bf.is_contiguous() && bf.numel() == CH * CH * CZ, "bf: Wo [CZ, CH*CH] bf16, contiguous");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && bias.numel() == CZ, "bias: fp32 [CZ]");
  TORCH_CHECK(ni % BI == 0, "the epilogue tiles ", BI, " i tokens, got ", ni);
  TORCH_CHECK(nj % BJ == 0, "the epilogue tiles ", BJ, " j tokens, got ", nj);
  if (residual.has_value()) {
    TORCH_CHECK(residual->device() == O.device() && residual->scalar_type() == torch::kBFloat16
                && residual->is_contiguous() && residual->sizes() == torch::IntArrayRef({1, ni, nj, CZ}),
                "residual: contiguous bf16 [1, ni, nj, CZ] on O's device");
  }
  const long M = O.size(1);
  auto out = torch::empty({1, ni, nj, (long)CZ}, O.options());
  dim3 grid((nj / BJ) * NZS, ni / BI);
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool ep_attr = false;                 // the staged weight pushes the tiles past 48 KiB
  if (!ep_attr) {
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_WIDE)
    cudaFuncSetAttribute(opm_epilogue_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 192 * 1024);
#else
    cudaFuncSetAttribute(opm_epilogue_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024);
#endif
    ep_attr = true;
  }
  // z[i, j, z] as TMA sees it, innermost first: 64 z, then j, then the z half, then i
  alignas(64) CUtensorMap out_map{};
  {
    uint64_t gdim[4] = {64, (uint64_t)nj, (uint64_t)(CZ / 64), (uint64_t)ni};
    uint64_t gstride[3] = {(uint64_t)CZ * 2, 128, (uint64_t)nj * CZ * 2};
    uint32_t bdim[4] = {64, BJ, 2, BI};
    uint32_t estride[4] = {1, 1, 1, 1};
    CUresult r = tma_encode()(&out_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                              out.data_ptr(), gdim, gstride, bdim, estride,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (epilogue) failed: ", (int)r);
  }
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)

  alignas(64) CUtensorMap o_map{}, w_map{};
  {
    uint64_t dims[4]={32,(uint64_t)nj,(uint64_t)ni,32};
    uint64_t strides[3]={64,(uint64_t)32*M*2,(uint64_t)M*2};
    uint32_t box[4]={32,BJ,BI,2},elem[4]={1,1,1,1};
    CUresult r=tma_encode()(&o_map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,O.data_ptr(),dims,strides,box,elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r==CUDA_SUCCESS,"OPM gathered input TMA encode failed ",int(r));
  }
  {
    uint64_t dims[2]={CH*CH,CZ},strides[1]={CH*CH*2};
    uint32_t box[2]={KC,CZ},elem[2]={1,1};
    CUresult r=tma_encode()(&w_map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,bf.data_ptr(),dims,strides,box,elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r==CUDA_SUCCESS,"OPM weight TMA encode failed ",int(r));
  }

#endif
  opm_epilogue_kernel<<<grid, THREADS, SMEM_BYTES + 1024, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
      norm.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(bf.data_ptr<at::BFloat16>()),
      bias.data_ptr<float>(),
      residual.has_value() ? reinterpret_cast<const __nv_bfloat16*>(residual->data_ptr<at::BFloat16>()) : nullptr,
#if defined(MINIWORLD_OPM_EPILOGUE_TMA_OVERLAP)
      (int)ni, (int)nj, M, (int)after, out_map, o_map, w_map);
#else
      (int)ni, (int)nj, M, (int)after, out_map);
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> opm_native_pack(std::vector<torch::Tensor> p) {
  TORCH_CHECK(p.size()==6,"six parameters required");
  const int sizes[6]={64,64,2048,2048,131072,128};
  int types=0;
  for(int k=0;k<6;++k) {
    TORCH_CHECK(p[k].is_cuda() && p[k].device()==p[0].device() && p[k].is_contiguous()
        && p[k].numel()==sizes[k],"native OPM canonical parameter contract");
    TORCH_CHECK(p[k].scalar_type()==torch::kFloat32 || p[k].scalar_type()==torch::kBFloat16,
        "native OPM parameters must be BF16 or FP32");
    if(p[k].scalar_type()==torch::kFloat32)types|=1<<k;
  }
  auto fo=p[0].options().dtype(torch::kFloat32), bo=p[0].options().dtype(torch::kBFloat16);
  auto g=torch::empty({64},fo),b=torch::empty({64},fo);
  auto a=torch::empty({32,64},bo),c=torch::empty({32,64},bo),o=torch::empty({128,1024},bo);
  auto ob=torch::empty({128},fo),ot=torch::empty({1024,128},bo),bw=torch::empty({64,64},bo);
  opm_native_launch(opm_pack_native,dim3(512),dim3(256),
      p[0].data_ptr(),p[1].data_ptr(),p[2].data_ptr(),p[3].data_ptr(),p[4].data_ptr(),p[5].data_ptr(),types,
      g.data_ptr<float>(),b.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(c.data_ptr()),reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
      ob.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(ot.data_ptr()),reinterpret_cast<__nv_bfloat16*>(bw.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {g,b,a,c,o,ob,ot,bw};
}
std::vector<torch::Tensor> opm_native_prologue_tma(torch::Tensor m,torch::Tensor mask,
 std::vector<torch::Tensor> p,double eps,bool save_stats){
 TORCH_CHECK(m.is_cuda()&&m.is_contiguous()&&m.scalar_type()==torch::kBFloat16&&m.dim()==3&&m.size(2)==64,"TMA prologue BF16 [S,N,64]");
 int S=m.size(0),N=m.size(1);
 TORCH_CHECK(S%256==0&&N%64==0&&mask.is_contiguous()&&mask.scalar_type()==torch::kBool&&mask.device()==m.device()&&mask.numel()==S*N&&p.size()==8,"TMA prologue shape/mask");
 auto fo=m.options().dtype(torch::kFloat32);
 auto a=torch::empty({N*32,S},m.options()),b=torch::empty_like(a),mk=torch::empty({S,N},m.options());
 auto norm=torch::empty({N,N},fo),stats=torch::empty(save_stats?std::vector<int64_t>{S,N,2}:std::vector<int64_t>{0},fo);
 auto bits=torch::empty({N,S/32},m.options().dtype(torch::kInt32));
 alignas(64) CUtensorMap xm{},mm{},wam{},wbm{},am{},bm{};
 auto enc=[&](CUtensorMap* map,CUtensorMapDataType dt,int rank,void* ptr,uint64_t* dims,uint64_t* strides,uint32_t* box,CUtensorMapSwizzle swz){
  uint32_t es[3]={1,1,1};auto rc=tma_encode()(map,dt,rank,ptr,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,swz,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc==CUDA_SUCCESS,"OPM prologue TMA encode: ",int(rc));
 };
 uint64_t xd[3]={64,uint64_t(N),uint64_t(S)},xs[2]={128,uint64_t(N)*128};uint32_t xb[3]={64,1,128};
 enc(&xm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,m.data_ptr(),xd,xs,xb,CU_TENSOR_MAP_SWIZZLE_128B);
 uint64_t md[2]={uint64_t(N),uint64_t(S)},ms[1]={uint64_t(N)};uint32_t mb[2]={16,128};
 enc(&mm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,mask.data_ptr(),md,ms,mb,CU_TENSOR_MAP_SWIZZLE_NONE);
 uint64_t wd[2]={64,32},ws[1]={128};uint32_t wb[2]={64,32};
 enc(&wam,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p[2].data_ptr(),wd,ws,wb,CU_TENSOR_MAP_SWIZZLE_128B);
 enc(&wbm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p[3].data_ptr(),wd,ws,wb,CU_TENSOR_MAP_SWIZZLE_128B);
 uint64_t od[2]={uint64_t(S),uint64_t(N)*32},os[1]={uint64_t(S)*2};uint32_t ob[2]={64,32};
 enc(&am,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,a.data_ptr(),od,os,ob,CU_TENSOR_MAP_SWIZZLE_128B);
 enc(&bm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,b.data_ptr(),od,os,ob,CU_TENSOR_MAP_SWIZZLE_128B);
 C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_tma_hopper,cudaFuncAttributeMaxDynamicSharedMemorySize,opm_pro_tma::SMEM));
 cudaKernel_t kernel;C10_CUDA_CHECK(cudaGetKernel(&kernel,opm_prologue_tma_hopper));
 const float* gp=p[0].data_ptr<float>();const float* bp=p[1].data_ptr<float>();
 __nv_bfloat16* mp=reinterpret_cast<__nv_bfloat16*>(mk.data_ptr());float* sp=save_stats?stats.data_ptr<float>():nullptr;unsigned* bitp=reinterpret_cast<unsigned*>(bits.data_ptr());
 int tiles=N*(S/128);float ep=float(eps);
 void* args[]={&xm,&mm,&wam,&wbm,&am,&bm,&gp,&bp,&mp,&sp,&bitp,&S,&N,&tiles,&ep};
 int grid=std::min(tiles,at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
 C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(grid),dim3(512),args,opm_pro_tma::SMEM,at::cuda::getCurrentCUDAStream()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
#ifdef MINIWORLD_OPM_CUBLAS_COUNT
 // Keep the established FP32 cuBLAS mask product; only its cast/clamp are native replacements.
 // Binary inputs make each count exact throughout the supported MSA depths, including TF32 mode.
 auto mf=torch::empty({S,N},fo);
 opm_native_launch(opm_mask_fp32_selective,dim3((S*N+255)/256),dim3(256),mask.data_ptr<bool>(),mf.data_ptr<float>(),S*N);
#ifdef MINIWORLD_OPM_OWNED_BLAS
 opm_owned_count(norm,mf);
#else
 at::mm_out(norm,mf.t(),mf);
#endif
 opm_native_launch(opm_norm_clamp_selective,dim3((N*N+255)/256),dim3(256),norm.data_ptr<float>(),N*N);
#else
 opm_count_launch(bitp,norm.data_ptr<float>(),S,N);
#endif
 return {a,b,stats,mk,norm};
}

std::vector<torch::Tensor> opm_native_prologue(torch::Tensor m,torch::Tensor mask,
    std::vector<torch::Tensor> p,double eps,bool save_stats) {
  TORCH_CHECK(m.is_cuda() && m.is_contiguous() && m.scalar_type()==torch::kBFloat16 && m.dim()==3 && m.size(2)==64,
      "native OPM input must be contiguous BF16 [S,N,64]");
  int S=m.size(0),N=m.size(1);
  TORCH_CHECK(S%256==0 && N%64==0 && mask.is_contiguous() && mask.scalar_type()==torch::kBool
      && mask.device()==m.device() && mask.numel()==S*N && p.size()==8,"native OPM mask/shape contract");
  auto fo=m.options().dtype(torch::kFloat32);
  auto a=torch::empty({N*32,S},m.options()),b=torch::empty_like(a);
  auto mk=torch::empty({S,N},m.options()),norm=torch::empty({N,N},fo);
  auto stats=torch::empty(save_stats?std::vector<int64_t>{S,N,2}:std::vector<int64_t>{0},fo);
  auto bits=torch::empty({N,S/32},m.options().dtype(torch::kInt32));
  auto stream=at::cuda::getCurrentCUDAStream();
  opm_native_launch(opm_prologue_wgmma_native,dim3(N,S/64),dim3(128),
      reinterpret_cast<const __nv_bfloat16*>(m.data_ptr()),mask.data_ptr<bool>(),
      p[0].data_ptr<float>(),p[1].data_ptr<float>(),reinterpret_cast<const __nv_bfloat16*>(p[2].data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(p[3].data_ptr()),reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(b.data_ptr()),reinterpret_cast<__nv_bfloat16*>(mk.data_ptr()),
      save_stats?stats.data_ptr<float>():nullptr,S,N,float(eps));
  opm_native_launch(opm_mask_bits_native,dim3((S*N+255)/256),dim3(256),mask.data_ptr<bool>(),reinterpret_cast<unsigned*>(bits.data_ptr()),S,N);
  opm_count_launch(reinterpret_cast<const unsigned*>(bits.data_ptr()),norm.data_ptr<float>(),S,N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a,b,stats,mk,norm};
}
// The persistent TMA prologue at this build's d_msa (OPM_CM = 64 or 128): LayerNorm + both projections + mask into the grouped
// A2 [(i,c), s] / BT [(j,e), s] layouts, the LayerNorm statistics, the bf16 mask, and the exact mask count [N, N] (popcount of
// the bit mask the prologue wrote).  Parameters arrive already as gamma / beta fp32 [CM] and Wa / Wb bf16 [32, CM].
std::vector<torch::Tensor> opm_prologue_wide(torch::Tensor m, torch::Tensor mask, torch::Tensor gamma, torch::Tensor beta,
                                             torch::Tensor wa, torch::Tensor wb, double eps, bool save_stats) {
  using Cf = opm_pro_wide::Cfg<CM_MSA>;
  TORCH_CHECK(m.is_cuda() && m.is_contiguous() && m.scalar_type() == torch::kBFloat16 && m.dim() == 3 && m.size(2) == CM_MSA,
              "wide OPM prologue: m is contiguous bf16 [S, N, ", CM_MSA, "]");
  const int S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % 256 == 0 && N % 64 == 0 && mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.device() == m.device()
              && mask.numel() == (long)S * N, "wide OPM prologue: S % 256, N % 64, bool mask [S, N]");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32 && gamma.numel() == CM_MSA
              && beta.numel() == CM_MSA && gamma.is_contiguous() && beta.is_contiguous(), "wide OPM prologue: gamma / beta fp32 [CM]");
  TORCH_CHECK(wa.scalar_type() == torch::kBFloat16 && wb.scalar_type() == torch::kBFloat16 && wa.is_contiguous() && wb.is_contiguous()
              && wa.numel() == CH * CM_MSA && wb.numel() == CH * CM_MSA, "wide OPM prologue: Wa / Wb bf16 [32, CM]");
  auto fo = m.options().dtype(torch::kFloat32);
  auto a = torch::empty({(long)N * CH, S}, m.options()), b = torch::empty_like(a), mk = torch::empty({S, N}, m.options());
  auto norm = torch::empty({N, N}, fo);
  auto stats = torch::empty(save_stats ? std::vector<int64_t>{S, N, 2} : std::vector<int64_t>{0}, fo);
  auto bits = torch::empty({N, S / 32}, m.options().dtype(torch::kInt32));
  alignas(64) CUtensorMap xm{}, mm{}, wam{}, wbm{}, am{}, bm{};
  auto enc = [&](CUtensorMap* map, CUtensorMapDataType dt, int rank, void* ptr, uint64_t* dims, uint64_t* strides, uint32_t* box,
                 CUtensorMapSwizzle swz) {
    uint32_t es[3] = {1, 1, 1};
    auto rc = tma_encode()(map, dt, rank, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rc == CUDA_SUCCESS, "OPM wide prologue TMA encode: ", int(rc));
  };
  uint64_t xd[3] = {(uint64_t)CM_MSA, (uint64_t)N, (uint64_t)S}, xs[2] = {(uint64_t)CM_MSA * 2, (uint64_t)N * CM_MSA * 2};
  uint32_t xb[3] = {64, 1, 128};
  enc(&xm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, m.data_ptr(), xd, xs, xb, CU_TENSOR_MAP_SWIZZLE_128B);
  uint64_t md[2] = {(uint64_t)N, (uint64_t)S}, ms[1] = {(uint64_t)N};
  uint32_t mb[2] = {16, 128};
  enc(&mm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, mask.data_ptr(), md, ms, mb, CU_TENSOR_MAP_SWIZZLE_NONE);
  uint64_t wd[2] = {(uint64_t)CM_MSA, 32}, ws[1] = {(uint64_t)CM_MSA * 2};
  uint32_t wbx[2] = {64, 32};
  enc(&wam, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wa.data_ptr(), wd, ws, wbx, CU_TENSOR_MAP_SWIZZLE_128B);
  enc(&wbm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wb.data_ptr(), wd, ws, wbx, CU_TENSOR_MAP_SWIZZLE_128B);
  uint64_t od[2] = {(uint64_t)S, (uint64_t)N * CH}, os[1] = {(uint64_t)S * 2};
  uint32_t obx[2] = {64, 32};
  enc(&am, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.data_ptr(), od, os, obx, CU_TENSOR_MAP_SWIZZLE_128B);
  enc(&bm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, b.data_ptr(), od, os, obx, CU_TENSOR_MAP_SWIZZLE_128B);
  C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_tma_wide<CM_MSA>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM));
  cudaKernel_t kernel;
  C10_CUDA_CHECK(cudaGetKernel(&kernel, opm_prologue_tma_wide<CM_MSA>));
  const float* gp = gamma.data_ptr<float>(); const float* bp = beta.data_ptr<float>();
  __nv_bfloat16* mp = reinterpret_cast<__nv_bfloat16*>(mk.data_ptr());
  float* sp = save_stats ? stats.data_ptr<float>() : nullptr;
  unsigned* bitp = reinterpret_cast<unsigned*>(bits.data_ptr());
  int tiles = N * (S / 128), s_arg = S, n_arg = N; float ep = float(eps);
  void* args[] = {&xm, &mm, &wam, &wbm, &am, &bm, &gp, &bp, &mp, &sp, &bitp, &s_arg, &n_arg, &tiles, &ep};
  int grid = std::min(tiles, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(grid), dim3(512), args, Cf::SMEM,
                                  at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  opm_count_launch(bitp, norm.data_ptr<float>(), S, N);
  return {a, b, stats, mk, norm};
}

torch::Tensor opm_native_cast(torch::Tensor value,torch::Tensor like) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type()==torch::kFloat32,
      "native gradient cast requires contiguous FP32");
  TORCH_CHECK(like.device()==value.device() && like.numel()==value.numel(),"native gradient cast shape/device");
  if(like.scalar_type()==torch::kFloat32)return value;
  TORCH_CHECK(like.scalar_type()==torch::kBFloat16,"native gradient cast destination must be BF16/FP32");
  auto out=torch::empty(value.sizes(),like.options());
  opm_native_launch(opm_cast_native_kernel,dim3((value.numel()+255)/256),dim3(256),
      value.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),value.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> opm_native_cast_grads(std::vector<torch::Tensor> values,std::vector<torch::Tensor> likes) {
  TORCH_CHECK(values.size()==6 && likes.size()==6,"six gradients and parameters required");
  OpmGradCast p{};std::vector<torch::Tensor> out;int count=0;
  for(int k=0;k<6;++k) {
    TORCH_CHECK(values[k].is_contiguous() && values[k].scalar_type()==torch::kFloat32
        && values[k].device()==likes[k].device() && values[k].numel()==likes[k].numel(),"native gradient cast contract");
    TORCH_CHECK(likes[k].scalar_type()==torch::kFloat32 || likes[k].scalar_type()==torch::kBFloat16,"native gradient type");
    out.push_back(torch::empty(likes[k].sizes(),likes[k].options()));
    p.src[k]=values[k].data_ptr<float>();p.dst[k]=out.back().data_ptr();p.sizes[k]=values[k].numel();
    p.fp32[k]=likes[k].scalar_type()==torch::kFloat32;count=std::max(count,p.sizes[k]);
  }
  opm_native_launch(opm_grad_cast_native,dim3((count+255)/256),dim3(256),p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_epilogue", &opm_epilogue, "fused OPM epilogue (div-norm + proj_out + bias + optional residual)",
        py::arg("O"), py::arg("norm"), py::arg("bf"), py::arg("bias"), py::arg("ni"), py::arg("nj"), py::arg("residual") = py::none(), py::arg("after") = 0);
  m.def("opm_dgrad", &opm_dgrad, "fused OPM dO: (dz @ Wo) / n straight into the grouped layout",
        py::arg("dz"), py::arg("norm"), py::arg("bw"), py::arg("ni"), py::arg("nj"), py::arg("after") = 0);
  m.def("opm_dwo", &opm_dwo, "dWo straight off the grouped outer product, no permute");
  m.def("opm_prologue_bwd", &opm_prologue_bwd, "fused OPM prologue backward: mask, both projections and the LayerNorm in one pass",
        py::arg("dA"),py::arg("dB"),py::arg("m"),py::arg("stats"),py::arg("mask"),py::arg("gamma"),py::arg("beta"),
        py::arg("eps"),py::arg("bwp"),py::arg("N"),py::arg("S"),py::arg("native_finish")=false,
        py::arg("raw_weights")=std::vector<torch::Tensor>{});
  m.def("opm_prologue_wide", &opm_prologue_wide, "persistent TMA OPM prologue at this build's d_msa (64 / 128) + exact mask count",
        py::arg("m"), py::arg("mask"), py::arg("gamma"), py::arg("beta"), py::arg("wa"), py::arg("wb"), py::arg("eps"), py::arg("save_stats"));
  m.def("opm_native_pack",&opm_native_pack);
  m.def("opm_native_prologue",&opm_native_prologue);
  m.def("opm_native_prologue_tma",&opm_native_prologue_tma);
  m.def("opm_native_cast",&opm_native_cast);
  m.def("opm_native_cast_grads",&opm_native_cast_grads);
}
