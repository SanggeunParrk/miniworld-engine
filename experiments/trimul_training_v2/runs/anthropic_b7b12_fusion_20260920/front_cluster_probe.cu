#define DIRECT_FOUR_WEIGHTS 1
#include "front_primitives.cuh"
#include <cooperative_groups.h>
struct Params{
 CUtensorMap dl,dr,pre,xn,dg,wlg,wl,wrg,wr,wgate,x,res,dx;
 const __nv_bfloat16* mask;
 const float *mean,*rs,*gamma;
 __nv_bfloat16* dw;
 float *dgam,*dbeta,*partw,*partln;
 unsigned int* counts;
 __nv_bfloat16 *debugdc,*debugxn;
 int M,L,tiles;
};
extern "C" __global__ __cluster_dims__(8,1,1) __launch_bounds__(256,1) void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) unsigned char sm[];auto cluster=cooperative_groups::this_cluster();unsigned* data=(unsigned*)sm;data[threadIdx.x]=blockIdx.x*256+threadIdx.x;cluster.sync();unsigned* other=cluster.map_shared_rank(data,cluster.block_rank()^1);unsigned got=other[threadIdx.x];p.partln[blockIdx.x*256+threadIdx.x]=float(got);cluster.sync();
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){}
