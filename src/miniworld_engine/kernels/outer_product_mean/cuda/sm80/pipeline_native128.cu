#define opm_pipeline_mask80 opm_pipeline_native128_mask80
#define pipeline_mask_launch native128_mask_launch
#define pipeline_mask_result native128_mask_result
#define opm_pipeline_mask_epilogue_pack_kernel opm_native128_mask_epilogue_pack_kernel
// E055 isolated candidate: own CUDA replaces the two L128 gradient GEMMs.
// E053 isolates host overhead: same own-CUDA primitives, allocations in C++, same ATen cuBLAS GEMMs.
// This candidate is explicitly not a completion of the own-CUDA GEMM replacement.
#include "pipeline_ops.cuh"
#include "pipeline_mask_epilogue.cuh"
#include "pipeline_finalize.cuh"
#include <ATen/ops/mm.h>
#include "opm_lt_gemm.cuh"
#include "rect_gemm_core.cuh"

using Tensor=torch::Tensor;
std::vector<Tensor> kernel_leaves(std::vector<Tensor> leaves){
  TORCH_CHECK(leaves.size()==6,"six OPM parameters");
  std::vector<Tensor> out;
  for(int i=0;i<6;++i){TORCH_CHECK(leaves[i].is_cuda()&&leaves[i].is_contiguous(),"CUDA contiguous parameter");
    auto value=leaves[i].detach();
    // The module rounds its output bias to BF16 before using its FP32 value.
    if(i==5)value=value.to(at::kBFloat16).to(at::kFloat);
    else value=value.to((i==0||i==1)?at::kFloat:at::kBFloat16);
    out.push_back(value.contiguous());}
  return out;
}
void check_pipeline_input(Tensor m,Tensor mask,std::vector<Tensor> leaves){
  TORCH_CHECK(m.is_cuda()&&m.is_contiguous()&&m.scalar_type()==at::kBFloat16&&m.dim()==3&&m.size(2)==64,"pipeline m BF16 [S,L,64]");
  TORCH_CHECK(leaves.size()==6&&leaves[4].sizes()==at::IntArrayRef({128,1024}),"pipeline d_pair128");
  TORCH_CHECK(mask.numel()==0||(mask.is_cuda()&&mask.device()==m.device()&&mask.scalar_type()==at::kByte&&mask.is_contiguous()&&mask.sizes()==at::IntArrayRef({m.size(0),m.size(1)})),"mask byte [S,L] or empty");
}
std::vector<Tensor> forward_pipeline(Tensor m,Tensor mask,Tensor residual,std::vector<Tensor> leaves,double eps,bool norm_first,bool training,bool save_o,int64_t pl_variant,int64_t epi_variant,double norm_min,double norm_add){
  check_pipeline_input(m,mask,leaves);at::cuda::CUDAGuard guard(m.device());at::NoGradGuard no_grad;
  auto p=kernel_leaves(leaves);const int64_t S=m.size(0),L=m.size(1);auto f32=m.options().dtype(at::kFloat);
  auto a=torch::empty({S,L*32},m.options()),b=torch::empty({S,L*32},m.options());
  auto stats=training?torch::empty({S,L,2},f32):torch::empty({0},f32);
  prologue(m,mask,p[0],p[1],p[2],p[3],a,b,stats,eps,pl_variant);
  auto o=opm_lt::mm(a,b,true,false);Tensor z,norm;
  if(mask.numel()){
    const int cfg=(S<=1024||L==768)?2:1;
    auto pair=pipeline_mask_result(o,mask,p[4],p[5],residual,norm_first,training,cfg,norm_min,norm_add);z=pair[0];norm=pair[1];
  }else{
    norm=torch::empty({0},f32);z=torch::empty({L,L,128},m.options());
    epilogue(o,norm,double(S),p[4],p[5],residual,z,norm_first,epi_variant,norm_min,norm_add);
  }
  return {z,a,b,stats,training&&save_o?o:torch::empty({0},m.options()),norm};
}
std::vector<Tensor> backward_pipeline(Tensor m,Tensor mask,std::vector<Tensor> leaves,Tensor a,Tensor b,Tensor stats,Tensor saved_o,Tensor norm,Tensor dz,bool norm_first,int64_t dg_variant,int64_t pb_variant,double norm_min,double norm_add){
  check_pipeline_input(m,mask,leaves);at::cuda::CUDAGuard guard(m.device());at::NoGradGuard no_grad;
  auto p=kernel_leaves(leaves);const int64_t S=m.size(0),L=m.size(1),CM=64,CZ=128;auto f32=m.options().dtype(at::kFloat);
  auto dO=torch::empty({L*32,L*32},m.options()),dzn=torch::empty({L,L,CZ},m.options());
  const int64_t parts_rows=((L+31)/32)*((L+3)/4);
  auto dbo_parts=torch::empty({parts_rows,CZ},f32),dbo=torch::empty({CZ},f32);
  dgrad(dz,norm,norm.numel()?1.0:double(S),p[4],dO,dzn,dbo_parts,norm_first,dg_variant,norm_min,norm_add);
  reduce_rows(dbo_parts,dbo);
  auto da=torch::empty({S,L*32},m.options()),db=torch::empty_like(da);
  TORCH_CHECK(S==1024 && L==128,"E055 validated candidate S1024L128 only");
  // the two gradient GEMMs: cuBLASLt with the searched algorithm (opm_lt_gemm.cuh, 1.3x the default heuristic at L128, 6 % faster than the own rectangular GEMM of E055 over the step);
  // the own GEMM stays behind OPM_NATIVE128_OWN_GEMM (compile-time)
#ifdef OPM_NATIVE128_OWN_GEMM
  opm_rect80::run(b,dO,da,0,16,0);
  opm_rect80::run(a,dO,db,2,12,0);dO=Tensor();
#else
  da=opm_lt::mm(b,dO,false,true);db=opm_lt::mm(a,dO,false,false);dO=Tensor();
#endif
  auto o=saved_o.numel()?saved_o:opm_lt::mm(a,b,true,false);
  const int64_t i_per=(L+23)/24,splits=(L+i_per-1)/i_per;
  auto dwo_parts=torch::empty({splits,CZ,1024},f32),dwout=torch::empty({CZ,1024},leaves[4].options());
  dwo(dzn,o,dwo_parts,i_per);reduce_rows(dwo_parts.view({splits,CZ*1024}),dwout.view({CZ*1024}));
  dzn=Tensor();o=Tensor();dwo_parts=Tensor();dbo_parts=Tensor();
  const int64_t ntile=((S+63)/64)*((L+1)/2),nsm=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t nblk=std::min(ntile,nsm*(pb_variant==1?1:2));
  auto dm=torch::empty_like(m),pb_parts=torch::empty({nblk,66*CM},f32),total=torch::empty({66*CM},f32);
  prologue_bwd(da,db,m,stats,mask,p[0],p[1],p[2],p[3],dm,pb_parts,pb_variant);reduce_rows(pb_parts,total);
  auto wl=total.slice(0,0,32*CM).view({32,CM}),wr=total.slice(0,32*CM,64*CM).view({32,CM});
  auto gamma=total.slice(0,64*CM,65*CM),beta=total.slice(0,65*CM,66*CM);
  auto typed=result(std::vector<Tensor>{gamma,beta,wl,wr,dbo},std::vector<Tensor>{leaves[0],leaves[1],leaves[2],leaves[3],leaves[5]});
  return {dm,typed[0],typed[1],typed[2],typed[3],dwout,typed[4]};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward_pipeline);m.def("backward",&backward_pipeline);}
