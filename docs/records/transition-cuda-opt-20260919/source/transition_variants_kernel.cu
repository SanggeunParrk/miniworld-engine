// Transition saved-xn kernels: explicit SM90 TMA + WGMMA, two K schedules.
// Same fused mathematics for both schedules; no expanded activation in fwd HBM.
// Shared IO is explicit PTX: generic CuTe-pointer loads/stores otherwise produce
// 64-bit generic addressing, inhibit BF16 pair packing and spill wide forwards.
// Backward publishes h/dA/dB with TMA stores after the proxy fence and CTA barrier.
// Store completion protects the shared tiles before reuse; rounding is unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

namespace transition_variants {
using namespace cute;
using BF = cutlass::bfloat16_t;
struct NoOutputMap {};
constexpr int D=MWV_D, H=4*D, BK=MWV_BK, BN=MWV_BN, BO=MWV_BO;
constexpr int MG=MWV_MGROUPS, NG=MWV_NGROUPS, S=MWV_STAGES, BM=64*MG;
constexpr bool FULL=MWV_FULL;
constexpr int NK=(D+BK-1)/BK;
using LX=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},make_shape(Int<BM>{},Int<BK>{})));
using LW=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},make_shape(Int<BN>{},Int<BK>{})));
using HAtom=std::conditional_t<(BN>=64),GMMA::Layout_K_SW128_Atom<BF>,GMMA::Layout_K_SW64_Atom<BF>>;
using LH=decltype(tile_to_shape(HAtom{},make_shape(Int<BM>{},Int<BN>{})));
template<int N> using LWS=decltype(tile_to_shape(HAtom{},make_shape(Int<N>{},Int<BN>{})));
using Pipe=cutlass::PipelineTmaAsync<S>;
using Single=cutlass::PipelineTmaAsync<1>;

__device__ __forceinline__ uint32_t shared_address(const void* p){return static_cast<uint32_t>(__cvta_generic_to_shared(p));}
__device__ __forceinline__ void shared_store_pair(void* p,uint32_t v){asm volatile("st.shared.u32 [%0], %1;" :: "r"(shared_address(p)),"r"(v):"memory");}
__device__ __forceinline__ uint32_t shared_load_pair(const void* p){uint32_t v;asm volatile("ld.shared.u32 %0, [%1];":"=r"(v):"r"(shared_address(p)):"memory");return v;}
__device__ __forceinline__ uint4 shared_load_vector(const void* p){uint4 v;asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(shared_address(p)):"memory");return v;}
union PackedPair { uint32_t u; BF b[2]; };

constexpr int align16(int n){return (n+15)/16*16;}

template<bool BWD> struct Storage {
    static constexpr int GROUPS=BWD?1:NG;
    static constexpr int ON=((D+GROUPS*BO-1)/(GROUPS*BO))*BO;
    static constexpr int PAD=GROUPS*ON;
    static constexpr int XE=cosize_v<LX>;
    static constexpr int WE=cosize_v<LW>;
    static constexpr bool REUSE_H=!BWD && GROUPS>1 && BK>=BM;
    static constexpr int HE=(BWD?3:((GROUPS==1 || REUSE_H)?0:1))*cosize_v<LH>;
    static constexpr int SE=BWD?cosize_v<LH>:cosize_v<LWS<PAD>>;
    static constexpr int DATA=2*((FULL?1:S)*XE+2*S*WE+HE+SE);
    static constexpr int SHUFFLE=BWD?0:2*BM*PAD;
    static constexpr int OFF=align16(DATA>SHUFFLE?DATA:SHUFFLE);
    static constexpr int BYTES=OFF+align16(sizeof(typename Pipe::SharedStorage))+2*align16(sizeof(typename Single::SharedStorage));
};

template <class MMA, class Layout0>
CUTE_HOST_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    using Traits = MMA_Traits<MMA>;

    static_assert(decltype(rank<0>(acc_layout))::value == 3, "expected SM90 GMMA accumulator layout");
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    static_assert(sizeof(typename Traits::ValTypeA) == 2, "this transform is for FP16/BF16 RS A");

    auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});  // ((2, N / 16))
    return make_layout(
        make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
        get<1>(acc_layout),
        coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout)))
    );
}

template<class P> __device__ auto params(int bytes,int threads,int tid){
    typename P::Params p;
    p.transaction_bytes=bytes;
    p.role=P::ThreadCategory::ProducerConsumer;
    p.is_leader=tid==0;
    p.num_consumers=threads;
    return p;
}

template<bool BWD,class TX,class TA,class TB,class TS,class TO,class TD>
__global__ __launch_bounds__(128*MG*(BWD?1:NG),MWV_MIN_BLOCKS)
void variant_kernel(const BF* residual,const BF* dh,BF* out,BF* dab,int64_t M,
                    __grid_constant__ const TX tx,__grid_constant__ const TA ta,
                    __grid_constant__ const TB tb,__grid_constant__ const TS ts,
                    __grid_constant__ const TO tout,__grid_constant__ const TD tdab){
    using ST=Storage<BWD>;
    constexpr int G=ST::GROUPS,NT=128*MG*G;
    int tid=threadIdx.x,wg=tid/128,lane=tid%128,mg=wg/G,ng=wg%G;
    int64_t row0=static_cast<int64_t>(blockIdx.x)*BM;
    extern __shared__ __align__(128) unsigned char sm[];
    BF* px=reinterpret_cast<BF*>(sm);
    BF* pa=px+(FULL?1:S)*ST::XE;
    BF* pb=pa+S*ST::WE;
    BF* ph=pb+S*ST::WE;
    BF* ps=ph+ST::HE;
    auto& ep=*reinterpret_cast<typename Pipe::SharedStorage*>(sm+ST::OFF);
    auto& xp=*reinterpret_cast<typename Single::SharedStorage*>(sm+ST::OFF+align16(sizeof(ep)));
    auto& sp=*reinterpret_cast<typename Single::SharedStorage*>(sm+ST::OFF+align16(sizeof(ep))+align16(sizeof(xp)));
    Pipe pipe(ep,params<Pipe>(2*(2*ST::WE+(FULL?0:ST::XE)),NT,tid),Shape<_1,_1,_1>{});
    Single xpipe(xp,params<Single>(2*ST::XE,NT,tid),Shape<_1,_1,_1>{});
    Single wpipe(sp,params<Single>(2*ST::SE,NT,tid),Shape<_1,_1,_1>{});
    __syncthreads();
    auto gx=tx.get_tma_tensor(make_shape(M,Int<D>{}));
    auto ga=ta.get_tma_tensor(make_shape(Int<H>{},Int<D>{}));
    auto gb=tb.get_tma_tensor(make_shape(Int<H>{},Int<D>{}));
    auto gs=[&](){
        if constexpr(BWD)return ts.get_tma_tensor(make_shape(M,Int<H>{}));
        else return ts.get_tma_tensor(make_shape(Int<D>{},Int<H>{}));
    }();
    auto sx=[&](int s){return make_tensor(make_smem_ptr(px+(FULL?0:s)*ST::XE),LX{});};
    auto sa=[&](int s){return make_tensor(make_smem_ptr(pa+s*ST::WE),LW{});};
    auto sb=[&](int s){return make_tensor(make_smem_ptr(pb+s*ST::WE),LW{});};
    auto sha=make_tensor(make_smem_ptr(ph+cosize_v<LH>),LH{});
    auto shb=make_tensor(make_smem_ptr(ph+2*cosize_v<LH>),LH{});
    auto sw=make_tensor(make_smem_ptr(ps),LWS<ST::PAD>{});
    auto sd=make_tensor(make_smem_ptr(ps),LH{});
    auto producer=cutlass::make_producer_start_state<Pipe>();
    typename Pipe::PipelineState consumer;
    auto xproducer=cutlass::make_producer_start_state<Single>();
    typename Single::PipelineState xconsumer;
    auto wproducer=cutlass::make_producer_start_state<Single>();
    typename Single::PipelineState wconsumer;
    if constexpr(FULL){
        if(tid==0){
            xpipe.producer_acquire(xproducer);
            auto bar=xpipe.producer_get_barrier(xproducer);
            auto tile=local_tile(gx,make_shape(Int<BM>{},Int<BK>{}),make_coord(blockIdx.x,0));
            auto sl=tx.get_slice(_0{});
            copy(tx.with(*bar),sl.partition_S(tile),sl.partition_D(sx(0)));
        }
        xpipe.consumer_wait(xconsumer);
    }
    auto issue=[&](int seq){
        if(tid==0){
            int st=seq%S,hn=seq/NK,kn=seq%NK;
            pipe.producer_acquire(producer);
            auto bar=pipe.producer_get_barrier(producer);
            auto at=local_tile(ga,make_shape(Int<BN>{},Int<BK>{}),make_coord(hn,kn));
            auto bt=local_tile(gb,make_shape(Int<BN>{},Int<BK>{}),make_coord(hn,kn));
            auto as=ta.get_slice(_0{});auto bs=tb.get_slice(_0{});
            copy(ta.with(*bar),as.partition_S(at),as.partition_D(sa(st)));
            copy(tb.with(*bar),bs.partition_S(bt),bs.partition_D(sb(st)));
            if constexpr(!FULL){
                auto xt=local_tile(gx,make_shape(Int<BM>{},Int<BK>{}),make_coord(blockIdx.x,kn));
                auto xs=tx.get_slice(_0{});
                copy(tx.with(*bar),xs.partition_S(xt),xs.partition_D(sx(st)));
            }
            ++producer;
        }
    };
    constexpr int TOTAL=(H/BN)*NK;
    for(int q=0;q<S-1 && q<TOTAL;++q)issue(q);
    using Expand=std::conditional_t<BN==32,SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>,
                 std::conditional_t<BN==64,SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>,
                                           SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>>>;
    using SqueezeSS=std::conditional_t<BO==64,SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>,
                                            SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>>;
    using SqueezeRS=std::conditional_t<BO==64,SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>,
                                            SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>>;
    using Squeeze=std::conditional_t<!BWD && G==1,SqueezeRS,SqueezeSS>;
    auto me=make_tiled_mma(Expand{});auto te=me.get_slice(lane);
    auto ms=make_tiled_mma(Squeeze{});auto tt=ms.get_slice(lane);
    auto cout=partition_fragment_C(ms,make_shape(_64{},Int<BWD?BO:ST::ON>{}));
    clear(cout);
    auto ce=te.partition_C(make_identity_tensor(make_shape(_64{},Int<BN>{})));
    for(int hn=0;hn<H/BN;++hn){
        // The last consumed A stage is dead until the next hidden tile. Reuse
        // it for H when it fits, rather than keeping another shared allocation.
        BF* hp=ph;
        if constexpr(ST::REUSE_H)hp=pa+((hn*NK+NK-1)%S)*ST::WE;
        auto sh=make_tensor(make_smem_ptr(hp),LH{});
        {
            if(tid==0){
                wpipe.producer_acquire(wproducer);
                auto bar=wpipe.producer_get_barrier(wproducer);
                auto sl=ts.get_slice(_0{});
                if constexpr(BWD){
                    auto tile=local_tile(gs,make_shape(Int<BM>{},Int<BN>{}),make_coord(blockIdx.x,hn));
                    copy(ts.with(*bar),sl.partition_S(tile),sl.partition_D(sd));
                }else{
                // TMA box dimensions are <=256. Keep the shared destination
                // whole, but describe each output segment independently.
                CUTE_UNROLL
                for(int part=0;part<ST::PAD/BO;++part){
                    auto tile=local_tile(gs,make_shape(Int<BO>{},Int<BN>{}),make_coord(part,hn));
                    auto dest=local_tile(sw,make_shape(Int<BO>{},Int<BN>{}),make_coord(part,0));
                    copy(ts.with(*bar),sl.partition_S(tile),sl.partition_D(dest));
                }
                }
                ++wproducer;
            }
        }
        auto aa=partition_fragment_C(me,make_shape(_64{},Int<BN>{}));
        auto bb=make_fragment_like(aa);clear(aa);clear(bb);
        for(int kn=0;kn<NK;++kn){
            int seq=hn*NK+kn,st=seq%S;
            if(seq+S-1<TOTAL)issue(seq+S-1);
            pipe.consumer_wait(consumer);
            if(ng==0){
                auto xt=local_tile(sx(st),make_shape(_64{},Int<BK>{}),make_coord(mg,0));
                auto af=te.make_fragment_A(te.partition_A(xt));
                auto aw=te.make_fragment_B(te.partition_B(sa(st)));
                auto bw=te.make_fragment_B(te.partition_B(sb(st)));
                warpgroup_fence_operand(aa);warpgroup_fence_operand(bb);
                warpgroup_arrive();gemm(me,af,aw,aa);gemm(me,af,bw,bb);
                warpgroup_commit_batch();warpgroup_wait<0>();
                warpgroup_fence_operand(aa);warpgroup_fence_operand(bb);
            }
            if constexpr(ST::REUSE_H) { if(kn==NK-1) __syncthreads(); }
            pipe.consumer_release(consumer);++consumer;
        }
        if constexpr(BWD)wpipe.consumer_wait(wconsumer);
        auto hreg=make_fragment_like<BF>(aa);
        if(ng==0){
            CUTE_UNROLL
            for(int i=0;i<size(aa);i+=2){
                int mr=get<0>(ce(i)),nc=get<1>(ce(i));
                int64_t row=row0+64*mg+mr;
                PackedPair hv,da,db,grad;
                if constexpr(BWD)grad.u=shared_load_pair(&sd(64*mg+mr,nc));
                CUTE_UNROLL
                for(int j=0;j<2;++j){
                    float av=aa(i+j),bv=bb(i+j),sig=1.f/(1.f+__expf(-av)),silu=av*sig;
                    hv.b[j]=BF(silu*bv);
                    if constexpr(BWD){
                        float gv=float(grad.b[j]);
                        da.b[j]=BF(gv*bv*(sig+av*sig*(1.f-sig)));
                        db.b[j]=BF(gv*silu);
                    }else if constexpr(G==1)hreg(i+j)=hv.b[j];
                }
                if constexpr(BWD){
                    if(row<M){
                        shared_store_pair(&sh(64*mg+mr,nc),hv.u);
                        shared_store_pair(&sha(64*mg+mr,nc),da.u);
                        shared_store_pair(&shb(64*mg+mr,nc),db.u);
                    }
                }else if constexpr(G>1)shared_store_pair(&sh(64*mg+mr,nc),hv.u);
            }
        }
        if constexpr(BWD){
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if(tid==0){
                auto gout=tout.get_tma_tensor(make_shape(M,Int<H>{}));
                auto gdab=tdab.get_tma_tensor(make_shape(M,Int<2*H>{}));
                auto so=tout.get_slice(_0{});auto sdab=tdab.get_slice(_0{});
                auto oh=local_tile(gout,make_shape(Int<BM>{},Int<BN>{}),make_coord(blockIdx.x,hn));
                auto oa=local_tile(gdab,make_shape(Int<BM>{},Int<BN>{}),make_coord(blockIdx.x,hn));
                auto ob=local_tile(gdab,make_shape(Int<BM>{},Int<BN>{}),make_coord(blockIdx.x,H/BN+hn));
                copy(tout,so.partition_S(sh),so.partition_D(oh));
                copy(tdab,sdab.partition_S(sha),sdab.partition_D(oa));
                copy(tdab,sdab.partition_S(shb),sdab.partition_D(ob));
                tma_store_arrive();tma_store_wait<0>();
            }
            __syncthreads(); // TMA no longer reads any of the three shared tiles
            wpipe.consumer_release(wconsumer);++wconsumer;
        }
        if constexpr(!BWD){
            wpipe.consumer_wait(wconsumer);
            auto wt=local_tile(sw,make_shape(Int<ST::ON>{},Int<BN>{}),make_coord(ng,0));
            auto wb=tt.make_fragment_B(tt.partition_B(wt));
            warpgroup_fence_operand(cout);
            if constexpr(G==1){
                // One output group owns the expand fragment: feed registers
                // directly to RS WGMMA using the legacy b2b layout transform.
                auto ha=make_tensor(hreg.data(),convert_layout_acc_Aregs<Squeeze>(aa.layout()));
                warpgroup_arrive();gemm(ms,ha,wb,cout);
            }else{
                // Multiple output groups share H; publish once, then consume SS.
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                auto ht=local_tile(sh,make_shape(_64{},Int<BN>{}),make_coord(mg,0));
                auto ha=tt.make_fragment_A(tt.partition_A(ht));
                warpgroup_arrive();gemm(ms,ha,wb,cout);
            }
            warpgroup_commit_batch();warpgroup_wait<0>();
            warpgroup_fence_operand(cout);
            if constexpr(G>1) __syncthreads(); // shared H / reused A stage lifetime
            wpipe.consumer_release(wconsumer);++wconsumer;
        }
    }
    if constexpr(!BWD){
        __syncthreads();
        auto lo=tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},make_shape(Int<BM>{},Int<ST::PAD>{}));
        auto so=make_tensor(make_smem_ptr(px),lo);
        auto co=tt.partition_C(make_identity_tensor(make_shape(_64{},Int<ST::ON>{})));
        CUTE_UNROLL
        for(int i=0;i<size(cout);i+=2){
            int mr=64*mg+get<0>(co(i)),col=ng*ST::ON+get<1>(co(i));
            PackedPair v;v.b[0]=BF(cout(i));v.b[1]=BF(cout(i+1));
            shared_store_pair(&so(mr,col),v.u);
        }
        __syncthreads();
        for(int vec=tid;vec<BM*D/8;vec+=NT){
            int mr=vec/(D/8),col=(vec%(D/8))*8;int64_t row=row0+mr;
            if(row<M){
                uint4 value=shared_load_vector(&so(mr,col));
                uint4 identity=*reinterpret_cast<const uint4*>(residual+row*D+col);
                auto* a=reinterpret_cast<__nv_bfloat162*>(&value);
                auto* b=reinterpret_cast<__nv_bfloat162*>(&identity);
                #pragma unroll
                for(int q=0;q<4;++q)a[q]=__hadd2(a[q],b[q]);
                *reinterpret_cast<uint4*>(out+row*D+col)=value;
            }
        }
    }
}

void check(const torch::Tensor& x,const torch::Tensor& wa,const torch::Tensor& wb){
    TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.size(1)==D && x.is_contiguous() && x.scalar_type()==at::kBFloat16,"xn must be contiguous CUDA BF16 [M,D]");
    for(const auto& w:{wa,wb})TORCH_CHECK(w.device()==x.device() && w.scalar_type()==at::kBFloat16 && w.is_contiguous() && w.dim()==2 && w.size(0)==H && w.size(1)==D,"expand weights must be BF16 [4D,D] on xn device");
    auto prop=at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major==9 && prop->minor==0,"SM90 required");
}

template<bool BWD> std::vector<torch::Tensor> run(torch::Tensor xn,torch::Tensor residual,torch::Tensor wa,torch::Tensor wb,torch::Tensor ws,torch::Tensor dh){
    check(xn,wa,wb);c10::cuda::CUDAGuard guard(xn.device());
    using ST=Storage<BWD>;
    int64_t M=xn.size(0);
    TORCH_CHECK(ST::BYTES<=232448,"variant configuration exceeds shared memory: ",ST::BYTES);
    if constexpr(BWD){TORCH_CHECK(dh.device()==xn.device() && dh.is_contiguous() && dh.scalar_type()==at::kBFloat16 && dh.dim()==2 && dh.size(0)==M && dh.size(1)==H,"dh must be BF16 [M,4D]");}
    else{
        TORCH_CHECK(residual.device()==xn.device() && residual.sizes()==xn.sizes() && residual.is_contiguous() && residual.scalar_type()==at::kBFloat16,"residual must match xn");
        TORCH_CHECK(ws.device()==xn.device() && ws.is_contiguous() && ws.scalar_type()==at::kBFloat16 && ws.dim()==2 && ws.size(0)==D && ws.size(1)==H,"Ws must be BF16 [D,4D]");
    }
    auto out=torch::empty({M,BWD?H:D},xn.options());
    auto dab=torch::empty({M,BWD?2*H:0},xn.options());
    if(M==0)return {out,dab};
    auto x=make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(xn.data_ptr())),make_shape(M,Int<D>{}),make_stride(Int<D>{},_1{}));
    auto a=make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wa.data_ptr())),make_shape(Int<H>{},Int<D>{}),make_stride(Int<D>{},_1{}));
    auto b=make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wb.data_ptr())),make_shape(Int<H>{},Int<D>{}),make_stride(Int<D>{},_1{}));
    auto s=[&](){
        if constexpr(BWD)return make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(dh.data_ptr())),make_shape(M,Int<H>{}),make_stride(Int<H>{},_1{}));
        else return make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(ws.data_ptr())),make_shape(Int<D>{},Int<H>{}),make_stride(Int<H>{},_1{}));
    }();
    auto tx=make_tma_copy(SM90_TMA_LOAD{},x,LX{},make_shape(Int<BM>{},Int<BK>{}),_1{});
    auto ta=make_tma_copy(SM90_TMA_LOAD{},a,LW{},make_shape(Int<BN>{},Int<BK>{}),_1{});
    auto tb=make_tma_copy(SM90_TMA_LOAD{},b,LW{},make_shape(Int<BN>{},Int<BK>{}),_1{});
    auto ts=[&](){
        if constexpr(BWD)return make_tma_copy(SM90_TMA_LOAD{},s,LH{},make_shape(Int<BM>{},Int<BN>{}),_1{});
        else return make_tma_copy(SM90_TMA_LOAD{},s,LWS<BO>{},make_shape(Int<BO>{},Int<BN>{}),_1{});
    }();
    // Forward does not pass unused output tensor maps to the kernel.

    auto to=[&](){
        if constexpr(BWD){
            auto ot=make_tensor(make_gmem_ptr(reinterpret_cast<BF*>(out.data_ptr())),make_shape(M,Int<H>{}),make_stride(Int<H>{},_1{}));
            return make_tma_copy(SM90_TMA_STORE{},ot,LH{},make_shape(Int<BM>{},Int<BN>{}),_1{});
        }else return NoOutputMap{};
    }();
    auto td=[&](){
        if constexpr(BWD){
            auto dt=make_tensor(make_gmem_ptr(reinterpret_cast<BF*>(dab.data_ptr())),make_shape(M,Int<2*H>{}),make_stride(Int<2*H>{},_1{}));
            return make_tma_copy(SM90_TMA_STORE{},dt,LH{},make_shape(Int<BM>{},Int<BN>{}),_1{});
        }else return NoOutputMap{};
    }();
    auto kernel=variant_kernel<BWD,decltype(tx),decltype(ta),decltype(tb),decltype(ts),decltype(to),decltype(td)>;
    auto err=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ST::BYTES);
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
    kernel<<<(M+BM-1)/BM,128*MG*(BWD?1:NG),ST::BYTES,at::cuda::getCurrentCUDAStream()>>>(
        BWD?nullptr:reinterpret_cast<const BF*>(residual.data_ptr()),BWD?reinterpret_cast<const BF*>(dh.data_ptr()):nullptr,
        reinterpret_cast<BF*>(out.data_ptr()),reinterpret_cast<BF*>(dab.data_ptr()),M,tx,ta,tb,ts,to,td);
    err=cudaGetLastError();TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
    return {out,dab};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    using namespace transition_variants;
    m.def("forward",[](torch::Tensor xn,torch::Tensor r,torch::Tensor wa,torch::Tensor wb,torch::Tensor ws){return run<false>(xn,r,wa,wb,ws,torch::Tensor())[0];});
    m.def("gate_backward",[](torch::Tensor xn,torch::Tensor wa,torch::Tensor wb,torch::Tensor dh){return run<true>(xn,torch::Tensor(),wa,wb,torch::Tensor(),dh);});
    m.def("resources",[](){pybind11::dict r;r["forward_smem"]=Storage<false>::BYTES;r["backward_smem"]=Storage<true>::BYTES;return r;});
}
