// Pre-affine xhat: one global load per element; no raw tri input.
static_assert(!PRODUCER && LOWREG && GATE_PHASE, "selected shared schedule only");
TMN_DEVI float xhat_at(uint8_t* sm,int slot,int row,int c){
#if XHAT_FP32 == 1
 int q=row/16;uint8_t* p=q<3?sm+16384+slot*49152+q*16384:sm;
 return *reinterpret_cast<float*>(p+c*64+(((row%16)*4)^(((c>>1)&3)*16)));
#elif XHAT_FP32 >= 2
 float a=__half2float(*reinterpret_cast<__half*>(sm+32768+slot*49152+swz128(c,row*2)));
#if XHAT_FP32 == 3
 float* mu=reinterpret_cast<float*>(sm+225280);float rs=mu[64+row];
 float raw=math::round_bf16(__fadd_rn(__fdiv_rn(a,rs),mu[row]));
 return __fmul_rn(__fsub_rn(raw,mu[row]),rs);
#else
 return a;
#endif
#else
 return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(sm+32768+slot*49152+swz128(c,row*2)));
#endif
}
