// Pre-affine xhat: one global load per element; no raw tri input.
static_assert(!PRODUCER && LOWREG && GATE_PHASE, "selected shared schedule only");
TMN_DEVI float xhat_at(uint8_t* sm,int slot,int row,int c){
#if XHAT_FP32
 uint8_t* p=row<32?sm+32768+slot*49152:sm+114688;
 return *reinterpret_cast<float*>(p+swz128(c,(row%32)*4));
#else
 return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(sm+32768+slot*49152+swz128(c,row*2)));
#endif
}
