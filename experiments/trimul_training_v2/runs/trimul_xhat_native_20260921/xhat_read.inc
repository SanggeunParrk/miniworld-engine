// Native FP32 normalized-value layout. No tri / mean input.
static_assert(XHAT_FP32==1 && !PRODUCER && LOWREG && GATE_PHASE,"native saved-xhat schedule only");
TMN_DEVI void xhat_frag(float (&h)[8],uint8_t* sm,int slot,int w,int lane,int ks){
 uint8_t* p=ks<12?sm+16384+slot*49152:sm;int local=ks<12?ks:ks-12;
 uint32_t offset=local*4096+w*1024+lane*16;
 uint4 a=lds128(smem_u32(p)+offset),b=lds128(smem_u32(p)+offset+512);
 h[0]=__uint_as_float(a.x);h[1]=__uint_as_float(a.y);h[2]=__uint_as_float(a.z);h[3]=__uint_as_float(a.w);
 h[4]=__uint_as_float(b.x);h[5]=__uint_as_float(b.y);h[6]=__uint_as_float(b.z);h[7]=__uint_as_float(b.w);
}
