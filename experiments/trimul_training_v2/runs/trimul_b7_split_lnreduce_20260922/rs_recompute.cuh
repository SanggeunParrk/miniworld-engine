// SPDX-License-Identifier: Apache-2.0
#pragma once
template <int OFF_BYTES>
TMN_DEVI void recompute_rs_off(float (&d)[32], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, int scale_d) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    ".reg .b32 lo;\n"
    ".reg .b64 dsc;\n"
    "setp.ne.b32 p, %38, 0;\n"
    "add.u32 lo, %36, %39;\n"
    "mov.b64 dsc, {lo, %37};\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},"
    "{%32, %33, %34, %35}, dsc, p, 1, 1, 1;\n"
    "}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(desc_lo), "r"(desc_hi), "r"(scale_d), "n"(OFF_BYTES >> 4));
}

TMN_DEVI void gemm_recompute_regs(float (&acc)[32],uint32_t (&f)[8][4],uint8_t* wb){
 #pragma unroll
 for(int j=0;j<32;++j)acc[j]=0.f;
 uint64_t desc=smem_desc(smem_u32(wb),8192,1024,1);uint32_t lo=uint32_t(desc),hi=uint32_t(desc>>32);
 fence_regs(acc);wgmma_fence();
 static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;recompute_rs_off<(k/4)*8192+(k%4)*2048>(acc,f[k],lo,hi,k>0);});
 wgmma_commit();wgmma_wait<0>();fence_regs(acc);
}
