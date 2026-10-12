// Included inside Hopper K1's epilogue after all preactivation MMAs retire.
// Recompute sigmoid/projection into a token-major SW128 staging tile.
// Drain the previous async shared read before reusing this warpgroup's tile.
if (st_elect) tma_store_wait_read<0>();
fence_proxy_async();
named_bar_sync(bar_id, 128);
// The same 8 KB stage first receives [32 channels][64 tokens] dplanes.
// Every lane reads it before the whole warpgroup transposes into dcat.
if (st_elect) {
  mbar_arrive_expect_tx(barD_full + cw, 4096);
  tma_load_2d(sStage + cw * 8192, &p.tm_dpl, barD_full + cw,
              iw * p.N + jw, b * 32);
}
mbar_wait(barD_full + cw, (t_local * NBLK + b) & 1);
// Invert K1's plane stmatrix.trans layout directly into accumulator pairs.
uint32_t da[2][4];
ldsm_x4_t(da[0], stage_u + sts_off0);
ldsm_x4_t(da[1], stage_u + sts_off1);
uint32_t pk[4][4];
#pragma unroll
for (int q = 0; q < 4; ++q) {
  auto read = [&](bool valid, float mask, int r, int out) {
    const float s0 = math::sigmoid(ac[4*q+r]), s1 = math::sigmoid(ac[4*q+r+1]);
    const float v0 = ac[4*(q+4)+r], v1 = ac[4*(q+4)+r+1];
    const uint32_t dw = da[q / 2][2 * (q & 1) + r / 2];
    const float d0 = valid ? bf16lo(dw) * mask : 0.f;
    const float d1 = valid ? bf16hi(dw) * mask : 0.f;
    pk[q][out] = pack_bf16(d0*s0*v0*(1.f-s0), d1*s1*v1*(1.f-s1));
    pk[q][out+1] = pack_bf16(d0*s0, d1*s1);
  };
  read(vA, mA, 0, 0);
  read(vB, mB, 2, 2);
}
// WAR: the channel-major input and token-major output share the stage.
named_bar_sync(bar_id, 128);
const int tr = 16 * wiw + 8 * (idx & 1) + rr;
const int cb = 8 * (idx >> 1);
stsm_x4(stage_u + swz128(tr, 2 * cb), pk[0][0], pk[0][2], pk[1][0], pk[1][2]);
stsm_x4(stage_u + swz128(tr, 2 * (cb + 16)), pk[2][0], pk[2][2], pk[3][0], pk[3][2]);
stsm_x4(stage_u + swz128(tr, 2 * (cb + 32)), pk[0][1], pk[0][3], pk[1][1], pk[1][3]);
stsm_x4(stage_u + swz128(tr, 2 * (cb + 48)), pk[2][1], pk[2][3], pk[3][1], pk[3][3]);
fence_proxy_async();
named_bar_sync(bar_id, 128);
if (st_elect) {
  const int col = b * 64, tok = iw * p.N + jw;
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(&p.tm_dcat), "r"(stage_u), "r"(col), "r"(tok) : "memory");
  tma_store_commit();
}
