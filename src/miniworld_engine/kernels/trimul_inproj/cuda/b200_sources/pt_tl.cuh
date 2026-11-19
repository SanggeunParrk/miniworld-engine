// pt_tl.cuh -- optional per-CTA phase trace of the B200 TriMul training kernels (build with -DPT_TL=1; probe builds only).
// PT_BEGIN at kernel entry (every thread keeps its own entry clock); PTM(arr, slot) stores clock64 - entry clock into
// arr[blockIdx.x][slot] (slots 1..14); PT_G(arr, slot) stores %globaltimer (ns; slot 0 at entry, 15 at exit: cross-CTA alignment and the
// clock rate). Without PT_TL every macro is empty.
#pragma once
#if defined(PT_TL) && PT_TL
#define PT_BEGIN const long long ptc0_ = clock64();
#define PTM(arr, slot) { arr[blockIdx.x][slot] = (unsigned long long)(clock64() - ptc0_); }
#define PT_G(arr, slot) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); arr[blockIdx.x][slot] = t_; }
#else
#define PT_BEGIN
#define PTM(arr, slot) ((void)0)
#define PT_G(arr, slot) ((void)0)
#endif

// (2026-11-18 part 10) The end of a CTA's TMA stores: TM_NWC 1 waits only until the stores have READ shared memory (cp.async.bulk
// .wait_group.read, as CUTLASS's tma_store_wait at kernel end); the writes complete with the grid (a dependent's griddepcontrol.wait
// and the next ordinary launch see them). 0: wait for the writes themselves (the previous form).
#ifndef TM_NWC
#define TM_NWC 0   // 1 measured level (part 10): not taken
#endif
#define BULK_END() do { if (TM_NWC) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); else asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); } while (0)
