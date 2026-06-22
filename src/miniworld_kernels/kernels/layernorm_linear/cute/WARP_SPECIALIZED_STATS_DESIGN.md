# Warp-specialized LN stats for the fused LayerNormLinear (design)

Goal: beat M1 at **large d** by making the LN reduction **free** — move it off the
math (WGMMA) warps onto otherwise-idle warps, reading X from **gmem** (stable, no sA
recycle race), overlapped with the tensor-core compute.

## Why (measured motivation)
- `LNL_GEMM_ONLY=1` diag: the fused **GEMM alone ≈ or faster than M1** at every d
  (d=768 M=262144: 0.63 vs M1 0.63; d=512: 0.26 vs 0.36).
- The **entire** large-d gap is the in-kernel reduction (d=768: 0.63→1.21 ms, ~2x),
  because it runs on the **same warps that issue the WGMMA** → steals MMA issue slots
  in the compute-bound regime. (Confirmed not sync_warp, not per-k-tile serialization,
  not the per-(m,n) redundancy — `prev_m` reuse can't trigger: persistent strides
  tiles across CTAs. BLK_N=192 cuts tiles but reintroduces the sA recycle race.)
- Corollary: if stats become free, M2 ≈ fused-GEMM ≈ M1's GEMM **<** M1-total (which
  also pays a separate stats kernel) → **M2 wins large d too**.

This is the FlashAttention-4 pattern: dedicated non-MMA warps doing the side compute
(there: softmax/correction) overlapped with the MMA warps.

## CTA warp budget (pingpong + persistent, today)
`threads_per_cta = (mma_warp_groups+1)*128 = 3*128 = 384 = 12 warps`:
- warps **0–3**  : math WG0 (WGMMA + epilogue, even tiles)
- warps **4–7**  : math WG1 (odd tiles)
- warps **8–11** : load WG. `ab_load_warp_id=8`; `num_ab_load_warps=1` ⇒ **warp 8 = TMA
  producer/scheduler; warps 9–11 are IDLE** (they hit `setmaxregister_decrease` then fall
  through). ← our free stats workers (96 threads).

## Core design: a stats producer→consumer pipeline

The stats are **per output tile** (one reduction of X[m-tile] per (m,n) tile). Model it
exactly like quack's AB pipeline, but at output-tile granularity:

- **Producer = warps 9–11** (the idle load-WG warps). Iterate the SAME scheduler
  sequence; for each tile reduce X[m_base:m_base+BLK_M] from gmem (coalesced) into the
  pipeline's current stage; commit.
- **Consumers = the 2 math WGs** (leapfrog). Per tile: do the mma (NO reduction),
  `consumer_wait` the stats stage, run the epilogue reading rstd/c1 from that stage,
  `consumer_release`. Advance the stat state by `mma_warp_groups` (skip the other WG's
  stage), exactly like `ab_read_state.advance_iters(...)`.

### SMEM additions (in `__call__`'s SharedStorage)
```
sStat_full : MemRange[Int64, S_STAT*2]   # mbarrier array (full)
sStat_empty: MemRange[Int64, S_STAT*2]   # mbarrier array (empty)
sStat      : Align[MemRange[Float32, S_STAT * 2 * BLK_M], 16]  # rstd|c1 per stage
```
`S_STAT` = 3–4 (enough for the producer to run ahead of the 2 in-flight math tiles).
Each stage = 2*BLK_M f32 = 1 KB; 4 stages = 4 KB (negligible vs sA/sB).

### Pipeline object
A `cutlass.pipeline.PipelineAsync` (or a NamedBarrier pair) with:
- producer_group = 96 threads (warps 9–11),
- consumer arrive count = 1 per math WG per tile (the WG's signalling thread), matching
  how the AB empty barrier counts warps. Mirror `make_ab_pipeline`.

### Producer loop (new, in the `warp_idx>=ab_load_warp_id` branch, warps 9–11)
```python
if 9 <= warp_idx <= 11:                      # stats workers
    sched = TileSchedulerCls(); wt = sched.initial_work_tile_info()
    pstate = make_pipeline_state(Producer, S_STAT)
    while wt.is_valid_tile:
        m_base = wt.tile_idx[0]*BLK_M
        stat_pipe.producer_acquire(pstate)
        _reduce_gmem_coop(mX, m_base, sStat[pstate.index], ..., nwarps=3)  # 3-warp coalesced
        stat_pipe.producer_commit(pstate)     # async-proxy commit, like cp.async
        pstate.advance()
        sched.advance_to_next_work(); wt = sched.get_current_work()
```
`_reduce_gmem_coop` already does coalesced warp-per-row + butterfly; just parameterize
`nwarps=3` and a per-stage smem base. Writes are all-lane / own-row (converged → no
divergent-barrier deadlock, the bug synccheck caught earlier).

### Consumer changes (math WGs)
- Drop the in-mma sA reduction entirely (mma → stock GEMM, full WGMMA pipelining).
- Per tile: `stat_pipe.consumer_wait(rstate)`; the epilogue's `SmemColVec` reads
  `sStat[rstate.index]` (pass the stage base into the epi ctx, OR copy the 128 floats
  to a fixed per-WG s_rstd first — cheaper to plumb); `consumer_release(rstate)` after
  the epilogue; `rstate.advance_iters(mma_warp_groups)`.
- WG1 pre-skips one stage (`rstate.advance_iters(1)` before the loop), like it already
  does for `ab_read_state`/`epi_read_state`.

## Correctness properties
- **No sA recycle race**: stats read X from gmem (immutable input), never the recycled
  sA stages. (This is what BLK_N=192 broke; gone here.)
- **No divergent-barrier deadlock**: producer writes are all-lane/own-row converged
  stores; the pipeline uses mbarriers (async proxy), not divergent named barriers.
- **No MMA-throughput theft**: the reduction is on warps 9–11; the math warps issue
  WGMMAs back-to-back. Overlap is real (different warps, same SM).
- **Ordering**: producer commits tile T's stats before any consumer can `consumer_wait`
  it; the math WG's epilogue for T blocks until then. Producer runs ahead up to S_STAT.

## Risks / open questions (to settle during impl)
1. **Stats throughput**: only 3 warps (96 thr) reduce 256 rows (2 in-flight tiles) ×
   K. For large d (long WGMMA) it hides; for **small/mid d** (fast tiles) the 3 warps
   may become the bottleneck and stall the math WGs. Mitigations, in order:
   (a) keep the CURRENT in-math-warp path for d ≤ 256 (already wins there) and switch
       to warp-specialized only for large d — a compile-time `d`-threshold branch;
   (b) a dedicated 4th warpgroup (threads_per_cta=512) — more workers but lower
       occupancy; measure.
2. **Stage count S_STAT**: too few → producer stalls; too many → smem. Start 4, tune.
3. **Plumbing the stage index into `SmemColVec`**: simplest is the consumer copying
   `sStat[stage] → fixed s_rstd half` right after `consumer_wait` (128-float copy,
   negligible), so `SmemColVec.begin` keeps reading the fixed per-WG buffer unchanged.
4. **Scheduler instance for the producer — RESOLVED (big simplifier).** Our kernel runs
   **PersistenceMode.STATIC** on SM90 (CLC needs arch>=100; semaphore=None ⇒ not
   DYNAMIC; `get_scheduler_arguments` ⇒ STATIC). In STATIC the tile order is a pure
   formula: `work_idx = cluster_idx (block_idx.z)`, advance `work_idx += grid_dim.z`,
   tiles = `_delinearize_work_idx(work_idx)` (handles the swizzle). So the stats warps
   **independently replicate the exact same tile sequence** the math WGs get — NO
   sched_pipeline coupling, NO make_sched_pipeline override. (quack even has this path
   commented out at tile_scheduler.py get_current_work: `# elif STATIC: return
   self._delinearize_work_idx()`.) The math WGs still get tiles via the broadcast
   sched_pipeline; the stats warps just recompute the formula. Only the **stats
   pipeline** (stats→math, S_STAT stages) remains to build.
5. **`make_fake_*` compile path**: add the stats pipeline mbar ptr + fake sStat so the
   COMPILE_ONLY trace matches the runtime smem struct.

## Incremental bring-up plan (each step verified on a compute node)
1. ✅ **DONE & VERIFIED.** Stats warps (9–11) independently replicate the STATIC tile
   sequence and reduce X→rstd from gmem into a debug buffer `mDbg`; math WGs untouched.
   `diag_ws.py` (LNL_WS_DEBUG=1): **0 unfilled rows, max|rstd-ref|=1.2e-7, cos=1.0** on
   all shapes incl QKV. No deadlock; shipping path (LNL_WS_DEBUG=0, mDbg gated to None)
   stays cos=0.999997. ⇒ the scheduler unknown is fully retired; the idle-warp reduction
   is correct. Helpers: `_stats_dump_gmem`; branch is `elif const_expr(_WS_DEBUG):` in
   the load-WG section; mDbg plumbed through EpilogueArguments (gated by `_WS_DEBUG`).
2. **NEXT.** Stand up the stats pipeline (mirror `make_sched_pipeline`: a thread-producer
   `PipelineAsync`, producer=stats warps, consumer=math WGs). Producer: per tile,
   `producer_acquire` → reduce X → write the WG-half s_rstd/s_c1 → `producer_commit`.
   Consumer (math WG): drop the in-mma sA reduction (plain GEMM), `consumer_wait` before
   the epilogue (SmemColVec reads the WG-half unchanged), `consumer_release` after; WG1
   pre-advances one stage. Tile→WG split is `g = tile_count % mma_warp_groups`, matching
   the leapfrog, so producer full-arrives and consumer waits match 1:1 (no deadlock).
   Termination is automatic (producer's STATIC `is_valid_tile`; consumers' sched stream).
3. Verify cos=0.999997 all shapes. Bench: large d should drop to ≈ fused-GEMM-only
   (≈ M1's GEMM, measured 0.63 ms @ d768 M262144) → **win** (M1 also pays a stats
   kernel). Watch small-d regression (3 stats warps may bottleneck fast tiles → risk-1a:
   keep the in-math-warp path for d≤256).
4. Tune S_STAT; risk-1b (4th WG) if 3 warps bottleneck.

## Fallback
If 3-warp throughput / scheduler-sync proves intractable, keep the current shipping
kernel (wins d≤256, correct everywhere) and use M1 for large d — both already correct.
