# Dual B9+B10 SM90 optimization round2

## Outcome

No production change was promoted. Starting checkout `0f2d455b`, retained production source SHA256 `9a0586e65a5da41d3f2d913dbf3d566af203457387659cb625e1f282cea9dfe4`. This round tested three new implementation approaches while preserving the actual D128 module shape: M=L², KG128, KP1024, N128; two GEMMs with a BF16 gate-gradient boundary and final BF16 output. The 1.15x target remains unmet.

All GPU work ran in allocated node02 job13235, one GPU,6CPUs,40GB. Artifact directory was the working directory, keeping generated binaries outside the checkout root. L128 was not benchmarked.

## Diagnosis before rewriting

Fresh NCU detailed/source profiling of the retained L384 kernel showed:

- DRAM84.30% of peak, L2 approximately79.92%; separate instrumented duration136.10us.
- Only15.62% of scheduler cycles had an eligible warp.
- Long-scoreboard stalls accounted for56.7% of cycles per issued instruction. The source-correlated front TMA barrier wait (`NANOSLEEP.SYNCS`) had5,131 such samples.
- The front WGMMA wait0 had2,006 barrier-stall samples.
- Each front K stage issues three TMA commands: one F slab and two V slabs. F has64 contiguous BF16 rows=128B per reduction column, with aligned source strides; there was no source-level partial-cacheline waste at this production shape. This does not by itself prove optimal hardware request behavior.

Evidence: `ncu_diagnose.ncu-rep`, `ncu_diagnose.txt`, `ncu_diagnose.csv`, `ncu_source.txt`. Profiler replay timing is separate from graph benchmark medians.

## 1. Eight-warp producer/consumer pipeline

`specialized.py` uses exactly eight physical warps, split into one producer warpgroup and one four-warp WGMMA consumer. The producer manages TMA independently through per-slot full/empty mbarriers; the consumer releases a slot only after WGMMA wait0 and a128-thread barrier. Gate/front retain separate full-barrier phases while sharing the physical operand ring. No additional global intermediate or shared operand tile is introduced. The bounded artifact supports BM64, BN<=128 only and explicitly rejects other shapes/configurations.

A separate CPU reviewer enumerated6,120 gate/front/stage phase combinations and found the phase/lifetime mapping consistent. This is supporting analysis, not a substitute for GPU sanitizers.

All six tested K/stage neighbors matched Triton bit-for-bit, but were slower. The strongest tested ordinary version was K64/stages4 at162.920us versus retained132.856us. Lowering the consumer register request232->144 did not lower static allocation: cubin still used136registers/thread, stack0, with256threads, restricting CTA residency.

An explicit two-CTA launch bound produced REG128, stack0, and improved K64/stages3 to139.121us, but still lost to both the retained strongest4-warp path132.382us and matched8-warp path133.641us. Triton was142.011us in that comparison. Attempting the three-CTA/80-register static budget failed PTXAS: its WGMMA instruction required at least90registers. No spill-based or slower specialization was promoted.

Evidence: `probe.json`, `specialized144_probe.json`, `specialized2cta_probe.json`, `probe136.log`, and `compiled/`. These are three rotating graph rounds per configuration. Since the candidates lost materially, no claim of sanitizer validation or production readiness is made.

## 2. TMA tensor data prefetch into L2

Public `cute.prefetch` accepts the existing partitioned TMA source and issues an operand-data prefetch hint. It leaves ordinary TMA loads, completion barriers, shared storage and numerical math unchanged. The SM90 instruction is documented in [NVIDIA PTX8.0](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor); installed CuTe `algorithm.py:621` provides the public helper.

Two variants prefetched front F operands one or two additional stage windows ahead of the ordinary three-stage shared ring. Both matched exactly but regressed in five rotating graph rounds:

| L | Triton us | Retained us | Extra one-window us | Extra two-window us |
|---|---:|---:|---:|---:|
|384|141.692|132.721|137.205|153.494|
|768|535.494|509.889|516.641|587.339|

A narrower variant issues only the impending refill's F hint after WGMMA commit and before wait0, with no prologue burst. Five rotating rounds showed no stable improvement:

| L | Triton us | Retained us | Near-refill hint us |
|---|---:|---:|---:|
|384|142.130|133.059|133.454|
|768|537.340|509.200|508.829|

Evidence: `prefetch.json`, `prefetch_near.json`. These results do not establish why prefetch regresses; increased request/cache pressure is a hypothesis, not a measured causal conclusion. No hint policy was promoted.

## 3. Nonswizzled V operand request geometry

`v_linear.py` attempts a BN-contiguous nonswizzled front-weight shared layout, preserving gate W and front F layouts and the ring capacity. The purpose was to replace two128B-swizzled TMA slabs with one256B contiguous rectangle. The compiler rejected the layout when lowering the WGMMA MN-major shared descriptor:

`make_gmma_smem_desc` could not represent layout `((128,16),1,4,3):((1,128),0,2048,8192)`.

Thus no executable or measured TMA-command reduction exists for this candidate. The existing two-slab geometry remains. `v_linear.log` records the bounded rejection; no attempt was made to hide a transpose/copy or change the fusion algorithm.

## Production status

Owned production source/test files remain unchanged in this round. The retained prefetch-only implementation already has its prior36-case memcheck/racecheck and whole-module validation. No new runtime option, cache axis or unverified implementation was deployed. GPU slot was returned to the parent after the last bounded probe.
