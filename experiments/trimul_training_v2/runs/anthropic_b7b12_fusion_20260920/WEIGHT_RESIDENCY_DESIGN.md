# Next architecture: resident input-projection weights

Status: prototypes implemented but NOT selected. The persistent-register prototype passes L64, then fails input-dgrad after the first tile at larger lengths. A reload-each-tile control passes L128/L768, but dX alone takes2651.968µs at L768/count80, slower than the full current B7–B12 (~1.7ms). The cache prototype has no qualified performance result. Current published checkpoint is front_twocta_kindwg; the older Plan default is unchanged.

## Motivation

The current two-CTA source uses roughly4.36GB DRAM at L768,76%DRAM and79%L2 throughput. Repeated per-row TMA transfers of four weights total about2.42GB of L2 traffic (262144bytes per64-row tile ×9216tiles). Caching them is a concrete route beyond simply adding warps. Window progress barriers only help slightly:2048tile windows measured1690.544µs vs1727.504µs without windows, with unchanged arithmetic. Smaller windows are slower.

## Proposed dX CTA

512threads: two GLU/TMA producer warpgroups, two consumer warpgroups, one CTA/SM. Each consumer owns64input channels. Cache W_Lg/W_Rg in128KiBshared memory. Cache W_L/W_R as WGMMA-RS A fragments in consumer registers:2matrices ×16 K16steps ×4packedBF16words =128registers/thread. Use explicit TMA and WGMMA-SS/RS. Anthropic's load_frag_bf16 and RS wrappers provide the reference fragment layout; their register A layout is [row0,k0], [row8,k0], [row0,k8], [row8,k8]. The existing RS wrapper has transB=0; front dgrad transposed operands need transB=1.

Shared memory: two48KiBactivation slots (pre32KiB+upstream16KiB each), followed by128KiBresident gate weights =224KiB. Static gamma/row statistics/barriers must fit the remaining per-SM budget. Proposed dynamic register allocation: producers48, consumers208;512thread initial128register quota supports this exactly. Reject all stack/spill allocations before execution.

Transpose contraction: compute dXn^T = Wfront × dConcat. For each side, G128+G128 then P128+P128 preserves Lg256→Lp256→Rg256→Rp256. SS consumes resident gate weights; RS consumes resident projection weights. Gate dgrad computes Wgate × dgate^T and rounds separately. Preserve all established BF16 boundaries. Verify that operand transposition retains required numerical tolerance before timing.

Producer GLU stores G into the upstream-gradient shared slot and P into preact storage. Process hidden channels in two64-channel chunks, holding only8packed P values/thread before a producer barrier and compaction. All reads of a chunk finish before its preact is overwritten. This reduces producer registers compared with holding full G/P tiles.

Both left G/P activation slots remain available through P contraction, then are recycled for right side. Publish ready after GLU; release only after the P MMA retires. After right P0, load X/residual into slot0; after right P1, prefetch next-row gate into slot1. This overlaps LN with the next gate transfer.

## Input LN with resident registers

A normal expanded LN epilogue plus128resident weight registers will spill. Instead, transpose the rounded dXn into16KiBshared storage in slot0. X and residual occupy the next16KiB each. Consumer threads prefetch mean/rstd into static shared arrays. Accumulators and gate packed values must die before this epilogue.

Compute dgamma/dbeta first: each thread owns one channel and32rows (two128thread halves). Accumulate two scalar partials in registers, reading shared dXn/X/mean/rstd. Then compute dx in row groups of8threads,16channels/thread, two passes across64rows. Keep explicit RN multiplication before LN subtraction. This changes reduction association, so accept only if the existing dx2e-5 / LNparams5e-6 limits pass all replay tests; do not loosen limits.

After dx consumes all dXn, reuse dXn shared space for two channel-partial arrays and merge the row halves. Keep one running parameter-gradient value per consumer thread across persistent row tiles. TMA-store dx from the X region and release slot0. Resident weights are never overwritten. Keep A-fragment register fences until the final asynchronous WGMMA use retires, as Anthropic does.

## dW role and host wiring

The shared full kernel would use4hidden128 groups, DW_SPLITS per group, and132CTAs total. A512thread dW CTA uses4WGs: (hidden-half64, gate/projection) pairs. Each WG has one N128 accumulator[64]. GLU distributed across all512threads; double-buffered96KiB stages fit. Keep2segments of partial weight accumulation for numerical accuracy. This has the same per-SM working lanes as the current two H64CTAs, but it is not yet tested.

Use Plan-style H128 maps/partial layout, with configurable512threads in the launcher; do not change the selected256thread default. No production dispatch edits until the integrated one-launch implementation qualifies. Do not modify Claude's B1–B4 code.

## Concrete experiment status

Source generator `make_cached_variants.py` had an overlapping textual substitution that reduced GLU coverage. It was corrected to distinguish hi/qi loops. Current q2 sources process all128 hidden channels as8chunks×2register pairs. Use the `_loop` variants: a bounded outer loop avoids compiler spills. Four such cache configurations pass L64; large shapes expose a separate repeated-tile issue. Never use the earlier incorrect source timings.

Intermediate debugging: GLU, gate dgrad and generic shared P values match the reference; first-tile dgrad matches, later projection-half0 contraction differs. Cached-register XOR checks remain unchanged across tiles. Compiler fences and an extra proxy fence do not fix it. Reloading projection fragments each tile fixes correctness, but its dX-only latency is uncompetitive. No production use. `cached-*-L*.log`, `debug-cached-loop-L128.pt`, and debug source variants retain evidence. All executed cubins are required to have zero ptxas stack and spills.
