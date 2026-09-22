# Local two-tile reuse: 512-thread CTA

Experimental; not selected or production-wired. `make_paircta.py`, `paircta_plan.py`.

The previous132-SM grid had264CTAs,256threads each. This design merges corresponding work into132CTAs,512threads each with196608dynamicshared bytes. Four warpgroups remain active;128registers/thread and no stack/spills in initial unroll8/4 compilations.

- dW: four hidden groups(H128) ×20splits =80CTAs. One CTA computes the same two H64 groups as before, reusing a single saved x_n tile. Four WGs own(hidden-half, gate/projection). Each WG keeps the same N128 WGMMA, row sequence, accumulation segments and output mapping. Double-buffered98KiB stages load32KiB preactivation,16KiB upstream and16KiB x_n, plus32KiB G/P outputs. B7 is evaluated once.
- dX:52CTAs, each owns two64-row tiles. Four WGs own(row-half, channel-half). Projection and gate weights are read once for both row tiles. G stages64KiB + retained P64KiB; LN X/residual occupy128..192KiB; next gate occupies0..64KiB. LN temporary reductions reuse dead P space at64..74KiB.
- Global ring remains12MiB. Each H128 producer publishes two H64 generation counters. Consumers wait and free two adjacent row tiles. Full store completion, release/acquire and end-of-work reset remain.
- Final reduction preserves the same logical8hidden groups and104LNpartials. It remains one cooperative kernel launch, with the original forward saves and BF16 rounding boundaries.

Unlike the earlier four-WG experiment (splitting N into32-channel pieces), this increases independent row/hidden work per CTA and reduces redundant L2 reads. Intended L768 traffic savings: about0.604GB x_n reads and1.359GB projection/gate weight reads, before cache effects. Accuracy atL64 initially passed; training-bucket performance/verification pending. Shared byte counts and overlap lifetimes must be checked by memcheck/racecheck/synccheck before promotion.

Training-bucket initial checks pass at fixed tolerances. Initial sp20/u8 L384433.520µs andL7681591.504µs are slower than selected364.848/1429.136µs. Sweep unroll8/4/2 × splits14/16/18/20/22/24 found~1556.7µs best(u8,s20), still slower. NCU `paircta-full-L768.ncu-rep`:1.64ms,DRAM48.21%,L281.07%,SM33.89%;34.1% warp cycles between issued instructions attributed to CTA barrier stalls. No selected-source promotion.

Two outstanding WGMMA groups, with a stage reused only after wait_group1 retires its previous reader, were prototyped in `make_mma_pipeline.py`. Ordinary C++ accumulator versions spill at128register budget and were refused. Named PTX accumulators initially remained live across row iterations; resetting at each row removed those extended lifetimes. The non-unrolled reset sources execute with no spills and pass initial checks, but measured~471µs directL384 /1582µs ringL768. Expanded variants spill and remain rejected. No asynchronous group variant is selected or fully sanitizer-qualified. `make_named_acc_pipeline.py` plus `*_namedpipe_reset` retain evidence.
