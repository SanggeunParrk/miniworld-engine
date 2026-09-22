# Active B1–B4 SoL90 goal: continuation from v50

This goal turn made verified progress. Current development entry `../trimul_training_current.py` loads this directory's `epilogue_policy.py`. Production remains blocked by inherited B7 independent-reference dWL error, unchanged. SoL90 not reached; do not complete the goal.

User authorized node01 despite older goal text sayingnode02. Read current Slurm state before new allocations. Our latest jobs13606(full),13609(delaystress),13612(entry) produced success artifacts. Node01 had4 freeH100s; other4 are job12751 and must be left alone. Node02 full of own/othertraining; leave those jobs alone. No subagent delegation authorized.

## Current selection and results

BF16 C128/H256 L384/L768 bidirectional dropout25%/mask/residual. Savedinputaffine xn, originalBF16tri, FP32outputmu/rstd. No saved outputactivation. All11gradients match preceding v49 including mutatedinputs/gamma0/graphreplay.

Current config: count132, GATE_PHASE1, DIRECT_DP1, SPLIT_DN1, DN_PAIR1, DN_REG1, REG_EARLY_RAW1, PARAM_SHARED1, PARAM_UNROLL4, DTRI_ASYNC_EPI1, LOCAL_GATE1, PREFETCH_DY0; affineunroll4/8 and dTri storewidth32/64 for384/768 (read JSON). 253/255registers,0spills.

Node01 paired v49->current:
L384B1 206.304->191.872us (-6.9955%); wholeFWD+BWD1206.160->1189.216us(-1.4048%).
L768B1 695.904->644.448us(-7.3941%); whole4915.520->4857.216us(-1.1861%).
NCU191.456/639.008us. Measuredtraffic roofline60.38%/65.86%; optimisticunique-payloadmodel41.38%/49.60%. Do not equate either with rigorous fullalgorithmSoL90.

## Implemented

- Replace output LN parameter-gradient warp shuffles with32KiBshared scratch over dead dy/currentxn. Balanced32-way summation preserves exact addition tree. Scalar loads/row layout/unroll4 selected.
- Start nexttri/xn/stats TMA after paired dNorm WGMMA and a REQUIRED CTA barrier.
- Start dTri TMA store before gamma/beta shared reductions; wait before tile reuse.
- Remove first gridbarrier betweenPhaseA/B: eachCTA reads precisely its own dGate rows. Keep CTA fences, reinitialize local mbarriers, keep last gridbarrier for crossCTA parameterpartial reduction.

## Important race found and fixed

Earlier `trimul_b1_epilogue_20260921` and derived early-prefetch candidates passed normal numerical tests but failed numerics under memcheck with0addresserrors. `wgmma_wait<0>` releases operands per warp-group, not CTA. Thread0 could overwrite opposite-slot dProj while WG1 still read it. This fixed directory adds `allsync()` immediately after paired dNorm wait and before load_raw. MemcheckbothL/racecheck384 thenpass. Preserve this barrier unless a stronger explicit per-group completion mechanism replaces it. Early TMA ablation timings in prior experimentaldirs are not correctness-certified. Currentrootentry was never pointed at the unsafecandidate.

Deliberately delaying blockIdx%17==0 beforePhaseB passed bothlengths,3seeds,20graphreplay each. This tests elimination of the between-CTA barrier, not intraCTA operand reuse; sanitizer caught the latter independently.

## Rejected experiments this turn

- `reg_prefetch`: dy prefetch negligible; incompatible with32KiBparam scratch. Earlyraw needs above barrierfix.
- `param_vector`: transpose/swirled scratch and128-bitshared loads slower; retained scalarrowlayout.
- `gate_pipeline`: TMA2..7buffers gave <1% changes.
- `gate_group`: combine2/3consecutive CTA-owned tiles preserving exactWGMMAsummation; <1% changes. Neitherselected.
- Prioronepass Wg fusion/sharedparking already slower; do not rerun without new design.

## Next promising independent pipeline change

Use current `source-L768.csv` (ncu export succeeded with env -u PYTHONPATH PYTHONNOUSERSITE=1). Deduplicate SASS addresses; inline source lines duplicatecounts. Biggest remainingwaits are gateTMA(~8671longSBsamples) and WGMMAwaits(~2943/1746barriersamples). Deepgateprefetch did not help; gatephase may already be bandwidthlimited. Do not infer source-sample sums as durationpercent.

Inspect possibility of issuing dWproj weight_pair asynchronously, then dNorm GEMMs before waiting:
- Both read dProj; neither overwrites it. Wpaccumulators128regs and dNorm64regs already coexist in registerliveness.
- Current weight_pair does WGMMAcommit+wait0 before lowreg_dgrad. Could defer its wait to dNorm's pairedwait0, launching both independent GEMMs and overlapping dNormsetup scalarwork.
- Must preserve proper fence_regs for wp0/wp1 AFTER the finalwait; likely pass their references into lowreg_dgrad or split helper. Do not readaccumulators before asynccomplete.
- Current shared_role waits dGateTMA store+CTAbarrier before lowreg. Can only defer that wait if completed BEFORE sm0 is reused by PARAM_SHARED. Combine with required CTAbarrier before earlyraw, and ensure both warp-groups' WP andDN consumers completed before opposite-slot overwrite.
- Do not remove the discovered crucial CTAbarrier based solely on one warp-group wait.
- Use newisolatedderiveddirectory, testbaseline/candidate onnode01, then fullsanitizer and numericalchecks beforeselecting.

## Site / artifacts

v50 published owner-private:
https://miniworld-kernel-status.psk6950.chatgpt.site/trimul.html#b1-epilogue
commitf8d15c7de76971ad487ce24e2cb4b682cb84db93; receiptpublication-v50.json.
HTML/SVG current selected; first gridbarrier correctly removed in drawing, final one retained. Renderer`s ACTIVE points here. AllXML/HTML/scripts/14PyTorchsketcheschecked; croppedimage inspected.

Publishing authorization persists: user's instruction '앞으로 현재 개발상황 시각화를 그 html에서'. Existingowner-private site, onlyperformanceJSON/SVG/MD/HTML. Previouspushrejection was resolved by verifyingthisinstruction and sameartifactcategories/owner1groups0external0; subsequentpushapproved. No pendingapproval. Alwaysrecheckownerprivate/preserveaudience and push exactcommit before archive/save/deploy. Neverprintcredentials.
