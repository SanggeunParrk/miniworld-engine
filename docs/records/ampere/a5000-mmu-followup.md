# Follow-up of four A5000 MMU faults

2026-09-12. Follow-up to [the initial memory-access fix](a5000-memory-access-fix.md).

The four original Xid 31 failures have no confirmed root cause. This investigation tests candidate schedules inferred from the incomplete original logs; those logs do not prove which exact config faulted. No additional production kernel candidates are excluded on the strength of this investigation.

## Isolation and original conditions

Slurm job 1680273 allocated two A5000s on gpu02. The runner checked the allocated device UUIDs and verified no compute processes were using either card before starting. Original card: `GPU-bbfd1018-c7ea-a60c-5567-ec618e5dcada`; comparison card: `GPU-4124b909-d7ce-3bdd-53f0-64cac0c27535`. Driver: 570.124.06. Triton: 3.6.0.

The probes call the actual Pairformer/AugmentedAttention modules at the original shape, dtype, dispatch pin and training/inference mode. They intercept only the named Triton operation and force the suspect configuration on the module's live tensors. The records include shapes, strides, dtypes, scalar arguments, tensor addresses, available device memory and workload identities. Available workload identities are compared against the original failing shards, not merely against the same cache bucket.

| Target | Original module invocation | Candidate warp/stage pairs |
| --- | --- | --- |
| Transition forward | Pairformer train, L384, d_pair256, persistent LN backward | (1,3), (1,4), (1,5) |
| Transition recompute backward | Pairformer train, L384, d_pair128, persistent LN backward | (2,1), (2,2) |
| Triangle multiplication output projection | Pairformer eval, L512, d_pair256, fused gate | (1,3), (1,4) |
| AdaLN forward gate | AugmentedAttention train, L512, d_single768/d_cond384, BF16 core | (2,4), (4,1) |

Tile sizes are recorded in the companion JSON and raw probe records. The test matrix covers:

- Nine candidates on the original GPU, 64 forced launches at each actual module call site.
- The same nine candidates on a comparison A5000, also 64 launches per call site.
- Nine candidates on the original GPU with a live additional allocation leaving about 2 GiB free, eight launches per call site. This preserves the actual module's intermediate tensors; it is not a standalone contiguous-tensor test.
- Compute Sanitizer memcheck of the named target kernels, two forced launches per call site.
- Four representative candidates with a separate parent process holding a CUDA context on the original GPU, 64 launches per call site. The build coordinator also holds a context on GPU0, so the original faults' GPU0 correlation alone does not establish defective hardware.

The ordinary/pressure/two-context tests compare the first 65536 output elements with the cached winner on identical inputs. This is a check for configuration-dependent numerical divergence, not an independent full-module accuracy qualification.

## Sanitizer scope and OOM

Five full-module sanitizer trials (three Transition forward and two recompute-backward candidates) completed the target, then hit `cudaErrorMemoryAllocation` in later module work. The original capacity-failure logs are retained. For those cases, retries stop immediately after checking the target on its actual module tensors. A target-only sanitizer pass does not mean the rest of the module completed under instrumentation.

The sanitizer filters to the named kernel. Other kernels execute to construct the real module state but are not instrumented. Triangle multiplication and AdaLN sanitizer cases retain the complete module execution. Numerical reference comparisons are disabled under sanitizer; the separate ordinary and pressure tests perform them.

## Interpretation and artifacts

The companion [JSON record](a5000-mmu-followup.json) contains the final statuses, checked-launch counts, maximum sampled relative L2 difference, memory-pressure readings and original-workload comparisons. Raw scripts and outputs live in `.scratch/a5000-mmu-followup/`.

ECC is disabled on both cards, so unavailable ECC counters cannot establish healthy hardware. Remapped-row counters are zero, and no remapping failure is reported. The original four fault messages are preserved in `xid-history.out`. Neither a fixed bad candidate nor a hardware defect can be inferred solely from Xid 31.

The tests do not reproduce every original sequence of compiled-module loads, allocator events or scheduling decisions. A passing result narrows the investigation; it does not establish why the original faults occurred. The improved fatal-config logging in the builder remains necessary if the fault recurs.

Build job 1680229 was stopped through Slurm to obtain isolated diagnostic resources; existing shards and compiler-cache entries were retained. Job 1680274 verifies the follow-up results, merges valid saved measurements and resumes incremental `build all`.

## Final outcome

All 40 accepted scenario checks passed, with **1838 forced target-kernel launches**. The 51 recorded workload identities available for comparison matched the original shards exactly. The sampled output differences against the cached winner were zero in the ordinary, pressure and two-context checks. All nine final target-kernel sanitizer configurations reported `ERROR SUMMARY: 0 errors`; the five earlier full-module OOM trials remain separately recorded and are not counted as clean full-module sanitizer runs.

No new Xid 31 appeared in the driver log through the final check after build job 1680274 started. Four representative tests with a resident parent CUDA context also passed. The four historical MMU faults remain **unreproduced and unexplained**, rather than classified as fixed kernel bugs or hardware defects.

Job 1680274 passed the diagnostic gate, merged 345 retained/replay shards with no skipped records, and resumed the cache build. After that merge, 25 required keys remained to be filled, followed by the alternative-kernel driver pass. This is a build-progress snapshot, not a complete A5000 qualification.
