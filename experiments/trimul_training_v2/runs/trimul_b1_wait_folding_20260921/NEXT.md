# Active SoL90 goal continuation from v51

Goal remains active and incomplete. User authorized node01. No delegation authorized. Preserve BF16 C128/H256 L384/L768, dropout25%/mask/residual, saved input affine x_n, original BF16 tri and FP32 output mean/rstd. No output LN activation saving. B7/cuBLAS unchanged. Production blocked by inherited independent B7 dWL error0.055569% vs0.05%; current entry is development only.

## Selected verified checkpoint

Current entry `runs/trimul_training_current.py` loads this directory's `wait_policy.py`. Parent baseline is `trimul_b1_epilogue_fixed_20260921/epilogue_policy.py` (v50). Read selected-L*.json; count132/part2, existing flags plus DEFER_DG_STORE1/SKIP_FINAL_GROUP_SYNC1, ASYNC_WP_DN0/WG_PARAM_SYNC0. All arithmetic and tensor policies unchanged.

Changes: defer dGate store wait to paired dNorm completion before existing CTA barrier; remove final dTri warp-group barrier subsumed by immediate caller CTA barrier. **Retain the CTA barrier after paired dNorm wait before load_raw**: each WGMMA wait only completes its own warp-group; omitting this caused a verified race previously.

Paired v50 -> v51 CUDA event medians:

|L|B1 us|BWD us|module us|
|---|---|---|---|
|384|190.432 ->188.480|901.216 ->899.168|1188.304 ->1185.728|
|768|646.704 ->635.888|3588.288 ->3576.560|4789.040 ->4774.000|

NCU B1 188.768/624.320us, measured-traffic roofline61.22%/67.42%, optimistic unique-payload41.97%/50.76%; not SoL90. Definitions/limitations in README. Independent diagnostic gate phase DRAM73.71%/85.00%, ordinary standalone37.024/109.472us; excludes final CTA reduction. Do not treat gate-only85% as full algorithm SoL.

All11gradients and outputs bit-exact against parent, mutated live graph/eager/gamma0 pass. Memcheckboth/racecheck384 pass. CTA delay +WG1 delay before dTri wait, both lengths3seeds20replays bit-exact. Current entry passes. Jobs13636/13641/13648 all completed. Job13632 fullbench succeeded but its first profile attempt removed PYTHONPATH and failed imports; fixed verify script cleans env ONLY for ncu --import and reran13636. No numerical/code issue from that failed profiling attempt.

## Rejected candidates this continuation

- `trimul_b1_gemm_overlap_20260921`: asynchronous dWproj+dNorm alone negligible; dGate wait deferral retained.
- `trimul_b1_recompute_overlap_20260921`: overlap LN affine with gate GEMM and sigmoid with projection GEMM, no meaningful gain; not selected.
- `trimul_b1_param_barriers_20260921`: WG-only LN parameter barriers no gain; keep full CTA barriers. Final redundant group barrier removed.
- `trimul_b1_reduce_ilp_20260921`: unroll1..132 final partial reduction. Low unroll much slower; original compiler unrolling already good. Valid run13639; earlier13634 failed pragma macro expansion. CUDA pragma should reference constexpr int, not macro identifier.
- `trimul_b1_reduce_load_20260921`: volatile vs __ldcg vs plain loads no gain; retain volatile. All six outputs exact, no new sanitizer for these rejected variants.
- `trimul_b1_gate_bound_20260921`: diagnostic kernel only; not deployment. NCU raw exports cleaned and successfully regenerated. Default Plan diagnostic requires dGate buffer copied from original p1 output before measuring.

## Next direction

Avoid repeating the above small tweaks. v50 NEXT.md lists earlier failed deep gate buffers, row grouping, LN scratch vectorization and shared parking. Gate phase is already near bandwidth limit at L768 and only ~17% of B1, so concentrate on main phase's scalar LN/derivative/shared-memory instruction cost or a materially different overlap design. Use current NCU source/SASS attribution rather than summed inline sample percentages. 253/255 registers and1CTA/SM remain; benchmark any register-pressure split against the additional HBM traffic explicitly, since user allowed considering a backward split. Do not inflate SoL by redefining the bound around inefficiencies. Keep full error checks and synchronization proof.

## Website

v51 published owner-private successfully; `publication-v51.json` has exact IDs.
https://miniworld-kernel-status.psk6950.chatgpt.site/trimul.html#b1-waits
Commit28a72efa684b9371e46f300e08df4e92971782a3. Access verifiedowner1/groups0/external0. Standing user instruction authorizes keeping this HTML current. Follow exact-source push/archive/save/private-deploy workflow and never print credentials. Existing source checkout `runs/anthropic_b1b4_pipeline_20260919/site-visuals`; .openai/hosting.json identifies project. Site remote clean afterpublish. Local SVG/HTML14formula/XML/link checks passed.

Recheck Slurm before launching. Other node01 jobs include12751 (worud4GPU) and independent opm-bwd agent jobs; do not touch. No jobs from this continuation remain active. Node02 training unchanged.
