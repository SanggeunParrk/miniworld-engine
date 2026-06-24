"""READABLE map of quack's gated GEMM (CuTeDSL / SM90) + the bdll store.

quack's kernel is hard to read not because the GEMM is exotic, but because the
GEMM body (`GemmSm90`, ~2000 lines) is buried under mixins + a persistent tile
scheduler + pingpong + varlen + gather_A + multicast clusters + stochastic
rounding. The ACTUAL gated-GEMM logic you care about is small.

This file is two things, both pointing at the real (working) source so you can
develop against it:

  1. NAVIGATION MAP  — what each phase is and which real line it lives on.
  2. The two pieces you'll actually MODIFY, lifted out and de-cluttered:
       (a) the gated GLU epilogue  (sigmoid(gate) * proj, fused in registers)
       (b) the bdll [B,D,L,L] direct store wiring (host side + the patched assert)

Paths below are relative to:
  .pixi/envs/default/lib/python3.12/site-packages/quack/

⚠️ This is a READING aid, not a drop-in kernel. The real control flow (TMA
producer warp / WGMMA consumer warpgroup / mbarrier pipeline) stays in quack —
don't hand-reimplement it blind. Compile/verify anything you build on a COMPUTE
NODE (srun), never the login node.
"""

# =============================================================================
# 1. NAVIGATION MAP — read the real kernel in this order
# =============================================================================
#
# HOST (per launch), GemmSm90.__call__            gemm_sm90.py:372
#   - set dtypes/layouts from the torch tensors    :411-418
#   - build TMA atoms for A, B                      :437-452  (_make_tma_atoms_and_tensors :1911)
#   - build TMA atom for the OUTPUT (D/PostAct)     :458-478  (_make_tma_epi_atoms_and_tensors :1877)
#   - epi_to_underlying_arguments (epilogue setup)  :480      (gated override: gemm_act.py:196)
#   - tile scheduler + grid                         :483-490  ← persistent-GEMM bookkeeping, ignore for understanding
#   - SharedStorage struct (sA/sB/sD + mbar ptrs)   :495-523
#   - launch kernel                                 :526-553
#
# DEVICE, GemmSm90.kernel                          gemm_sm90.py:557
#   - warp specialization split:
#       warp_idx >= ab_load_warp_id  → PRODUCER (TMA loads A/B → smem)  :684-788
#       warp_idx <  ab_load_warp_id  → CONSUMER (WGMMA + epilogue)      :790-...
#   - PRODUCER: persistent loop, per tile: local_tile A/B, tma_get_copy_fn,
#               load_AB drives the mbarrier pipeline                    :714-778 (load_AB :983)
#   - CONSUMER: partition_fragment_ABC (acc/tCrA/tCrB regs)             :810
#               mma() = prologue + mainloop WGMMA, wait_group           :861 (mma :1126)
#               epilogue() = acc → regs → (gated act) → smem → TMA → gmem :1192
#
# THE GATED EPILOGUE (what makes it "gated"), per epi subtile:
#   GemmGatedMixin.epi_visit_subtile   gemm_act.py:222   ← sigmoid(gate)*proj
#   GemmGatedMixin.epi_to_underlying_arguments gemm_act.py:196 ← N//2 + asserts
#   _gated_epi_tile_fn                  gemm_act.py:183   ← halves epi tile N
#   sigmoid (tanh-based)               activation.py:37
#   gate_fn_map["glu"]                 activation.py (sigmoid(g)*p; see act table)
#
# Key idea of "gated": the GEMM computes D = A @ B where B's N dim is 2x wide and
# holds [gate | proj] interleaved. The epilogue reads adjacent register pairs
# (gate, proj), does sigmoid(gate)*proj in-register, and stores HALF the N width.
# gate/proj never hit gmem.


# =============================================================================
# 2a. THE GATED GLU EPILOGUE  — lifted from gemm_act.py, de-cluttered
# =============================================================================
#
# Real source: GemmGatedMixin.epi_visit_subtile  (gemm_act.py:222-242)
#
# `tRS_rD` = this thread's slice of the accumulator (acc_dtype=f32) for one epi
# subtile, AFTER the base epilogue (alpha/beta/bias) has run. Its N dim is the
# full 2x width: even lanes = gate, odd lanes = proj (because we interleaved the
# B weight that way on the host — see launch.py:_interleave).
#
# `tRS_rPostAct` = the HALF-width output we actually store: sigmoid(gate)*proj.
#
# The SM90 (arch==90) branch reads pairs and writes pairs. The 4*i indexing is
# because on SM90 the accumulator register layout interleaves two values per
# "logical" element (STSM/wgmma frag layout) — that's why a plain `2*i/2*i+1`
# wouldn't line up; see also permute_gated_Cregs_b16 (gemm_act.py:253).
#
#   @cute.jit
#   def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
#       # run base epilogue first (bias/scale/etc.) — fills tRS_rD in place
#       GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
#
#       # output buffer is half the N width of the accumulator
#       post_layout = cute.recast_layout(2, 1, tRS_rD.layout)        # N -> N/2
#       tRS_rPostAct = cute.make_rmem_tensor(post_layout.shape, self.acc_dtype)
#
#       # SM90: act_fn is gate_fn_map["glu"] == lambda g, p: sigmoid(g) * p
#       for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
#           tRS_rPostAct[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
#           #                               ^gate          ^proj   (adjacent cols)
#       return tRS_rPostAct
#
# (the SM100 branch at :237 packs f32x2 for the newer MMA; ignore for H100.)
#
# act_fn for "glu" is, in plain math (activation.py sigmoid is tanh-based):
#
#   def glu(gate, proj):
#       return sigmoid(gate) * proj          # sigmoid(x) = 0.5 + 0.5*tanh(0.5*x)
#
# To add a SECOND output that is plain sigmoid (your `gate = sigmoid(x@Wg)`,
# no proj partner): you can't reuse this glu epilogue for it — its epilogue is
# `sigmoid(g)` with NO multiply and FULL N width (not halved). That's the
# gemm_act.py "act" path (GemmActMixin.epi_visit_subtile :145, activation="sigmoid"),
# a different epilogue class. Hence: left/right use the "gated" GEMM, the final
# gate uses a separate "act" GEMM (or its own tiny launch).


# =============================================================================
# 2b. THE bdll [B,D,L,L] DIRECT STORE  — host wiring + the patched assert
# =============================================================================
#
# The GEMM writes its output D as row-major (M, N) = (B*L*L, d). Naturally that
# reshapes to [B, L, L, d]. The bmm contraction wants [B, d, L, L]. Instead of a
# post-hoc permute().contiguous() (a full extra read+write — the bottleneck),
# you hand the kernel an OUTPUT TENSOR whose strides already describe [B,d,L,L].
#
# HOST side (this is exactly launch.py:79-96, "bdll_direct"):
#
#   left_bdll = torch.empty(B, d, L, L, ...)        # the real storage
#   # M-major view: shape (M=L*L, N=d), strides (1, L*L)
#   left_view = left_bdll.view(d, L * L).T          # <-- pass THIS as PostAct
#   gemm_act(A=x_flat, B=B_left, activation="glu", postact_out=left_view)
#
# Why it works: the GEMM stores logical element (m, n) at
#     base + m*stride_m + n*stride_n  =  base + m*1 + n*(L*L)
# m = flattened (row,col) of the LxL block  → consecutive m fill one L*L plane
# n = channel d                             → each channel is one L*L plane apart
# == the memory layout of contiguous [d, L, L]. No transpose kernel needed.
#
# WHAT QUACK NEEDED PATCHING for this:
#   GemmGatedMixin.epi_to_underlying_arguments (gemm_act.py:202-203) asserts the
#   postact tensor is "n-major-c":
#       assert self.d_layout is None or self.d_layout.is_n_major_c()
#       assert LayoutEnum.from_tensor(args.mPostAct).is_n_major_c()
#   Our bdll view is M-MAJOR (strides (1, L*L), i.e. the contiguous dim is M, not
#   N). So the stock assert rejects it. The patch (see module.py docstring
#   "Requires the patched quack") drops/relaxes that n-major assert on postact and
#   lets pa_leading_dim follow the detected layout (compare gemm_act.py:306,321
#   `pa_leading` / `pa_leading_dim`). TMA store atom is built from the tensor's
#   real strides at :458-478, so once the assert is gone the store just works.
#
# So the bdll trick is HOST-side (a strided view) + ONE relaxed assert. The
# device epilogue store path (copy_postact at gemm_act.py:102-110 / used in
# epilogue() gemm_sm90.py:1339) is unchanged — it writes through whatever TMA
# atom the host built from the view's strides.


# =============================================================================
# 3. SO, to build your combined front kernel (left+right+gate), the cute-level
#    plan is:
# =============================================================================
#   - left+right: ONE "gated" GEMM. Host B operand = concat of the two
#     interleaved [gate|proj] weights -> (d, 4d). PostAct = bdll view as above,
#     but now width 2d (left and right side by side). Split after.
#       * caveat: bdll view math for a 2d-wide postact: storage [B, 2d, L, L],
#         view = storage.view(2d, L*L).T. left = [:, :d] planes, right = [:, d:].
#   - gate: a separate "act" GEMM (activation="sigmoid"), normal [B,L,L,d] out.
#   - the heavy lifting (TMA/pipeline/WGMMA) you REUSE from quack by calling
#     gemm_act/gemm_gated — you only choose tile_M/tile_N, the interleaved B, and
#     the postact view. You do NOT rewrite gemm_sm90.kernel.
#
# If you want to OWN the kernel end-to-end later, fork GemmGatedSm90 and override
# just epi_visit_subtile (2a) — that's the only part that's "yours".
