
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared, warpgroup_mma, warpgroup_mma_wait, warpgroup_mma_init,
)

LOG2E = gl.constexpr(1.4426950408889634)
MSEL = gl.constexpr(-1.0e9)          # per-key select level for masked keys (rendered per dtype: -1e9 bf16, -32768 fp16)
BM = gl.constexpr(128)           # query rows per CTA (two consumer warpgroups x 64)
ROWS = gl.constexpr(3)


@gluon.jit
def _keep(x):
    """Zero-instruction use of x (tied in/out registers): extends x's register lifetime to this point, so ptxas sees no
    write-after-read hazard between the asynchronous wgmma still reading x and later values that would reuse its registers."""
    return gl.inline_asm_elementwise("", "=r,0", [x], dtype=x.dtype, is_pure=False, pack=2)


@gluon.jit
def _phase_a(acc, m_i, l_i, s, b32, mrow, m_lo, offs_n, kvm, qk_scale, USE_MASK: gl.constexpr, O_L: gl.constexpr):
    """Row phase A (FMA pipe): logits t = scale*s + bias (natural units; keys masked for every row already carry -1e9 in the
    bias tile, and with USE_MASK this row's own mask is applied per key), running max, rescale of the running output/sum by
    alpha = 2^((m_old - m_new) log2 e).  Returns t and nml = -m_new*log2e for phase B."""
    t = s * qk_scale + b32
    if USE_MASK:
        mk = gl.load(mrow + m_lo + offs_n, mask=kvm, other=1)
        t = gl.where(gl.expand_dims(mk, 0) != 0, t, MSEL)
    m_new = gl.maximum(m_i, gl.max(t, axis=1))
    alpha = gl.exp2((m_i - m_new) * LOG2E)
    nml = m_new * (-LOG2E)
    alpha_o = gl.convert_layout(alpha, gl.SliceLayout(1, O_L), assert_trivial=True)
    acc = acc * gl.expand_dims(alpha_o, 1)
    return acc, m_new, l_i * alpha, t, nml


@gluon.jit
def _phase_b(acc, l_i, t, nml, v_view, P_L: gl.constexpr):
    """Row phase B (MUFU pipe): p = 2^(t log2e - m log2e), row-sum, P -> bf16, asynchronous PV wgmma.  Returns the fp32 P
    (dead: register donor for the next QK^T accumulator) and the bf16 P (must stay live until the PV wgmma retires)."""
    p = gl.exp2(t * LOG2E + gl.expand_dims(nml, 1))
    l_new = l_i + gl.sum(p, axis=1)
    p16 = gl.convert_layout(p.to(v_view.dtype), P_L, assert_trivial=True)
    acc = warpgroup_mma(p16, v_view, acc, is_async=True)
    return acc, l_new, p16, p



KXD = gl.constexpr(16)
ROWS_P2 = gl.constexpr(4)          # rows padded to a power of two (the visible-tiles scan holds one row per warp)


@gluon.jit
def _stamp(Dbg, slot, pid, it, k: gl.constexpr, L: gl.constexpr):
    """DBG&8: write %globaltimer (ns) into Dbg[pid, it, k] for the first 16 items of each CTA."""
    z = gl.full([1], 0, gl.int32, L)
    t = gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [z], dtype=gl.int64, is_pure=False, pack=1)
    gl.store(Dbg + (pid * 16 + it) * 8 + k + gl.arange(0, 1, layout=L), t, mask=(gl.arange(0, 1, layout=L) == 0) & (it < 16))


@gluon.jit
def _fdiv(a, d, inv_d):
    """a // d for 0 <= a < 2^22 via the fp32 reciprocal (exact; avoids the ~50-instruction integer division sequence)."""
    qt = ((a.to(gl.float32) + 0.5) * inv_d).to(gl.int32)
    return qt, a - qt * d


@gluon.jit
def _item(w, n_q, n_r, H, inv_nq, inv_nr, inv_h, N_ROWS, SEQ_Q, SEQ_K):
    """Work item w -> (b, h, q0, r0); q tiles vary fastest so concurrently running CTAs share the K/V rows in L2."""
    t, pid_q = _fdiv(w, n_q, inv_nq)
    pid_bh, pid_r = _fdiv(t, n_r, inv_nr)
    b, h = _fdiv(pid_bh, H, inv_h)
    return b, h, pid_bh, pid_q * BM, pid_r * ROWS


@gluon.jit
def _visible_tiles(mask_b, smi, SEQ_K, n_tiles, i0, i1, i2, BLOCK_N: gl.constexpr, V_L: gl.constexpr, SCW: gl.constexpr):
    """Key tiles this item visits: up to the last key kept by any of its rows; all of them if a row keeps nothing.  One
    [ROWS_P2, SCW] load per SCW keys with each pair row on its own warp (row reductions are warp shuffles)."""
    VR_L: gl.constexpr = gl.SliceLayout(1, V_L)
    VC_L: gl.constexpr = gl.SliceLayout(0, V_L)
    ridx = gl.arange(0, ROWS_P2, layout=VR_L)
    rptr = mask_b + i0 * smi + ridx * 0
    rptr = gl.where(ridx >= 1, mask_b + i1 * smi, rptr)
    rptr = gl.where(ridx >= 2, mask_b + i2 * smi, rptr)
    offs = gl.arange(0, SCW, layout=VC_L)
    last_k = gl.full([ROWS_P2], -1, gl.int32, VR_L)
    for k0 in range(0, SEQ_K, SCW):
        cols = gl.expand_dims(k0 + offs, 0)
        c_ok = (cols < SEQ_K) & (gl.expand_dims(ridx, 1) >= 0)
        mk = gl.load(gl.expand_dims(rptr, 1) + cols, mask=c_ok, other=0) != 0
        last_k = gl.maximum(last_k, gl.max(gl.where(mk, cols, -1), axis=1))
    keeps = gl.min(last_k, axis=0) >= 0                       # every row keeps some key
    n_vis = gl.where(keeps, gl.max(last_k, axis=0) // BLOCK_N + 1, n_tiles)
    return n_vis


@gluon.jit
def _producer(q_desc, k_desc, v_desc, b_desc, q_smem, k_smem, v_smem, b_smem, kx_smem, nvis_smem, ready, empty, q_ready, q_empty,
              Mask, smb, smi, N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, X0, X1, X2, XR,
              QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH,
              BLOCK_N: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr, KXW: gl.constexpr, DBG: gl.constexpr):
    """TMA producer (default partition, 4 warps).  Per item: the ROWS Q tiles into Q buffer it % 2, then per key tile ROWS K +
    ROWS V tiles + one [128, BLOCK_N] bias tile into the ring (which never drains between items); with a mask it also writes
    the per-row [BLOCK_N, 16] additive tiles (k-slot pieces X0..XR for masked keys, 0 for kept keys, from the mask bytes) into the stage and arrives on `ready`."""
    KV_BYTES: gl.constexpr = k_desc.block_type.nbytes
    B_BYTES: gl.constexpr = b_desc.block_type.nbytes
    Q_BYTES: gl.constexpr = q_desc.block_type.nbytes
    PNW: gl.constexpr = gl.num_warps()
    KX_L: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [PNW, 1], [1, 0])
    KROW_L: gl.constexpr = gl.SliceLayout(1, KX_L)
    KCOL_L: gl.constexpr = gl.SliceLayout(0, KX_L)
    SCW: gl.constexpr = 512
    SCL: gl.constexpr = gl.BlockedLayout([1, SCW // 32], [1, 32], [PNW, 1], [1, 0])     # pair rows on warps, keys on lanes
    ONE_L: gl.constexpr = gl.BlockedLayout([1], [32], [PNW], [0])
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    offs_k = gl.arange(0, BLOCK_N, layout=KROW_L)
    colj = gl.expand_dims(gl.arange(0, KXD, layout=KCOL_L), 0)
    xcol = gl.where(colj == 0, X0, gl.where(colj == 1, X1, gl.where(colj == 2, X2, gl.where(colj < KXW, XR, 0.0))))   # [1, KXD] additive pieces per k-slot
    mk0 = gl.full([BLOCK_N], 1, gl.uint8, KROW_L)
    mk1 = gl.full([BLOCK_N], 1, gl.uint8, KROW_L)
    mk2 = gl.full([BLOCK_N], 1, gl.uint8, KROW_L)
    st = 0
    eph = 1          # parity to wait for on `empty[st]` before refilling stage st (first lap passes via pred)
    g = 0            # key tiles loaded so far (ring position)
    it = 0           # items started so far
    for w in range(pid, n_items, nprog):
        b, h, pid_bh, q0, r0 = _item(w, n_q, n_r, H, inv_nq, inv_nr, inv_h, N_ROWS, SEQ_Q, SEQ_K)
        i0 = gl.minimum(r0 + 0, N_ROWS - 1)
        ii0 = i0          # int32 copy for the TMA coordinate terms (computed inline at each load: nothing stays live)
        i1 = gl.minimum(r0 + 1, N_ROWS - 1)
        ii1 = i1          # int32 copy for the TMA coordinate terms (computed inline at each load: nothing stays live)
        i2 = gl.minimum(r0 + 2, N_ROWS - 1)
        ii2 = i2          # int32 copy for the TMA coordinate terms (computed inline at each load: nothing stays live)
        i0 = i0.to(gl.int64)
        i1 = i1.to(gl.int64)
        i2 = i2.to(gl.int64)
        n_tiles = gl.cdiv(SEQ_K, BLOCK_N)
        mask_b = Mask + b.to(gl.int64) * smb
        qb = it % 2
        # Q buffer qb was last used by item it-2: wait until the consumers released it, then load this item's Q tiles
        mbarrier.wait(q_empty.index(qb), ((it // 2) + 1) & 1, pred=it >= 2)
        qrb = q_ready.index(qb)
        mbarrier.expect(qrb, ROWS * Q_BYTES)
        tma.async_copy_global_to_shared(q_desc, [b * QRB + ii0 * QRI + h * QRH + q0, b * QCB + ii0 * QCI + h * QCH], qrb, q_smem.index(qb * ROWS + 0))
        tma.async_copy_global_to_shared(q_desc, [b * QRB + ii1 * QRI + h * QRH + q0, b * QCB + ii1 * QCI + h * QCH], qrb, q_smem.index(qb * ROWS + 1))
        tma.async_copy_global_to_shared(q_desc, [b * QRB + ii2 * QRI + h * QRH + q0, b * QCB + ii2 * QCI + h * QCH], qrb, q_smem.index(qb * ROWS + 2))
        if HAS_MASK:
            # while the Q tiles fly: the visible key tiles of this item -> the consumers read it from nvis_smem[qb] after
            # q_ready[qb] (released by the second arrive); the first tile's mask bytes are fetched now, later tiles one tile ahead
            if (DBG & 4) == 0:
                n_tiles = _visible_tiles(mask_b, smi, SEQ_K, n_tiles, i0, i1, i2, BLOCK_N, SCL, SCW)
            nvis_smem.index(qb).store(gl.full([1], n_tiles, gl.int32, ONE_L))
            mbarrier.arrive(qrb, count=1)
            mk0 = gl.load(mask_b + i0 * smi + offs_k, mask=offs_k < SEQ_K, other=1)
            mk1 = gl.load(mask_b + i1 * smi + offs_k, mask=offs_k < SEQ_K, other=1)
            mk2 = gl.load(mask_b + i2 * smi + offs_k, mask=offs_k < SEQ_K, other=1)
        bias_row = pid_bh * SEQ_Q + q0
        for n in range(n_tiles):
            mbarrier.wait(empty.index(st), eph, pred=g >= STAGES)
            rb = ready.index(st)
            mbarrier.expect(rb, 2 * ROWS * KV_BYTES + B_BYTES)
            n0 = n * BLOCK_N
            tma.async_copy_global_to_shared(k_desc, [b * KRB + ii0 * KRI + h * KRH + n0, b * KCB + ii0 * KCI + h * KCH], rb, k_smem.index(st * ROWS + 0))
            tma.async_copy_global_to_shared(v_desc, [b * VRB + ii0 * VRI + h * VRH + n0, b * VCB + ii0 * VCI + h * VCH], rb, v_smem.index(st * ROWS + 0))
            tma.async_copy_global_to_shared(k_desc, [b * KRB + ii1 * KRI + h * KRH + n0, b * KCB + ii1 * KCI + h * KCH], rb, k_smem.index(st * ROWS + 1))
            tma.async_copy_global_to_shared(v_desc, [b * VRB + ii1 * VRI + h * VRH + n0, b * VCB + ii1 * VCI + h * VCH], rb, v_smem.index(st * ROWS + 1))
            tma.async_copy_global_to_shared(k_desc, [b * KRB + ii2 * KRI + h * KRH + n0, b * KCB + ii2 * KCI + h * KCH], rb, k_smem.index(st * ROWS + 2))
            tma.async_copy_global_to_shared(v_desc, [b * VRB + ii2 * VRI + h * VRH + n0, b * VCB + ii2 * VCI + h * VCH], rb, v_smem.index(st * ROWS + 2))
            tma.async_copy_global_to_shared(b_desc, [bias_row, n0], rb, b_smem.index(st))
            if HAS_MASK:
                # this tile's additive mask tiles from the prefetched bytes; then prefetch the next tile's bytes
                if (DBG & 2) == 0:
                    kx_smem.index(st * ROWS + 0).store(gl.where(gl.expand_dims(mk0 == 0, 1), xcol, 0.0).to(k_desc.dtype))
                    kx_smem.index(st * ROWS + 1).store(gl.where(gl.expand_dims(mk1 == 0, 1), xcol, 0.0).to(k_desc.dtype))
                    kx_smem.index(st * ROWS + 2).store(gl.where(gl.expand_dims(mk2 == 0, 1), xcol, 0.0).to(k_desc.dtype))
                    fence_async_shared()
                mbarrier.arrive(rb, count=1)
                k_nx = n0 + BLOCK_N + offs_k
                k_ok = k_nx < SEQ_K
                mk0 = gl.load(mask_b + i0 * smi + k_nx, mask=k_ok, other=1)
                mk1 = gl.load(mask_b + i1 * smi + k_nx, mask=k_ok, other=1)
                mk2 = gl.load(mask_b + i2 * smi + k_nx, mask=k_ok, other=1)
            g += 1
            st += 1
            if st == STAGES:
                st = 0
                eph ^= 1
        it += 1


@gluon.jit
def _consumer(q_smem, k_smem, v_smem, b_smem, kx_smem, q1_smem, nvis_smem, o_smem, o_desc, ready, empty, q_ready, q_empty, Out, Mask, Dbg,
              sob, soi, soh, soq, smb, smi,
              N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, qk_scale,
              BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr, DBG: gl.constexpr,
              EVEN_Q: gl.constexpr, KXW: gl.constexpr, AONE: gl.constexpr):
    """The compute partition (8 warps = two warpgroups x 64 query rows, one code stream), persistent over work items; per
    item the k11 schedule (software pipeline over the ROWS pair rows of each key tile, dead-P register donors, explicit P
    liveness); with a mask every QK^T wgmma is followed by the chained rank-1 mask wgmma on the same accumulator."""
    NW: gl.constexpr = gl.num_warps()
    PM: gl.constexpr = 128
    WG: gl.constexpr = 0
    SWAP: gl.constexpr = 0
    PINGPONG: gl.constexpr = 0
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, BLOCK_N, 16])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, HEAD_DIM, 16])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
    ROW_L: gl.constexpr = gl.SliceLayout(1, S_L)
    COL_L: gl.constexpr = gl.SliceLayout(0, S_L)
    OROW_L: gl.constexpr = gl.SliceLayout(1, O_L)
    OCOL_L: gl.constexpr = gl.SliceLayout(0, O_L)
    Q1_L: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [NW, 1], [1, 0])
    ONE_L: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    dt: gl.constexpr = v_smem.dtype
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)

    if HAS_MASK:
        # the constant A operand of the rank-1 mask wgmma: column 0 = 1, columns 1..15 = 0
        q1c = gl.expand_dims(gl.arange(0, KXD, layout=gl.SliceLayout(0, Q1_L)), 0) < KXW
        q1r = gl.expand_dims(gl.arange(0, PM, layout=gl.SliceLayout(1, Q1_L)), 1) >= 0
        q1_smem.store(gl.where(q1c & q1r, AONE, 0.0).to(dt))
        fence_async_shared()          # (the partition-wide bar.sync before the first wgmma read is inserted by the membar pass)
    offs_n = gl.arange(0, BLOCK_N, layout=COL_L)
    pd = gl.zeros([PM, BLOCK_N], gl.float32, S_L)          # dead fp32 P tile: register donor for the next QK^T accumulator
    p16a = gl.zeros([PM, BLOCK_N], dt, P_L)                # P operands of the two most recent PV wgmmas (kept live until retired)
    p16b = gl.zeros([PM, BLOCK_N], dt, P_L)
    t_pv = gl.zeros([PM, BLOCK_N], gl.float32, S_L)        # phase-A output of the previous row, consumed by its phase B
    nml_pv = gl.zeros([PM], gl.float32, ROW_L)
    st = 0           # ring stage / parity of the next key tile (continues across items)
    ph = 0
    it = 0
    for w in range(pid, n_items, nprog):
        b, h, pid_bh, q0, r0 = _item(w, n_q, n_r, H, inv_nq, inv_nr, inv_h, N_ROWS, SEQ_Q, SEQ_K)
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 0, ONE_L)
        i0 = gl.minimum(r0 + 0, N_ROWS - 1)
        orow0 = ((b * N_ROWS + i0) * H + h) * SEQ_Q
        i0 = i0.to(gl.int64)
        i1 = gl.minimum(r0 + 1, N_ROWS - 1)
        orow1 = ((b * N_ROWS + i1) * H + h) * SEQ_Q
        i1 = i1.to(gl.int64)
        i2 = gl.minimum(r0 + 2, N_ROWS - 1)
        orow2 = ((b * N_ROWS + i2) * H + h) * SEQ_Q
        i2 = i2.to(gl.int64)
        n_tiles = gl.cdiv(SEQ_K, BLOCK_N)
        mask_b = Mask + b.to(gl.int64) * smb
        out_bh = Out + b.to(gl.int64) * sob + h.to(gl.int64) * soh
        qb = it % 2
        mbarrier.wait(q_ready.index(qb), (it // 2) & 1)          # completion it//2 of q_ready[qb]
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 1, ONE_L)
        if HAS_MASK:
            n_tiles = gl.max(nvis_smem.index(qb).load(ONE_L), axis=0)     # visible key tiles (written by the producer)
        n_sel = n_tiles
        qs0 = q_smem.index(qb * ROWS + 0)
        acc0 = warpgroup_mma_init(gl.zeros([PM, HEAD_DIM], gl.float32, O_L))
        m0 = gl.full([PM], float('-inf'), gl.float32, ROW_L)
        l0 = gl.zeros([PM], gl.float32, ROW_L)
        qs1 = q_smem.index(qb * ROWS + 1)
        acc1 = warpgroup_mma_init(gl.zeros([PM, HEAD_DIM], gl.float32, O_L))
        m1 = gl.full([PM], float('-inf'), gl.float32, ROW_L)
        l1 = gl.zeros([PM], gl.float32, ROW_L)
        qs2 = q_smem.index(qb * ROWS + 2)
        acc2 = warpgroup_mma_init(gl.zeros([PM, HEAD_DIM], gl.float32, O_L))
        m2 = gl.full([PM], float('-inf'), gl.float32, ROW_L)
        l2 = gl.zeros([PM], gl.float32, ROW_L)
        # prologue: QK^T of row 0, tile 0 of this item (ring position continues from the previous item)
        mbarrier.wait(ready.index(st), ph)
        s_nx = warpgroup_mma(qs0, k_smem.index(st * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)
        if HAS_MASK and (DBG & 1) == 0:
            s_nx = warpgroup_mma(q1_smem, kx_smem.index(st * ROWS).permute((1, 0)), s_nx, is_async=True)
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 2, ONE_L)
        for n in range(0, n_sel):
            st1 = st + 1
            ph1 = ph
            if st1 == STAGES:
                st1 = 0
                ph1 = ph ^ 1
            if STAGES == 2:
                stp = st1
            else:
                stp = (st + STAGES - 1) % STAGES
            m_lo = n * BLOCK_N
            b32 = b_smem.index(st).load(S_L).to(gl.float32)
            s_cu = s_nx
            s_cu, acc0 = warpgroup_mma_wait(num_outstanding=1, deps=[s_cu, acc0])
            p16a = _keep(p16a)
            s_nx = warpgroup_mma(qs1, k_smem.index(st * ROWS + 1).permute((1, 0)), pd, use_acc=False, is_async=True)
            if HAS_MASK and (DBG & 1) == 0:
                s_nx = warpgroup_mma(q1_smem, kx_smem.index(st * ROWS + 1).permute((1, 0)), s_nx, is_async=True)
            if SWAP:
                if PINGPONG:
                    mbarrier.wait(tok_mine, tph)
                if n > 0:
                    acc2, l2, p16n, pd = _phase_b(acc2, l2, t_pv, nml_pv, v_smem.index(stp * ROWS + 2), P_L=P_L)
                    p16a = p16b
                    p16b = p16n
                if PINGPONG:
                    mbarrier.arrive(tok_other, count=1)
                    tph = tph ^ 1
                acc0, m0, l0, t_new, nml_new = _phase_a(acc0, m0, l0, s_cu, b32, mask_b, m_lo, offs_n, offs_n, qk_scale, USE_MASK=False, O_L=O_L)
            else:
                acc0, m0, l0, t_new, nml_new = _phase_a(acc0, m0, l0, s_cu, b32, mask_b, m_lo, offs_n, offs_n, qk_scale, USE_MASK=False, O_L=O_L)
                if PINGPONG:
                    mbarrier.wait(tok_mine, tph)
                if n > 0:
                    acc2, l2, p16n, pd = _phase_b(acc2, l2, t_pv, nml_pv, v_smem.index(stp * ROWS + 2), P_L=P_L)
                    p16a = p16b
                    p16b = p16n
                if PINGPONG:
                    mbarrier.arrive(tok_other, count=1)
                    tph = tph ^ 1
            t_pv = t_new
            nml_pv = nml_new
            s_cu = s_nx
            s_cu, acc1 = warpgroup_mma_wait(num_outstanding=1, deps=[s_cu, acc1])
            p16a = _keep(p16a)
            s_nx = warpgroup_mma(qs2, k_smem.index(st * ROWS + 2).permute((1, 0)), pd, use_acc=False, is_async=True)
            if HAS_MASK and (DBG & 1) == 0:
                s_nx = warpgroup_mma(q1_smem, kx_smem.index(st * ROWS + 2).permute((1, 0)), s_nx, is_async=True)
            if SWAP:
                if PINGPONG:
                    mbarrier.wait(tok_mine, tph)
                acc0, l0, p16n, pd = _phase_b(acc0, l0, t_pv, nml_pv, v_smem.index(st * ROWS + 0), P_L=P_L)
                p16a = p16b
                p16b = p16n
                if PINGPONG:
                    mbarrier.arrive(tok_other, count=1)
                    tph = tph ^ 1
                acc1, m1, l1, t_new, nml_new = _phase_a(acc1, m1, l1, s_cu, b32, mask_b, m_lo, offs_n, offs_n, qk_scale, USE_MASK=False, O_L=O_L)
            else:
                acc1, m1, l1, t_new, nml_new = _phase_a(acc1, m1, l1, s_cu, b32, mask_b, m_lo, offs_n, offs_n, qk_scale, USE_MASK=False, O_L=O_L)
                if PINGPONG:
                    mbarrier.wait(tok_mine, tph)
                acc0, l0, p16n, pd = _phase_b(acc0, l0, t_pv, nml_pv, v_smem.index(st * ROWS + 0), P_L=P_L)
                p16a = p16b
                p16b = p16n
                if PINGPONG:
                    mbarrier.arrive(tok_other, count=1)
                    tph = tph ^ 1
            t_pv = t_new
            nml_pv = nml_new
            s_cu = s_nx
            s_cu, acc2 = warpgroup_mma_wait(num_outstanding=1, deps=[s_cu, acc2])
            p16a = _keep(p16a)
            # every wgmma reading the oldest live K/V/bias stage has retired -> release it (one elected arrive)
            mbarrier.arrive(empty.index(stp), count=1, pred=n > 0)
            # (for the last tile this reads a stale stage; drained after the loop, never used)
            mbarrier.wait(ready.index(st1), ph1, pred=(n + 1) < n_tiles)
            s_nx = warpgroup_mma(qs0, k_smem.index(st1 * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)
            if HAS_MASK and (DBG & 1) == 0:
                s_nx = warpgroup_mma(q1_smem, kx_smem.index(st1 * ROWS).permute((1, 0)), s_nx, is_async=True)
            if SWAP:
                if PINGPONG:
                    mbarrier.wait(tok_mine, tph)
                acc1, l1, p16n, pd = _phase_b(acc1, l1, t_pv, nml_pv, v_smem.index(st * ROWS + 1), P_L=P_L)
                p16a = p16b
                p16b = p16n
                if PINGPONG:
                    mbarrier.arrive(tok_other, count=1)
                    tph = tph ^ 1
                acc2, m2, l2, t_new, nml_new = _phase_a(acc2, m2, l2, s_cu, b32, mask_b, m_lo, offs_n, offs_n, qk_scale, USE_MASK=False, O_L=O_L)
            else:
                acc2, m2, l2, t_new, nml_new = _phase_a(acc2, m2, l2, s_cu, b32, mask_b, m_lo, offs_n, offs_n, qk_scale, USE_MASK=False, O_L=O_L)
                if PINGPONG:
                    mbarrier.wait(tok_mine, tph)
                acc1, l1, p16n, pd = _phase_b(acc1, l1, t_pv, nml_pv, v_smem.index(st * ROWS + 1), P_L=P_L)
                p16a = p16b
                p16b = p16n
                if PINGPONG:
                    mbarrier.arrive(tok_other, count=1)
                    tph = tph ^ 1
            t_pv = t_new
            nml_pv = nml_new
            acc2 = warpgroup_mma_init(acc2)
            st = st1
            ph = ph1
            if DBG & 8:
                if n == 0:
                    _stamp(Dbg, 0, pid, it, 3, ONE_L)       # end of the first tile
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 4, ONE_L)
        if STAGES == 2:
            stl = st ^ 1
        else:
            stl = (st + STAGES - 1) % STAGES
        acc2, l2, p16n, pd = _phase_b(acc2, l2, t_pv, nml_pv, v_smem.index(stl * ROWS + 2), P_L=P_L)
        acc0, acc1, acc2, s_nx = warpgroup_mma_wait(num_outstanding=0, deps=[acc0, acc1, acc2, s_nx])
        p16a = _keep(p16a)
        p16b = _keep(p16b)
        p16n = _keep(p16n)
        if EVEN_Q:
            tma.store_wait(0)              # the previous item's output tiles have left o_smem
            l_o = gl.convert_layout(l0, OROW_L, assert_trivial=True)
            o_smem.index(0).store((acc0 * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty))
            l_o = gl.convert_layout(l1, OROW_L, assert_trivial=True)
            o_smem.index(1).store((acc1 * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty))
            l_o = gl.convert_layout(l2, OROW_L, assert_trivial=True)
            o_smem.index(2).store((acc2 * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty))
            fence_async_shared()
            tma.async_copy_shared_to_global(o_desc, [orow0 + q0, 0], o_smem.index(0))   # rows past N_ROWS repeat row N-1 (identical bytes)
            tma.async_copy_shared_to_global(o_desc, [orow1 + q0, 0], o_smem.index(1))   # rows past N_ROWS repeat row N-1 (identical bytes)
            tma.async_copy_shared_to_global(o_desc, [orow2 + q0, 0], o_smem.index(2))   # rows past N_ROWS repeat row N-1 (identical bytes)
        else:
            offs_q = q0 + WG * 64 + gl.arange(0, PM, layout=OROW_L)
            offs_d = gl.arange(0, HEAD_DIM, layout=OCOL_L)
            o_ptr = out_bh + gl.expand_dims(offs_q.to(gl.int64) * soq, 1) + gl.expand_dims(offs_d, 0)
            q_ok = gl.expand_dims(offs_q < SEQ_Q, 1)
            l_o = gl.convert_layout(l0, OROW_L, assert_trivial=True)
            o = (acc0 * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty)
            gl.store(o_ptr + i0 * soi, o, mask=q_ok)
            l_o = gl.convert_layout(l1, OROW_L, assert_trivial=True)
            o = (acc1 * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty)
            gl.store(o_ptr + i1 * soi, o, mask=q_ok & ((r0 + 1) < N_ROWS))
            l_o = gl.convert_layout(l2, OROW_L, assert_trivial=True)
            o = (acc2 * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty)
            gl.store(o_ptr + i2 * soi, o, mask=q_ok & ((r0 + 2) < N_ROWS))
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 5, ONE_L)
        # everything has retired: release the ring stage(s) this item still holds (its last tile; the last two when ROWS == 2)
        # and its Q buffer, so the producer keeps streaming the next items
        mbarrier.arrive(empty.index(stl), count=1)
        if ROWS == 2:
            mbarrier.arrive(empty.index((stl + STAGES - 1) % STAGES), count=1, pred=n_tiles >= 2)
        mbarrier.arrive(q_empty.index(qb), count=1)
        it += 1
    if EVEN_Q:
        tma.store_wait(0)


@gluon.jit
def _fwd(q_desc, k_desc, v_desc, b_desc, o_desc, Out, Mask, Dbg,
         sob, soi, soh, soq, smb, smi,
         N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, qk_scale, X0, X1, X2, XR,
         QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH,
         BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr,
         REGS_CONS: gl.constexpr, KXW: gl.constexpr, AONE: gl.constexpr, DBG: gl.constexpr, EVEN_Q: gl.constexpr):
    KX_SL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, KXD], k_desc.dtype)
    Q1_SL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, KXD], k_desc.dtype)
    q_smem = gl.allocate_shared_memory(q_desc.dtype, [2 * ROWS, BM, HEAD_DIM], q_desc.layout)      # double-buffered per item
    k_smem = gl.allocate_shared_memory(k_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], k_desc.layout)
    v_smem = gl.allocate_shared_memory(v_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], v_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [STAGES, BM, BLOCK_N], b_desc.layout)
    kx_smem = gl.allocate_shared_memory(k_desc.dtype, [STAGES * ROWS, BLOCK_N, KXD], KX_SL)
    q1_smem = gl.allocate_shared_memory(k_desc.dtype, [BM, KXD], Q1_SL)
    nvis_smem = gl.allocate_shared_memory(gl.int32, [2, 1], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    o_smem = gl.allocate_shared_memory(o_desc.dtype, [ROWS, BM, HEAD_DIM], o_desc.layout)              # output staging for the TMA-store epilogue
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    q_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        if HAS_MASK:
            mbarrier.init(ready.index(s), count=2)          # TMA transaction bytes + the producer's arrive after its mask tiles
        else:
            mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    for s in gl.static_range(2):
        if HAS_MASK:
            mbarrier.init(q_ready.index(s), count=2)        # Q transaction bytes + the producer's arrive after the visible-tiles word
        else:
            mbarrier.init(q_ready.index(s), count=1)
        mbarrier.init(q_empty.index(s), count=1)
    fence_async_shared()
    gl.warp_specialize([
        (_producer, (q_desc, k_desc, v_desc, b_desc, q_smem, k_smem, v_smem, b_smem, kx_smem, nvis_smem, ready, empty, q_ready, q_empty,
                     Mask, smb, smi, N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, X0, X1, X2, XR,
                     QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH, BLOCK_N, STAGES, HAS_MASK, KXW, DBG)),
        (_consumer, (q_smem, k_smem, v_smem, b_smem, kx_smem, q1_smem, nvis_smem, o_smem, o_desc, ready, empty, q_ready, q_empty, Out, Mask, Dbg,
                     sob, soi, soh, soq, smb, smi,
                     N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, qk_scale,
                     BLOCK_N, HEAD_DIM, STAGES, HAS_MASK, DBG, EVEN_Q, KXW, AONE)),
    ], [8], [REGS_CONS])
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(s))
        mbarrier.invalidate(empty.index(s))
    for s in gl.static_range(2):
        mbarrier.invalidate(q_ready.index(s))
        mbarrier.invalidate(q_empty.index(s))
