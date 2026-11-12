"""64-bit address arithmetic of the H100 (sm_90a) glue kernels: the wide Transition's LayerNorm forward / backward and weight preparation
(``transition_fused_sm90a.cu``, ``MW_TRANSITION_AUX_ONLY``) and the TriMul live aux unit (``wide_forward/output.cu``, ``MW_AUX_ONLY``).

``big`` tests run each kernel on more than 2**31 elements. Rows (or mask blocks) are independent, so the input is a small block tiled R
times and every tile of the output must equal the small block's output bit for bit (an int32 offset wraps the tail tiles: an illegal
address or a wrong / unwritten tile). The affine gradients are R times the small block's, up to summation order. ``small`` tests check
every changed kernel against a PyTorch reference at served sizes (also the compute-sanitizer run: ``-k small``).
"""

import pytest
import torch

pytestmark = [pytest.mark.gpu]

BIG = 2**31


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


needs_h100 = pytest.mark.skipif(not _h100(), reason="H100 (sm_90a) kernels")


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _bf(shape, seed=0) -> torch.Tensor:
    return torch.randn(shape, device="cuda", generator=_gen(seed)).to(torch.bfloat16)


def _tiles(rows_small: int, d: int) -> int:
    """Tiles of ``rows_small`` rows so that the tiled tensor passes 2**31 elements by at least one tile."""
    return BIG // (rows_small * d) + 2


def _free():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ---- Transition (wide sm_90a LayerNorm + weight preparation) ------------------------------------------------------------------------


def _tr():
    from miniworld_engine.kernels.transition.cuda.fused_sm90a import _native_ext

    return _native_ext()


def _ln_inputs(rows: int, d: int, seed: int = 0):
    x, dy, res = _bf((rows, d), seed), _bf((rows, d), seed + 1), _bf((rows, d), seed + 2)
    gamma = torch.randn(d, device="cuda", generator=_gen(seed + 3)) * 0.5 + 1
    xf = x.float()
    mean, rs = xf.mean(-1), torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5)
    return dy, x, gamma, rs.contiguous(), (mean * rs).contiguous(), res


def _ln_bwd_ref(dy, x, gamma, rs, c1, res):
    xh = x.float() * rs[:, None] - c1[:, None]
    g = dy.float()
    wd = g * gamma
    a = (wd * xh).mean(-1, keepdim=True)
    b = wd.mean(-1, keepdim=True)
    dx = (wd - (xh * a + b)) * rs[:, None] + res.float()
    return dx, (g * xh).sum(0), g.sum(0)


LN_BWD = [("ln_bwd", 384, 4), ("ln_bwd", 512, 4), ("ln_bwd", 768, 3), ("ln_bwd_compact", 384, 5), ("ln_bwd_compact", 512, 5)]


@needs_h100
@pytest.mark.parametrize("fn,d,waves", LN_BWD)
def test_small_transition_ln_bwd(fn, d, waves):
    args = _ln_inputs(4096 + 37, d)
    dx, dg, db = getattr(_tr(), fn)(*args, waves)
    rdx, rdg, rdb = _ln_bwd_ref(*args)
    torch.testing.assert_close(dx.float(), rdx, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(dg, rdg, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(db, rdb, atol=1e-2, rtol=1e-3)


@needs_h100
@pytest.mark.slow
@pytest.mark.parametrize("fn,d,waves", LN_BWD)
def test_big_transition_ln_bwd(fn, d, waves):
    s = 4096
    small = _ln_inputs(s, d)
    dx_s, dg_s, db_s = getattr(_tr(), fn)(*small, waves)
    r = _tiles(s, d)
    dy, x, gamma, rs, c1, res = small
    big = (dy.repeat(r, 1), x.repeat(r, 1), gamma, rs.repeat(r), c1.repeat(r), res.repeat(r, 1))
    del small, dy, x, res
    assert big[1].numel() > BIG
    dx, dg, db = getattr(_tr(), fn)(*big, waves)
    del big
    dx = dx.view(r, s, d)
    for t in (0, r // 2, r - 2, r - 1):
        assert torch.equal(dx[t], dx_s), f"tile {t} of {r}"
    del dx
    _free()
    torch.testing.assert_close(dg, dg_s * r, rtol=1e-4, atol=1e-2 * r)
    torch.testing.assert_close(db, db_s * r, rtol=1e-4, atol=1e-2 * r)


@needs_h100
@pytest.mark.parametrize("d", [512, 768])
def test_small_transition_ln_fwd(d):
    x = _bf((4096 + 13, d))
    gamma, beta = torch.randn(d, device="cuda", generator=_gen(1)), torch.randn(d, device="cuda", generator=_gen(2))
    xn, rs, c1 = _tr().ln_fwd(x, gamma, beta, 1e-5)
    ref = torch.nn.functional.layer_norm(x.float(), (d,), gamma, beta, 1e-5)
    torch.testing.assert_close(xn.float(), ref, atol=3e-2, rtol=2e-2)
    xf = x.float()
    torch.testing.assert_close(rs, torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(c1, xf.mean(-1) * rs, rtol=1e-4, atol=1e-4)


@needs_h100
@pytest.mark.slow
@pytest.mark.parametrize("d", [512, 768])
def test_big_transition_ln_fwd(d):
    s = 4096
    x = _bf((s, d))
    gamma, beta = torch.randn(d, device="cuda", generator=_gen(1)), torch.randn(d, device="cuda", generator=_gen(2))
    xn_s, rs_s, c1_s = _tr().ln_fwd(x, gamma, beta, 1e-5)
    r = _tiles(s, d)
    xb = x.repeat(r, 1)
    assert xb.numel() > BIG
    xn, rs, c1 = _tr().ln_fwd(xb, gamma, beta, 1e-5)
    del xb
    xn, rs, c1 = xn.view(r, s, d), rs.view(r, s), c1.view(r, s)
    for t in (0, r // 2, r - 1):
        assert torch.equal(xn[t], xn_s) and torch.equal(rs[t], rs_s) and torch.equal(c1[t], c1_s), f"tile {t} of {r}"
    del xn
    _free()


@needs_h100
def test_small_transition_weight_prep():
    h, d = 1536, 384
    wa = torch.randn(d, h, device="cuda", generator=_gen(0)).t()  # non-contiguous fp32 [h, d]
    wb = torch.randn(h, d, device="cuda", generator=_gen(1))
    ws = torch.randn(d, h, device="cuda", generator=_gen(2)).to(torch.bfloat16)
    a, b, s, st = _tr().prepare(wa, wb, ws, True)
    assert torch.equal(a, wa.to(torch.bfloat16)) and torch.equal(b, wb.to(torch.bfloat16)) and torch.equal(s, ws)
    assert torch.equal(st, ws.t())
    a16, b16 = wa.to(torch.bfloat16).contiguous(), wb.to(torch.bfloat16)
    for block in (0, 128):
        for transpose in (False, True):
            out = _tr().pair_layout(a16, b16, block, transpose)
            if block:
                ref = torch.stack((a16.view(h // block, block, d), b16.view(h // block, block, d)), 1).reshape(2 * h, d)
            else:
                ref = torch.cat((a16, b16), 0)
            assert torch.equal(out, ref.t() if transpose else ref), (block, transpose)
    pg, pb = torch.randn(1000, d, device="cuda", generator=_gen(3)), torch.randn(1000, d, device="cuda", generator=_gen(4))
    dg, db = _tr().reduce_affine(pg, pb)
    torch.testing.assert_close(dg, pg.sum(0), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(db, pb.sum(0), rtol=1e-5, atol=1e-4)


@needs_h100
@pytest.mark.slow
def test_big_transition_weight_prep():
    """The int64 builds of prepare / pair_layout (selected when a weight passes 2**31 elements)."""
    d = 1024
    h = (BIG >> 10) // 2 + 32  # pair_layout: 2 h d > 2**31
    a, b = _bf((h, d), 0), _bf((h, d), 1)
    out = _tr().pair_layout(a, b, 0, False)
    assert out.numel() > BIG
    for rows in (slice(0, 64), slice(h - 64, h)):
        assert torch.equal(out[rows], a[rows]) and torch.equal(out[h:][rows], b[rows])
    del out, a, b
    _free()
    h = (BIG >> 10) + 32  # prepare: an fp32 [h, d] leaf of > 2**31 elements
    wa = torch.randn(h, d, device="cuda", generator=_gen(2))
    wb, ws = _bf((h, d), 3), _bf((d, h), 4)
    pa, pb, ps, _ = _tr().prepare(wa, wb, ws, False)
    assert pb is wb or torch.equal(pb, wb)
    for rows in (slice(0, 64), slice(h - 64, h)):
        assert torch.equal(pa[rows], wa[rows].to(torch.bfloat16))
    del pa, pb, ps, wa, wb, ws
    _free()


# ---- TriMul live aux unit (wide_forward/output.cu) ----------------------------------------------------------------------------------


def _pk():
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_pack as P

    return P


def _ctx():
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

    return T.native_context(torch.device("cuda", torch.cuda.current_device()))


def _norm(x, width):
    P = _pk()
    g = torch.randn(width, device="cuda", generator=_gen(5))
    b = torch.randn(width, device="cuda", generator=_gen(6))
    y = torch.empty_like(x)
    stats = torch.empty(x.shape[0] * 2, device="cuda")
    with _ctx():
        P.normalize_into(y, x, g, b, stats)
    return y, stats.view(-1, 2), g, b


@needs_h100
@pytest.mark.parametrize("width", [128, 384, 512])
def test_small_trimul_norm_warp(width):
    x = _bf((4096 + 5, width))
    y, stats, g, b = _norm(x, width)
    torch.testing.assert_close(y.float(), torch.nn.functional.layer_norm(x.float(), (width,), g, b, 1e-5), atol=4e-2, rtol=2e-2)
    torch.testing.assert_close(stats[:, 0], x.float().mean(-1), atol=1e-4, rtol=1e-4)


@needs_h100
@pytest.mark.slow
def test_big_trimul_norm_warp():
    s, width = 4096, 512
    x = _bf((s, width))
    y_s, st_s, _, _ = _norm(x, width)
    r = _tiles(s, width)
    xb = x.repeat(r, 1)
    assert xb.numel() > BIG
    y, st, _, _ = _norm(xb, width)
    del xb
    y, st = y.view(r, s, width), st.view(r, s, 2)
    for t in (0, r // 2, r - 1):
        assert torch.equal(y[t], y_s) and torch.equal(st[t], st_s), f"tile {t} of {r}"
    del y
    _free()


@needs_h100
def test_small_trimul_copies_and_fills():
    P = _pk()
    with _ctx():
        src = torch.randn(300, 200, device="cuda", generator=_gen(0)).t()  # fp32 strided
        assert torch.equal(P.dense_native(src, torch.bfloat16), src.to(torch.bfloat16))
        dst = torch.empty(200, 300, device="cuda")
        P.dense_copy_into(dst, src.to(torch.bfloat16))
        assert torch.equal(dst, src.to(torch.bfloat16).float())
        many = [src, torch.randn(64, 32, device="cuda", generator=_gen(1)).to(torch.bfloat16).t(), src.contiguous()]
        out = P.dense_native_many(many, [torch.bfloat16, torch.float32, torch.float32])
        assert torch.equal(out[0], src.to(torch.bfloat16)) and torch.equal(out[1], many[1].float()) and out[2] is many[2]
        z = torch.ones(1000003, device="cuda", dtype=torch.uint8)
        assert int(P.zero_native_(z).sum()) == 0
        tok = torch.rand(3 * 50, device="cuda", generator=_gen(2)) > 0.3
        m = P.owned_bf16_mask(tok, 50, tok.device, 3).view(3, 50, 50)
        t3 = tok.view(3, 50)
        assert torch.equal(m, (t3[:, :, None] & t3[:, None, :]).to(torch.bfloat16))
        pair = torch.rand(3, 50, 50, device="cuda", generator=_gen(3))
        assert torch.equal(P.owned_bf16_mask(pair, 50, tok.device, 3), pair.view(150, 50).to(torch.bfloat16))
        one = P.pair_mask(tok[:50], 50, tok.device)
        assert torch.equal(one, (tok[:50, None] & tok[None, :50]).to(torch.bfloat16))
        u = P._training_unit_scale(torch.empty(2, 37, 37, 128, device="cuda"))
        assert u.shape == (74, 128) and bool((u == 1).all())


@needs_h100
def test_small_trimul_weight_glue():
    P = _pk()
    r, d = 384, 128
    quad = [torch.randn(r, d, device="cuda", generator=_gen(i)).to(torch.bfloat16) for i in range(4)]
    ref = torch.empty(4 * r, d, device="cuda", dtype=torch.bfloat16)
    P._pack_blocks(ref, *quad)
    with _ctx():
        dst = torch.empty_like(ref)
        P.pack_into_cuda(dst, *[q.float() for q in quad])  # mw_aux_pack (fp32 leaves)
        assert torch.equal(dst, ref)
        L = P.T._launch_module()
        vec = torch.empty_like(ref)
        p = L.Struct([vec, *quad, r, d, *(w.stride(0) for w in quad), *(w.stride(1) for w in quad), 0, 0, 0, 0])
        P._launch("mw_aux_pack_vec", ((4 * r * d // 8 + 255) // 256, 1, 1), [p])
        assert torch.equal(vec, ref)
        wg = torch.randn(d, d, device="cuda", generator=_gen(9)).to(torch.bfloat16)
        packed, stack = torch.empty_like(ref), torch.empty(d + 4 * r, d, device="cuda", dtype=torch.bfloat16)
        P.pack_training_pair(packed, stack, (wg, *quad))
        assert torch.equal(packed, ref) and torch.equal(stack, torch.cat((wg, *quad), 0))
        assert torch.equal(P.stack_backward_weights((wg, *quad)), torch.cat((wg, *quad), 0))
        pg, pb = torch.randn(70, d, device="cuda", generator=_gen(10)), torch.randn(70, d, device="cuda", generator=_gen(11))
        dg, db = torch.empty(d, device="cuda"), torch.empty(d, device="cuda")
        P.finish_affine_partials(pg, pb, dg, db)
        torch.testing.assert_close(dg, pg.sum(0), rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(db, pb.sum(0), rtol=1e-5, atol=1e-4)
        qg, qb = torch.randn(9, 2 * d, device="cuda", generator=_gen(12)), torch.randn(9, 2 * d, device="cuda", generator=_gen(13))
        out = [
            torch.empty(d, device="cuda"),
            torch.empty(d, device="cuda"),
            torch.empty(2 * d, device="cuda"),
            torch.empty(2 * d, device="cuda"),
        ]
        P.finish_affine_pair((pg, pb, out[0], out[1]), (qg, qb, out[2], out[3]))
        for o, ref_sum in zip(out, (pg.sum(0), pb.sum(0), qg.sum(0), qb.sum(0)), strict=True):
            torch.testing.assert_close(o, ref_sum, rtol=1e-5, atol=1e-4)
        hidden = 2 * d
        wp = torch.randn(d, hidden, device="cuda", generator=_gen(14)).to(torch.bfloat16)
        go, bo = torch.randn(hidden, device="cuda", generator=_gen(15)), torch.randn(hidden, device="cuda", generator=_gen(16))
        gi, bi = torch.randn(d, device="cuda", generator=_gen(17)), torch.randn(d, device="cuda", generator=_gen(18))
        wpf, wgf, uv = torch.empty_like(wp), torch.empty_like(wg), torch.empty(4 * d, device="cuda")
        P.fold_into(wp, go, bo, wpf, wg, gi, bi, wgf, uv, gate=True)
        assert torch.equal(wpf, (wp.float() * go).to(torch.bfloat16)) and torch.equal(wgf, (wg.float() * gi).to(torch.bfloat16))
        torch.testing.assert_close(uv[d : 2 * d], wp.float() @ bo, rtol=1e-3, atol=1e-2)
        torch.testing.assert_close(uv[3 * d :], wg.float() @ bi, rtol=1e-3, atol=1e-2)
        # D64 training preparation (packed weights, gate, output, pair mask of B samples from a token mask)
        hid, length, samples = 128, 40, 3
        ws = [torch.randn(hid, 64, device="cuda", generator=_gen(20 + i)) for i in range(4)]
        ws += [
            torch.randn(64, 64, device="cuda", generator=_gen(24)),
            torch.randn(64, hid, device="cuda", generator=_gen(25)).to(torch.bfloat16),
        ]
        tok = torch.rand(samples * length, device="cuda", generator=_gen(26)) > 0.3
        x = torch.empty(samples, length, length, 64, device="cuda", dtype=torch.bfloat16)
        packed64, maskf, gate, output = P.fused_training_prepare([x, *ws], tok)
        ref64 = torch.empty(4 * hid, 64, device="cuda", dtype=torch.bfloat16)
        P._pack_blocks(ref64, *[w.to(torch.bfloat16) for w in ws[:4]])
        assert torch.equal(packed64, ref64) and torch.equal(gate, ws[4].to(torch.bfloat16)) and torch.equal(output, ws[5])
        t3 = tok.view(samples, length)
        assert torch.equal(maskf.view(samples, length, length), (t3[:, :, None] & t3[:, None, :]).float())


@needs_h100
@pytest.mark.slow
def test_big_trimul_copies_and_fills():
    P = _pk()
    with _ctx():
        # dense copy and the six-way copy past 2**31 elements: a transposed bf16 matrix (64-bit index path)
        base = _bf((1 << 15, (BIG >> 15) + 64))
        src = base.t()
        assert src.numel() > BIG
        out = P.dense_native(src)
        for rows in (slice(0, 64), slice(-64, None)):
            assert torch.equal(out[rows], src[rows])
        del out
        _free()
        small = torch.randn(33, 17, device="cuda", generator=_gen(1)).t()
        out, out_small = P.dense_native_many([src, small], [torch.bfloat16, torch.bfloat16])
        for rows in (slice(0, 64), slice(-64, None)):
            assert torch.equal(out[rows], src[rows])
        assert torch.equal(out_small, small.to(torch.bfloat16))
        del out, base, src
        _free()
        z = torch.ones(BIG + 4097, device="cuda", dtype=torch.uint8)
        P.zero_native_(z)
        assert int(z[-8192:].sum()) == 0 and int(z[:8192].sum()) == 0 and int(z.sum(dtype=torch.int64)) == 0
        del z
        _free()
        # masks: one sample of length 46341 (46341**2 > 2**31) and 129 samples of 4096
        n = 46341
        tok = torch.rand(n, device="cuda", generator=_gen(2)) > 0.3
        m = P.pair_mask(tok, n, tok.device)
        assert m.numel() > BIG
        for rows in (slice(0, 8), slice(-8, None)):
            assert torch.equal(m[rows], (tok[rows, None] & tok[None, :]).to(torch.bfloat16))
        del m
        _free()
        samples, n = 129, 4096
        tok = torch.rand(samples * n, device="cuda", generator=_gen(3)) > 0.3
        m = P.owned_bf16_mask(tok, n, tok.device, samples).view(samples, n, n)
        t = tok.view(samples, n)
        for s in (0, samples - 1):
            assert torch.equal(m[s], (t[s, :, None] & t[s, None, :]).to(torch.bfloat16)), f"sample {s}"
        del m
        _free()
        u = P._training_unit_scale(torch.empty((), device="cuda").expand(1, (BIG >> 9) + 64, 1, 512))
        assert u.numel() > BIG and bool((u[-4:] == 1).all()) and bool((u[:4] == 1).all())
        del u
        _free()
