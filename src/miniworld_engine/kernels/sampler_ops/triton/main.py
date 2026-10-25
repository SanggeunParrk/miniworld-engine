"""Rigid-motion glue of an EDM denoiser step (``sampler_ops``) as portable kernels: the contract and the algorithm of the H100 CUDA kernels on any card.

fp32 coordinates ``[B, N, 3]``, one rigid motion per sample, ``mask`` ``[N]`` fp32. Random numbers are drawn by the caller (``rv`` ``[B, 2, 3]``, ``tr`` ``[B, 3]``,
``noise`` ``[B, N, 3]``), exactly as for the CUDA op, so the same draws give the same result.

* ``augment_noise``: ``_stats`` (per-chunk partial sums ``[B, C, 16]`` of the mask and the mask-weighted coordinates), then ``_augment`` (every program adds the
  ``C`` partials of its sample, forms the centre, the Gram-Schmidt frame of ``rv`` and the churn scalars, and writes ``noisy``); no atomics, no host read.
* ``align_euler``: ``_stats`` over both tensors, ``_cov`` (the 3x3 weighted covariance of the centred sets as partials), ``_align`` (every program adds the
  partials, takes the rotation from Horn's quaternion method -- the dominant eigenvector of the symmetric 4x4 ``N + |N|_F I`` by 12 squarings with a
  max-norm rescale, no SVD, in scalar registers -- and writes the Euler update). Two passes over the coordinates (centres, then the centred covariance) as the
  CUDA kernels, not the one-pass raw second moments: those lose the digits of a structure far from the origin.
* ``pool_atoms``: one program per (token, 256-channel slice, sample) gathers the rows of the token's atoms, relu, masked mean, bf16.

Division and square root are the IEEE ones (``div_rn`` / ``sqrt_rn``), the sums are fixed-order (partials of ``CHUNK`` atoms, then a tree): bitwise reproducible.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import atom_key

CHUNK = 1024                 # atoms per program of the partial-sum passes at most (a launch smaller than ~2 waves of programs halves it, see ``chunk_for``)
_MIN_PROGRAMS = 264          # two waves of an H100
PART = 16                    # floats per partial record



@triton.jit
def _psum(PART_PTR, b, C, k, CP: tl.constexpr):
    """Sum of field ``k`` of the ``C`` partial records of sample ``b`` (a masked vector load and a tree sum)."""
    c = tl.arange(0, CP)
    return tl.sum(tl.load(PART_PTR + (b * C + c) * 16 + k, mask=c < C, other=0.0), axis=0)


@triton.autotune(configs=configs_for("sampler_ops_stats_triton"), key=["shape_key", "TWO"])
@triton.jit
def _stats(M, R, W, PARTS, n, TWO: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr, shape_key):
    """PARTS[b, chunk] = (sum w, sum w m_xyz, [sum w r_xyz]) over the atoms of the chunk (sub-tiles of BLOCK atoms in order, then a tree sum)."""
    c = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    z = tl.zeros([BLOCK], dtype=tl.float32)
    s0, s1, s2, s3, s4, s5, s6 = z, z, z, z, z, z, z
    for t in tl.static_range(0, CHUNK, BLOCK):
        lane = t + tl.arange(0, BLOCK)
        i = c * CHUNK + lane
        ok = (i < n) & (lane < CHUNK)
        w = tl.load(W + i, mask=ok, other=0.0)
        o = (b * n + i) * 3
        s0 += w
        s1 += w * tl.load(M + o, mask=ok, other=0.0)
        s2 += w * tl.load(M + o + 1, mask=ok, other=0.0)
        s3 += w * tl.load(M + o + 2, mask=ok, other=0.0)
        if TWO:
            s4 += w * tl.load(R + o, mask=ok, other=0.0)
            s5 += w * tl.load(R + o + 1, mask=ok, other=0.0)
            s6 += w * tl.load(R + o + 2, mask=ok, other=0.0)
    base = PARTS + (b * tl.num_programs(0) + c) * 16
    tl.store(base, tl.sum(s0, axis=0))
    tl.store(base + 1, tl.sum(s1, axis=0))
    tl.store(base + 2, tl.sum(s2, axis=0))
    tl.store(base + 3, tl.sum(s3, axis=0))
    if TWO:
        tl.store(base + 4, tl.sum(s4, axis=0))
        tl.store(base + 5, tl.sum(s5, axis=0))
        tl.store(base + 6, tl.sum(s6, axis=0))


@triton.autotune(configs=configs_for("sampler_ops_augment_triton"), key=["shape_key"])
@triton.jit
def _augment(P, W, PARTS, RV, TR, NOISE, SIGMA, SIGMA_NEXT, OUT, SC, n, C, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor,
             WINDOW: tl.constexpr, CLAMP: tl.constexpr, CHUNK: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr, shape_key):
    chunk = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    tot = tl.maximum(_psum(PARTS, b, C, 0, CP), 1e-6)
    cx = tl.math.div_rn(_psum(PARTS, b, C, 1, CP), tot)
    cy = tl.math.div_rn(_psum(PARTS, b, C, 2, CP), tot)
    cz = tl.math.div_rn(_psum(PARTS, b, C, 3, CP), tot)
    v = RV + b * 6
    v0x, v0y, v0z = tl.load(v), tl.load(v + 1), tl.load(v + 2)
    v1x, v1y, v1z = tl.load(v + 3), tl.load(v + 4), tl.load(v + 5)
    n0 = tl.maximum(tl.sqrt_rn(v0x * v0x + v0y * v0y + v0z * v0z), 1e-10)
    e0x, e0y, e0z = tl.math.div_rn(v0x, n0), tl.math.div_rn(v0y, n0), tl.math.div_rn(v0z, n0)
    d = v1x * e0x + v1y * e0y + v1z * e0z
    ux, uy, uz = v1x - e0x * d, v1y - e0y * d, v1z - e0z * d
    n1 = tl.maximum(tl.sqrt_rn(ux * ux + uy * uy + uz * uz), 1e-10)
    e1x, e1y, e1z = tl.math.div_rn(ux, n1), tl.math.div_rn(uy, n1), tl.math.div_rn(uz, n1)
    e2x, e2y, e2z = e0y * e1z - e0z * e1y, e0z * e1x - e0x * e1z, e0x * e1y - e0y * e1x
    tx, ty, tz = tl.load(TR + b * 3), tl.load(TR + b * 3 + 1), tl.load(TR + b * 3 + 2)
    # the churn scalars (the torch path's float32 arithmetic)
    sigma = tl.load(SIGMA)
    sigma_n = tl.load(SIGMA_NEXT)
    if WINDOW:
        on = (sigma >= sigma_min) & (sigma <= sigma_max)
    else:
        on = sigma_n > gamma_min
    sigma_hat = libdevice.mul_rn(sigma, tl.where(on, gamma, 1.0))
    variance = libdevice.sub_rn(libdevice.mul_rn(sigma_hat, sigma_hat), libdevice.mul_rn(sigma, sigma))
    if CLAMP:
        variance = tl.maximum(variance, variance_floor)
    mag = libdevice.mul_rn(noise_scale, tl.sqrt_rn(variance))
    first = (chunk == 0) & (b == 0)
    tl.store(SC, sigma_hat, mask=first)
    tl.store(SC + 1, mag, mask=first)
    tl.store(SC + 2, 0.0, mask=first)
    tl.store(SC + 3, 0.0, mask=first)

    for t in tl.static_range(0, CHUNK, BLOCK):
        lane = t + tl.arange(0, BLOCK)
        i = chunk * CHUNK + lane
        ok = (i < n) & (lane < CHUNK)
        o = (b * n + i) * 3
        x = tl.load(P + o, mask=ok, other=0.0) - cx
        y = tl.load(P + o + 1, mask=ok, other=0.0) - cy
        z = tl.load(P + o + 2, mask=ok, other=0.0) - cz
        w = tl.load(W + i, mask=ok, other=0.0)
        a0 = x * e0x + y * e1x + z * e2x
        a1 = x * e0y + y * e1y + z * e2y
        a2 = x * e0z + y * e1z + z * e2z
        tl.store(OUT + o, libdevice.add_rn(libdevice.mul_rn(libdevice.add_rn(a0, tx), w), libdevice.mul_rn(mag, tl.load(NOISE + o, mask=ok, other=0.0))), mask=ok)
        tl.store(OUT + o + 1, libdevice.add_rn(libdevice.mul_rn(libdevice.add_rn(a1, ty), w), libdevice.mul_rn(mag, tl.load(NOISE + o + 1, mask=ok, other=0.0))), mask=ok)
        tl.store(OUT + o + 2, libdevice.add_rn(libdevice.mul_rn(libdevice.add_rn(a2, tz), w), libdevice.mul_rn(mag, tl.load(NOISE + o + 2, mask=ok, other=0.0))), mask=ok)


@triton.autotune(configs=configs_for("sampler_ops_cov_triton"), key=["shape_key"])
@triton.jit
def _cov(M, R, W, PARTS, COV, n, C, CHUNK: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr, shape_key):
    """COV[b, chunk, 3p + q] = sum over the chunk of w (m_p - cm_p) (r_q - cr_q)."""
    chunk = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    tot = _psum(PARTS, b, C, 0, CP)
    cm0, cm1, cm2 = tl.math.div_rn(_psum(PARTS, b, C, 1, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 2, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 3, CP), tot)
    cr0, cr1, cr2 = tl.math.div_rn(_psum(PARTS, b, C, 4, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 5, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 6, CP), tot)
    z = tl.zeros([BLOCK], dtype=tl.float32)
    h00, h01, h02, h10, h11, h12, h20, h21, h22 = z, z, z, z, z, z, z, z, z
    for t in tl.static_range(0, CHUNK, BLOCK):
        lane = t + tl.arange(0, BLOCK)
        i = chunk * CHUNK + lane
        ok = (i < n) & (lane < CHUNK)
        o = (b * n + i) * 3
        w = tl.load(W + i, mask=ok, other=0.0)
        a0 = (tl.load(M + o, mask=ok, other=0.0) - cm0) * w
        a1 = (tl.load(M + o + 1, mask=ok, other=0.0) - cm1) * w
        a2 = (tl.load(M + o + 2, mask=ok, other=0.0) - cm2) * w
        c0 = tl.where(ok, tl.load(R + o, mask=ok, other=0.0) - cr0, 0.0)
        c1 = tl.where(ok, tl.load(R + o + 1, mask=ok, other=0.0) - cr1, 0.0)
        c2 = tl.where(ok, tl.load(R + o + 2, mask=ok, other=0.0) - cr2, 0.0)
        h00 += a0 * c0
        h01 += a0 * c1
        h02 += a0 * c2
        h10 += a1 * c0
        h11 += a1 * c1
        h12 += a1 * c2
        h20 += a2 * c0
        h21 += a2 * c1
        h22 += a2 * c2
    base = COV + (b * tl.num_programs(0) + chunk) * 16
    tl.store(base + 0, tl.sum(h00, axis=0))
    tl.store(base + 1, tl.sum(h01, axis=0))
    tl.store(base + 2, tl.sum(h02, axis=0))
    tl.store(base + 3, tl.sum(h10, axis=0))
    tl.store(base + 4, tl.sum(h11, axis=0))
    tl.store(base + 5, tl.sum(h12, axis=0))
    tl.store(base + 6, tl.sum(h20, axis=0))
    tl.store(base + 7, tl.sum(h21, axis=0))
    tl.store(base + 8, tl.sum(h22, axis=0))


@triton.jit
def _sq4(a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33):
    s00 = a00 * a00 + a01 * a10 + a02 * a20 + a03 * a30
    s01 = a00 * a01 + a01 * a11 + a02 * a21 + a03 * a31
    s02 = a00 * a02 + a01 * a12 + a02 * a22 + a03 * a32
    s03 = a00 * a03 + a01 * a13 + a02 * a23 + a03 * a33
    s10 = a10 * a00 + a11 * a10 + a12 * a20 + a13 * a30
    s11 = a10 * a01 + a11 * a11 + a12 * a21 + a13 * a31
    s12 = a10 * a02 + a11 * a12 + a12 * a22 + a13 * a32
    s13 = a10 * a03 + a11 * a13 + a12 * a23 + a13 * a33
    s20 = a20 * a00 + a21 * a10 + a22 * a20 + a23 * a30
    s21 = a20 * a01 + a21 * a11 + a22 * a21 + a23 * a31
    s22 = a20 * a02 + a21 * a12 + a22 * a22 + a23 * a32
    s23 = a20 * a03 + a21 * a13 + a22 * a23 + a23 * a33
    s30 = a30 * a00 + a31 * a10 + a32 * a20 + a33 * a30
    s31 = a30 * a01 + a31 * a11 + a32 * a21 + a33 * a31
    s32 = a30 * a02 + a31 * a12 + a32 * a22 + a33 * a32
    s33 = a30 * a03 + a31 * a13 + a32 * a23 + a33 * a33
    tiny = tl.full([], 1.1754943508222875e-38, tl.float32)
    am = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.abs(s00), tl.abs(s01)), tl.abs(s02)), tl.abs(s03)), tl.abs(s10)), tl.abs(s11)), tl.abs(s12)), tl.abs(s13)), tl.abs(s20)), tl.abs(s21)), tl.abs(s22)), tl.abs(s23)), tl.abs(s30)), tl.abs(s31)), tl.abs(s32)), tl.abs(s33)), tiny)
    return tl.math.div_rn(s00, am), tl.math.div_rn(s01, am), tl.math.div_rn(s02, am), tl.math.div_rn(s03, am), tl.math.div_rn(s10, am), tl.math.div_rn(s11, am), tl.math.div_rn(s12, am), tl.math.div_rn(s13, am), tl.math.div_rn(s20, am), tl.math.div_rn(s21, am), tl.math.div_rn(s22, am), tl.math.div_rn(s23, am), tl.math.div_rn(s30, am), tl.math.div_rn(s31, am), tl.math.div_rn(s32, am), tl.math.div_rn(s33, am)


@triton.jit
def _horn(sxx, sxy, sxz, syx, syy, syz, szx, szy, szz):
    """The rotation ``R`` (row vectors, ``aligned = a R``) of the covariance by Horn's quaternion method: 9 scalars, row-major."""
    n00 = sxx + syy + szz
    n01 = syz - szy
    n02 = szx - sxz
    n03 = sxy - syx
    n11 = sxx - syy - szz
    n12 = sxy + syx
    n13 = szx + sxz
    n22 = -sxx + syy - szz
    n23 = syz + szy
    n33 = -sxx - syy + szz
    sq = (n00 * n00 + n11 * n11 + n22 * n22 + n33 * n33) + 2.0 * (n01 * n01 + n02 * n02 + n03 * n03 + n12 * n12 + n13 * n13 + n23 * n23)
    shift = tl.sqrt_rn(sq)
    m00, m11, m22, m33 = n00 + shift, n11 + shift, n22 + shift, n33 + shift
    m01, m02, m03, m12, m13, m23 = n01, n02, n03, n12, n13, n23
    m10, m20, m30, m21, m31, m32 = n01, n02, n03, n12, n13, n23
    for _ in tl.static_range(12):
        m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 = _sq4(
            m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33)
    # the column of largest norm (the first one on a tie) is the quaternion up to scale and sign
    k0 = m00 * m00 + m10 * m10 + m20 * m20 + m30 * m30
    k1 = m01 * m01 + m11 * m11 + m21 * m21 + m31 * m31
    k2 = m02 * m02 + m12 * m12 + m22 * m22 + m32 * m32
    k3 = m03 * m03 + m13 * m13 + m23 * m23 + m33 * m33
    best, q0, q1, q2, q3 = k0, m00, m10, m20, m30
    p = k1 > best
    best, q0, q1, q2, q3 = tl.where(p, k1, best), tl.where(p, m01, q0), tl.where(p, m11, q1), tl.where(p, m21, q2), tl.where(p, m31, q3)
    p = k2 > best
    best, q0, q1, q2, q3 = tl.where(p, k2, best), tl.where(p, m02, q0), tl.where(p, m12, q1), tl.where(p, m22, q2), tl.where(p, m32, q3)
    p = k3 > best
    q0, q1, q2, q3 = tl.where(p, m03, q0), tl.where(p, m13, q1), tl.where(p, m23, q2), tl.where(p, m33, q3)
    tiny = tl.full([], 1.1754943508222875e-38, tl.float32)
    qn = tl.maximum(tl.sqrt_rn(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3), tiny)
    w, x, y, z = tl.math.div_rn(q0, qn), tl.math.div_rn(q1, qn), tl.math.div_rn(q2, qn), tl.math.div_rn(q3, qn)
    r00 = w * w + x * x - y * y - z * z
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (y * x + w * z)
    r11 = w * w - x * x + y * y - z * z
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (z * x - w * y)
    r21 = 2 * (z * y + w * x)
    r22 = w * w - x * x - y * y + z * z
    return r00, r10, r20, r01, r11, r21, r02, r12, r22     # row-major rot, transposed: rot[a, c] = Q[c, a]


@triton.autotune(configs=configs_for("sampler_ops_align_triton"), key=["shape_key"])
@triton.jit
def _align(M, R, PARTS, COV, SC, SIGMA_NEXT, OUT, step_scale, n, C, CHUNK: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr, shape_key):
    chunk = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    tot = _psum(PARTS, b, C, 0, CP)
    cm0, cm1, cm2 = tl.math.div_rn(_psum(PARTS, b, C, 1, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 2, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 3, CP), tot)
    cr0, cr1, cr2 = tl.math.div_rn(_psum(PARTS, b, C, 4, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 5, CP), tot), tl.math.div_rn(_psum(PARTS, b, C, 6, CP), tot)
    t00, t01, t02, t10, t11, t12, t20, t21, t22 = _horn(
        _psum(COV, b, C, 0, CP), _psum(COV, b, C, 1, CP), _psum(COV, b, C, 2, CP), _psum(COV, b, C, 3, CP), _psum(COV, b, C, 4, CP),
        _psum(COV, b, C, 5, CP), _psum(COV, b, C, 6, CP), _psum(COV, b, C, 7, CP), _psum(COV, b, C, 8, CP))
    sigma_hat = tl.load(SC)
    k = libdevice.mul_rn(step_scale, libdevice.sub_rn(tl.load(SIGMA_NEXT), sigma_hat))
    for t in tl.static_range(0, CHUNK, BLOCK):
        lane = t + tl.arange(0, BLOCK)
        i = chunk * CHUNK + lane
        ok = (i < n) & (lane < CHUNK)
        o = (b * n + i) * 3
        x = tl.load(M + o, mask=ok, other=0.0) - cm0
        y = tl.load(M + o + 1, mask=ok, other=0.0) - cm1
        z = tl.load(M + o + 2, mask=ok, other=0.0) - cm2
        al0 = libdevice.add_rn(x * t00 + y * t10 + z * t20, cr0)
        al1 = libdevice.add_rn(x * t01 + y * t11 + z * t21, cr1)
        al2 = libdevice.add_rn(x * t02 + y * t12 + z * t22, cr2)
        d0 = tl.math.div_rn(libdevice.sub_rn(al0, tl.load(R + o, mask=ok, other=0.0)), sigma_hat)
        d1 = tl.math.div_rn(libdevice.sub_rn(al1, tl.load(R + o + 1, mask=ok, other=0.0)), sigma_hat)
        d2 = tl.math.div_rn(libdevice.sub_rn(al2, tl.load(R + o + 2, mask=ok, other=0.0)), sigma_hat)
        tl.store(OUT + o, libdevice.add_rn(al0, libdevice.mul_rn(k, d0)), mask=ok)
        tl.store(OUT + o + 1, libdevice.add_rn(al1, libdevice.mul_rn(k, d1)), mask=ok)
        tl.store(OUT + o + 2, libdevice.add_rn(al2, libdevice.mul_rn(k, d2)), mask=ok)


@triton.autotune(configs=configs_for("sampler_ops_pool_triton"), key=["shape_key"])
@triton.jit
def _pool(X, IDX, GM, AM, OUT, A, T, R, C, BLOCK_C: tl.constexpr, shape_key):
    """OUT[s, t, c] = sum_a m[t, a] relu(g[t, a] x[s, idx[t, a], c]) / max(sum_a m[t, a], 1e-10), bf16; the atoms in order, as the CUDA kernel."""
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(2).to(tl.int64)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cin = c < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    den = 0.0
    for a in range(A):
        kk = t * A + a
        m = tl.load(AM + kk).to(tl.float32)
        g = tl.load(GM + kk).to(tl.float32)
        r = tl.load(IDX + kk).to(tl.int64)
        use = (m != 0.0) & (g != 0.0)
        v = tl.load(X + (s * R + r) * C + c, mask=cin & use, other=0.0).to(tl.float32)
        acc += m * tl.maximum(v, 0.0)
        den += m
    out = tl.math.div_rn(acc, tl.maximum(den, 1e-10))
    tl.store(OUT + (s * T + t) * C + c, out.to(tl.bfloat16), mask=cin)


def chunk_for(n: int, b: int) -> int:
    """Atoms per program of the partial-sum passes: ``CHUNK`` unless the launch would be smaller than about two waves of programs (a handful of samples of a small
    structure), then halved down to 256 so that more programs stream the coordinates; the partials of a sample stay a short list."""
    chunk = CHUNK
    while chunk > 256 and b * triton.cdiv(n, chunk) < _MIN_PROGRAMS:
        chunk //= 2
    return chunk


def _key(n: int, b: int) -> int:
    """The shape key: the atom-count bucket and the sample-count bucket (the best tile of a 1-sample launch is not that of a 48-sample one)."""
    return atom_key(n, B=1 if b < 4 else 4 if b < 16 else 16 if b < 48 else 48)


def _check(t: torch.Tensor, name: str, ndim: int) -> None:
    if not (t.is_cuda and t.is_contiguous() and t.dtype is torch.float32 and t.dim() == ndim):
        raise ValueError(f"{name}: contiguous fp32 cuda {ndim}-d tensor required")


def _cp(c: int) -> int:
    return max(2, triton.next_power_of_2(c))


def portable_augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, *, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp):
    """``(noisy [B, N, 3], sc [4])`` as ``cuda.augment_noise``: ``sc[0] = sigma_hat``, ``sc[1]`` the noise magnitude."""
    _check(coords, "coords", 3)
    _check(mask, "mask", 1)
    _check(noise, "noise", 3)
    B, N = coords.shape[0], coords.shape[1]
    chunk = chunk_for(N, B)
    C = triton.cdiv(N, chunk)
    parts = torch.empty(B, C, PART, device=coords.device, dtype=torch.float32)
    noisy = torch.empty_like(coords)
    sc = torch.empty(4, device=coords.device, dtype=torch.float32)
    key = _key(N, B)
    _stats[(C, B)](coords, coords, mask, parts, N, TWO=0, CHUNK=chunk, shape_key=key)
    _augment[(C, B)](coords, mask, parts, rv, tr, noise, sigma, sigma_next, noisy, sc, N, C, float(1.0 + gamma), float(gamma_min), float(sigma_min),
                     float(sigma_max), float(noise_scale), float(variance_floor), WINDOW=int(window), CLAMP=int(bool(clamp)), CHUNK=chunk, CP=_cp(C), shape_key=key)
    return noisy, sc


def portable_stats(noisy, denoised, mask):
    """The chunk partials ``[B, C, 16]`` of both tensors: (sum w, sum w noisy_xyz, sum w denoised_xyz, 0 ...)."""
    B, N = noisy.shape[0], noisy.shape[1]
    chunk = chunk_for(N, B)
    parts = torch.empty(B, triton.cdiv(N, chunk), PART, device=noisy.device, dtype=torch.float32)
    _stats[(parts.shape[1], B)](noisy, denoised, mask, parts, N, TWO=1, CHUNK=chunk, shape_key=_key(N, B))
    return parts


def portable_cov(noisy, denoised, mask, parts=None):
    """The covariance partials ``[B, C, 16]`` (the first nine: ``sum w (noisy - cm)_p (denoised - cr)_q``, row-major)."""
    B, N = noisy.shape[0], noisy.shape[1]
    parts = portable_stats(noisy, denoised, mask) if parts is None else parts
    C = parts.shape[1]
    cov = torch.empty_like(parts)
    _cov[(C, B)](noisy, denoised, mask, parts, cov, N, C, CHUNK=chunk_for(N, B), CP=_cp(C), shape_key=_key(N, B))
    return cov


def portable_align_euler(noisy, denoised, mask, sc, sigma_next, step_scale):
    """The Euler update from ``noisy`` Kabsch-aligned onto ``denoised`` (weights ``mask``), with ``sigma_hat = sc[0]``."""
    _check(noisy, "noisy", 3)
    _check(denoised, "denoised", 3)
    _check(mask, "mask", 1)
    B, N = noisy.shape[0], noisy.shape[1]
    parts = portable_stats(noisy, denoised, mask)
    C = parts.shape[1]
    cov = portable_cov(noisy, denoised, mask, parts)
    out = torch.empty_like(noisy)
    _align[(C, B)](noisy, denoised, parts, cov, sc, sigma_next, out, float(step_scale), N, C, CHUNK=chunk_for(N, B), CP=_cp(C), shape_key=_key(N, B))
    return out


def portable_pool_atoms(x, idx, gather_mask, atom_mask):
    """``out[s, t] = sum_a m[t, a] relu(g[t, a] x[s, idx[t, a]]) / max(sum_a m[t, a], 1e-10)`` in bf16; ``x`` bf16 ``[S, Q, C]``, the rest ``[T, A]``."""
    if not (x.is_cuda and x.is_contiguous() and x.dtype is torch.bfloat16 and x.dim() == 3):
        raise ValueError("x: contiguous bf16 cuda [S, Q, C] required")
    S, R, C = x.shape
    T, A = idx.shape
    out = torch.empty(S, T, C, device=x.device, dtype=torch.bfloat16)
    key = atom_key(T * A)
    _pool[lambda meta: (T, triton.cdiv(C, meta["BLOCK_C"]), S)](x, idx, gather_mask, atom_mask, out, A, T, R, C, shape_key=key)
    return out
