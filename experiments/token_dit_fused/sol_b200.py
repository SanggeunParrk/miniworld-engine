"""Speed-of-light model of one token DiT block on B200 (the SOL every table in this directory is measured against).

Essential work only -- what any implementation of THIS fusion algorithm must do:
  inference step (S samples, bf16 operands, fp32 residual): the 4 block GEMMs + the attention MMAs; HBM: residual x in/out
  (fp32), the block's hoisted pair bias (bf16, head-major), the block weights (bf16) and its AdaLN / gate rows (L rows, shared
  by the samples).
  training block (A augments, bf16 GEMM operands as torch "medium" gives on H100, fp32 parameters and residual): forward +
  backward GEMMs (3x forward), attention forward + backward MMAs (QK^T, PV forward; recomputed S, dP, dV, dK, dQ backward:
  7 L^2 d MMAs against the forward's 2); HBM: x read, out write, dout read, dx write (fp32), pair bias read + dbias write
  (fp32), weights read + weight grads write (fp32), cond read + dcond write.

Two ceilings:
  time    max(FLOPs / 2.38 PFLOP/s, bytes / 7.0 TB/s, exps / 4.65 T/s) -- tensor / HBM / MUFU peaks measured on this card
          (tcgen05 M128 N128 K16 chain 8188 FLOP/clk/SM x 148 x 1.965 GHz; streaming 7.0 TB/s; ex2 16/clk/SM)
  energy  (bytes_r 115 + bytes_w 72 pJ/B + FLOPs 0.47 pJ) / 750 W: the 1000 W cap minus ~250 W idle (measured, see memory
          b200-power-cap-sol-ceiling); a sustained kernel cannot beat it whatever its overlap.
"""
import argparse

D, DC, H, DH, DP, NT = 768, 384, 16, 48, 128, 2          # d_single, d_cond, heads, head dim, d_pair, transition n
TF, BW, EX = 2.38e15, 7.0e12, 4.65e12
PR, PW, PF, PDYN = 115e-12, 72e-12, 0.47e-12, 750.0


def gemm_macs_per_token():
    return D * 4 * D + D * D + D * 2 * NT * D + NT * D * D      # q|k|v|g, Wo, expand (a|b), squeeze


def weights():
    return gemm_macs_per_token()                                    # one parameter per MAC per token


def cond_macs_per_token():
    """AdaLN scale / shift for the attention and the transition, and their two output gates: six d_cond -> d GEMMs. In
    inference all samples share the noise level, so these run on L rows once per step; in training each augment has its own
    noise level, so they run on every token."""
    return 6 * DC * D


def inference(L, S):
    M = S * L
    fl = 2 * M * gemm_macs_per_token() + 4 * S * H * L * L * DH
    ex = S * H * L * L
    rd = M * D * 4 + H * L * L * 2 + weights() * 2 + L * 6 * D * 2      # x, bias, W, AdaLN scale/shift + gates (2 halves)
    wr = M * D * 4
    return fl, ex, rd, wr


def training(L, A):
    M = A * L
    fl = 3 * 2 * M * (gemm_macs_per_token() + cond_macs_per_token()) + 9 * 2 * A * H * L * L * DH + 3 * 2 * L * L * DP * H
    ex = 2 * A * H * L * L                                          # softmax forward + recomputed backward
    wc = weights() + cond_macs_per_token() + DP * H
    rd = 2 * M * D * 4 + M * DC * 4 + L * L * DP * 4 + wc * 4           # x, dout, cond, pair (fp32), W (fp32)
    wr = 2 * M * D * 4 + M * DC * 4 + L * L * DP * 4 + wc * 4           # out, dx, dcond, dpair, dW
    return fl, ex, rd, wr


def floors(fl, ex, rd, wr):
    t_time = max(fl / TF, (rd + wr) / BW, ex / EX)
    t_energy = (rd * PR + wr * PW + fl * PF) / PDYN
    return t_time * 1e6, t_energy * 1e6


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--samples", type=int, default=5); p.add_argument("--augment", type=int, default=48)
    a = p.parse_args()
    for L in (384, 768):
        for nm, (fl, ex, rd, wr) in (("inference S%d" % a.samples, inference(L, a.samples)), ("training A%d" % a.augment, training(L, a.augment))):
            tt, te = floors(fl, ex, rd, wr)
            print(f"L{L} {nm:14s} {fl/1e9:7.1f} GFLOP  {ex/1e6:7.1f} M exp  {(rd+wr)/1e6:7.1f} MB   "
                  f"SOL time {tt:8.1f} us  (tensor {fl/TF*1e6:.1f}, HBM {(rd+wr)/BW*1e6:.1f}, MUFU {ex/EX*1e6:.1f})   SOL energy {te:8.1f} us")
