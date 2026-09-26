# augattn_sm100 — the token DiT's augmented pair-bias attention core on B200 (sm_100a)

Port of the H100 work (`research/token-dit-overlap` `experiments/token_dit_train`, engine `kernels/augmented_attention/cuda_sm90`, not on
origin; sources copied from the H100 box) to tcgen05 / TMEM / TMA. The op (AF3 Alg. 24 as the token DiT runs it, A augmented samples
sharing one pair bias per head):

    o[a, :, i, h] = softmax_j( q[a, i, h] . k[a, j, h] / sqrt(48) + bias[h, i, j] ) v[a, j, h]      H = 16, D = 48, A = 48

Files: `src/attn_fwd.cu` (v1, one CTA per (head, q-tile, sample pair)), `src/attn_fwd2.cu` (persistent, current), `ops.py` (hosts),
`bench_base.py` (baselines), `test_fwd.py`, `trace_fwd*.py`, `prof_fwd.py` (ncu via gcsudo), `src/n48_test.cu` (layout unit test).
Rounds in `rounds/`.
