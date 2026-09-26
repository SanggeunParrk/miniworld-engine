# augattn_sm100 — the token DiT's augmented pair-bias attention core on B200 (sm_100a)

Port of the H100 work (`research/token-dit-overlap` `experiments/token_dit_train`, engine `kernels/augmented_attention/cuda_sm90`, not on
origin; sources copied from the H100 box) to tcgen05 / TMEM / TMA. The op (AF3 Alg. 24 as the token DiT runs it, A augmented samples
sharing one pair bias per head):

    o[a, :, i, h] = softmax_j( q[a, i, h] . k[a, j, h] / sqrt(48) + bias[h, i, j] ) v[a, j, h]      H = 16, D = 48, A = 48

Files: `src/attn_fwd2.cu` (forward, persistent), `src/attn_dqb.cu` (backward dQ + dbias), `src/attn_dkv.cu` (backward dK + dV),
`attn_op.py` (autograd op + backward glue), `ops.py` (cubin hosts), `bench_train.py` (fp64 check + inference / training
latency), `bench_base.py` (baselines), `test_fwd.py`, `test_bwd.py`, `trace_*.py`, `prof_*.py` (ncu via gcsudo), `energy.py`.
Older / rejected: `src/attn_fwd.cu` (v1), `src/attn_dkv4.cu`. Probes: `src/n48_test.cu`, `src/mufu_bench.cu`, `src/mma_lat.cu`.
Build: `./build.sh attn_fwd2 && ./build.sh attn_dqb && ./build.sh attn_dkv` on the B200 box. Rounds in `rounds/`.
