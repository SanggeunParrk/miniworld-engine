# atomdit_sm100 — the atom DiT block on B200 (sm_100a)

Target: the engine's `dit_atom` benchmark block (modules/dit.DiTBlock at atom widths: d_single = d_cond = 128, d_pair = 16, 4 heads x 32,
N = 8 * seq_len atoms, FULL pair-bias attention over all N atoms, A = 5 inference / 48 training samples sharing one pair tensor z [N, N, 16]).
There is no H100 atom DiT optimization to port (checked every H100 checkout: the H100 work is the token DiT); the attention kernels start
from the token-DiT B200 kernels (augattn_sm100 on perf/augattn-sm100-b200).

Files: `src/pair_bias.cu` (LN(z) Wb^T -> bias and bias^T, and its backward), `src/attn_fwd.cu`, `src/attn_dkv.cu`, `src/attn_dq.cu`,
`src/attn_dbias.cu` (attention core), `atom_block.py` (the block with these kernels as autograd Functions), `ops.py` (cubin hosts),
`pair_bias.py`, `test_pair_bias.py`, `test_attn.py`, `test_block.py`, `bench_base.py` (baselines), `prof_*.py`.
Build on the B200 box: `for k in pair_bias attn_fwd attn_dkv attn_dq attn_dbias; do ./build.sh $k; done`. Rounds in `rounds/`.
