# swaatom_sm100 — the SWA atom DiT block (team-gm SWAAtomBlock, block_style "esmfold2") on B200 (sm_100a)

The H100 work being ported: team-gm (CSSB-SNU/team-gm, origin/main) 14f2c73 "preserve opt-in fused atom transformer work" --
`swa_fused_triton.py` (vendored here as `h100_fused.py`) with the sm_90a CUDA kernels `swa_cuda/swa_qkvg_fwd.cu`, `swa_ffn_fwd.cu`,
`swa_ffn_bwd.cu`; enabled by MINIWORLD_SWA_FUSED=1. The fusion is kept as on H100 (per block forward: qkvg | window attention |
out-proj + FFN; backward: FFN | out-proj | attention dq / dkv | qkvg, weight grads on cuBLAS); the kernels move to tcgen05 / TMEM / TMA.

Shapes: C = 128, 4 heads x 32, window |i - j| <= 64, SwiGLU hidden 256, rows ((a B + b) S + s); benchmarks: S = 8 L atoms, B = 1, A = 5
(inference) / 48 (training). `anthropic_ef2_atom.py`: Anthropic's fused SWAAtomBlock (uplifting-biomolecular-modeling @f4f62fa,
esmfold2/opt/forward/fast_inference/driver/ef2_atom.py) for the inference comparison.

Files: `common.py` (inputs, fp64 reference), `bench_base.py` (torch eager / compile / H100 fused on B200), `bench_anthropic.py`,
`src/qkvg_fwd.cu` + `ops.py` + `test_qkvg.py`, `prof_*.py`. Build: `./build.sh qkvg_fwd` on the B200 box. Rounds in `rounds/`.
