# TurboPairFormer (Lambda / Stevens / OpenFold, Apache-2.0) - Hopper triangle attention

Deterministic single-reconstruction triangle attention fwd + bwd (and a TriMul kernel, not used here) for Hopper only (sm_90a), bf16, head width <= 64.
The PyPI wheel is cp314 + torch==2.10.0 only. Decision: rebuild its shipped CUDA sources as torch extensions in the engine environment (py3.12, torch 2.13 cu129) with the library own
development JIT (`TURBOPAIRFORMER_ALLOW_JIT=1`; `build_local.sh`, build cache `$PAPER_EXT_ATTN/torch_ext`) instead of an out-of-process measurement, so it is measured by the same measurer / CUDA-graph
timing as every other backend. A100 and RTX 5090: `unsupported` (Hopper only). Earlier (2026-10-30, separate py3.14 env, op bench) result: the engine core 2.2x faster fwd+bwd on H100.
