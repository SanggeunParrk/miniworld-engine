# MPNN rebase onto main — 2026-09-12

The latest ported MPNN work (`mpnn-restyle` at `eb27cfa8`) was rebased onto
`origin/main` at `fb677741`. Its 18 commits are retained in order on local `mpnn`,
followed by integration fixes. The older `mpnn` tip was `aa519265`.

Backup refs: `backup/mpnn-before-main-rebase-20260912` and
`backup/mpnn-restyle-before-main-rebase-20260912`. No remote ref was changed.
The separate main checkout and its A5000 build were not modified by this rebase.

## Integration checks

| Check | Result |
|---|---|
| Main kernel registry | All 98 existing rows preserved; 22 MPNN rows added |
| MPNN driver plan | 88 units: 22 kernels at 1,024 / 2,048 / 4,096 / 8,192 nodes |
| Non-GPU suite | 2,839 passed, 63 skipped, 379 deselected |
| Ruff, `src tests benchmarks` | Passed |
| ty, `src` | Passed |
| ty, `src tests benchmarks` | 228 inherited diagnostics; pre-rebase had 274; no new diagnostics |
| Wheel | Built through setuptools; all 337 Python files and runtime assets verified |
| sm86 dispatch derivation | 5,258 invocations, 0 errors, 1,008 required keys |
| Generated plan and documentation tests | 29 passed, 3 skipped |

The regenerated plan has exactly the same required keys and per-invocation mappings
as main. Its evidence now carries the rebased source identity
`08599d9fb51e6a117210be6c6f857bf62cacbd97c2aac8fca3c61a44f71431c8`.
The sweep HTML was regenerated and was unchanged. This does not constitute an MPNN
cache rebuild or new performance qualification.

## A6000 GPU results and remaining failures

`python -m pytest -q -m gpu tests/mpnn` initially produced **138 passed, 3 failed**
(88 deselected). All three failures were reproduced on a clean checkout of the
pre-rebase commit `eb27cfa8`, with the same Python 3.12 / PyTorch 2.10.0+cu128 /
Triton 3.6.0 environment and an RTX A6000.

The stale tile fixture in `test_every_gradient_survives_autotuning` was fixed:
`BLOCK_M` became `BLOCK_M1`, column-tiled GEMMs receive `BLOCK_N` and `GROUP_M`,
and the fused output projection retains its full-width configuration. Both tests
in `test_mpnn_edge_tail_compute.py` then passed on the GPU. Numerical tolerances
were not changed. The full GPU suite was not repeated after this fixture-only fix.

Two inherited GPU failures remain unresolved:

- `test_mpnn_edge_tail_model_matches_the_separate_operation_encoder`: relative
  gradient drift approximately 1.000 at
  `decoder.layers.1.node_message.hidden_projection.bias` exceeds its 0.25 bound.
  The rebase did not establish whether the cause is kernel arithmetic or the
  comparison's sensitivity to this particular gradient.
- `test_mpnn_message_inference_fusion_is_fullgraph_compilable`: `torch.compile`
  raises `RecursionError` while evaluating the `mpnn_message_inference` operator
  on fake tensors. It also fails when run alone before the rebase.

The rebase is complete, but the branch is **not fully green in CI or GPU tests**.
These failures must be resolved before claiming full MPNN production validation.
