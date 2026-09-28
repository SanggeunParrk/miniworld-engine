# H100 experiments excluded from routine optimization

- `transition_expand_gate_sm90_cuda`: developed=no excludes the standalone driver from build all.
  Commit b2ce8134 measured 0.98x/0.79x/0.55x against CuTe at d128/256/512.
  The direct API remains available to reproduce the experiment.
- M2 `lnl_ws=1`: retain as an explicitly selected historical experiment, default remains 0.
  a806c93b measured 5.8 ms versus M1 0.61 ms at d768, M262144.
- AdaLN CUTLASS TF32 source/sweeps: no production dispatch or routine builder entry.
  71fee1c8 measured the forward slower than the existing engine at every tested atom/token size.

Keep M2 with lnl_ws=0, TM2, CuTe gate backward, DAB+LN backward, and split-back experiments
available for qualified shape-specific comparisons. Inactive is not synonymous with slow.
