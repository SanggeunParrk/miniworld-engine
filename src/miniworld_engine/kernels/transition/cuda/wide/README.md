# Wide-width sm_90a Transition kernels (D = 64, 256, 384, 512)

Wired through `../fused_wide_sm90a.py`, dispatched from `modules/transition/module.py::_residual_forward` right after the
D = 128 path, under the same `transition_fused_sm90a` setting; opt out with `MINIWORLD_TRANSITION_WIDE_SM90A=0`.

`kernels/` holds the kernel sources exactly as developed in `experiments/transition_fused/src/` (record:
`experiments/transition_fused/records/widths.md`); some keep experiment switches that default off. Each `*.cu` at this
level is one kernel INSTANCE: it pins the build flags of the measured-best variant, renames the kernel symbol per width
and adds a host launcher. `bind.cu` is the torch binding; `-DWIDE_D` selects which launchers a build has, `-DWIDE_SMS`
sizes the persistent grids. One extension per (width, SM count).

| file | kernel | flags |
|---|---|---|
| d64_fwd.cu | transition_fwd_d64 | 2 CTAs/SM, runtime save (`FWD_SAVE 0`: the no_grad call passes 1-element placeholders) |
| d64_bwd.cu | transition_bwd_d64 + reduce_partials | DW_REPL 16 |
| d256_fwd.cu | transition_fwd_d256 | runtime save, PP (ping-pong consumer warpgroups) |
| lnsg.cu | ln_swiglu_gemm | KD = D, runtime save, SWP (software-pipelined epilogue) |
| squeeze.cu | squeeze_gemm | SQ_BN 256 (D % 256 == 0) else 192 -- must be a literal, it is token-pasted; WAIT0 |
| gate.cu | gate_gemm2 | TBK 32, NSTAGE 4, STG_HALF, WAIT0 |
| dxln.cu | dxn_lnbwd | D 256, COLS 2, TBK 64, NSTAGE 4, WAIT0, STGDX (dx staged + TMA-stored) |
