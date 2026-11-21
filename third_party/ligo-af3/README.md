# Ligo AlphaFold3 (Apache-2.0) - MSA pair weighted averaging

`src/models/components/msa_kernel.py` (`MSAWeightedAveragingFused`, credited to Alex Zhang): Triton forward kernel for `sigmoid(g) * softmax(b) @ v` (v / g `[B, S, R, H, C]`,
b `[B, R, R, H]`) with an einsum backward. Our class: `MSAPairWeightedAveraging` (backend `ligo_pwa`, forward + backward). No key mask in the kernel: the adapter masks the bias logits.
Upstream allocates the accumulators with `torch.empty` and reads them before writing (uninitialised memory): `patches/ligo-msa-kernel-fixes.patch` (applied by fetch.sh) makes them `torch.zeros`.
The kernel takes fp32 (its `tl.dot` mixes the fp32 softmax weights with v: a bf16 v fails to compile), so the adapter casts v / b / g to fp32 (TF32 dots) and the result back: its speed is that of an fp32 naive-channel-loop kernel (far slower than the engine). No Transition kernel in this repo (its Transition is plain torch / a Triton SwiGLU only in MegaFold).
