# FP8 training kernels (H100, opt-in, 3.1.0 line, EXPERIMENTAL)

FP8 inside the GEMMs of the training step (forward + backward). Everything here is opt-in, default OFF, separate from the inference
`safe` preset (`miniworld_engine.fp8`, docs/fp8.md). With the switches off the training route is the 2.5.14 one, bit for bit.

## MSAPairWeightedAveraging (`kernels/integrations/pwa_fp8_train.py`, branch `fp8t/pwa`, base 1a2bbe42e = rel/3.0.0)

PWA_SECTION_BODY
