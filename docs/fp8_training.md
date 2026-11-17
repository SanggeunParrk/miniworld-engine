# FP8 training routes (H100, opt-in, engine 3.1.0 line)

Everything here is EXPERIMENTAL, opt-in and off by default. Base: `rel/3.0.0` 1a2bbe42e (= 2.5.14 + the inference FP8 routes,
[fp8.md](fp8.md)). With no training switch set every route is bit-identical to 2.5.14 (sha1 of the output and every gradient, separate
processes). The training switch is separate from the inference `MINIWORLD_FP8` presets.

## Token DiT (`conditioned_transition.cuda.token_dit_fp8_train`, branch `fp8t/tokendit`)

(work in progress -- filled below)
