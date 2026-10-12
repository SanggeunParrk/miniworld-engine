# H100: protocol module inference comparison, 2026-10-05

Protocol: A100 branch commit `8e8f8edcaf9ac45f923076e88184296581fd10dc`, [module-comparison-protocol.md](https://github.com/SanggeunParrk/miniworld-engine/blob/8e8f8edcaf9ac45f923076e88184296581fd10dc/docs/gpus/module-comparison-protocol.md). MiniWorld source: `4ee68216232a2085f8fce7737948afb2d338d8d8`. The original campaign recorded 197 inference configurations in 48 groups. After the user clarified mandatory Token DiT bucketing, the production scope is 194 configurations in 45 groups; L200/450/700 remain historical diagnostic rows, excluded from production requirements and aggregates. The campaign ran with concurrent jobs on separate NVIDIA H100 80GB HBM3 GPUs (seven initially), Slurm `h100` / `normal_h100`.

The existing common `benchmarks/runners/bench.py` harness resets the seed for each backend, checks its measured call, warms it, and measures median CUDA-event time with `triton.testing.do_bench` (10 ms warmup, 100 ms sampling). Each reported comparison uses the same allocation for both backends. Backend timings are sequential; this is not an alternating 90-pair timing campaign. Inference uses CUDA Graph replay. MiniWorld/PyTorch request compilation; the actual compilation scope, compiled graph count, parameter/input dtype and execution metadata are preserved. BF16 activations, harness `precision=bf16-mixed`, TF32 allowed. Some module harnesses explicitly pin BF16 matrix parameters with FP32 norm parameters: the precision string alone does not certify FP32 master matrices. Both compared rows must have matching reported parameter/input dtypes.

Token/Bias-only Token DiT use A5 and shared conditioning with 20% key masks; Token heads are 16x48 with QK-norm OFF. Atom cases use actual N1024/2048/4096 at A1/A5, plus local A48/N4096. Local is the AF3 32x128 `cross_attention=True` block. The harness atom `seq_len` is multiplied by eight; the tables explicitly report actual N. Pair widths are 64/128/256/384/512, lengths 128/256/384/512/640/768. Transition uses L384/L768, n=4. OPM/PWA use MSA feature width64, pair width128 and depth1024/2048/4096.

Anthropic revision is `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. All 5,199 imported original files were checked against their original Git blobs during isolated import. The upstream source is unchanged. Anthropic rows disable MiniWorld's MSA overrides; MiniWorld rows mask upstream payload overrides. Named upstream rows are used rather than searching all upstream variants. Token/APB are graph-only Anthropic calls. Compilation failures and separate graph-only retries are retained, never relabeled as compiled.

## Anthropic TriMul binary compatibility

The shipped CUDA13 TriMul cubins crash in `cuModuleLoadData` on this H100 driver, using either CUDA Python or ctypes bindings (diagnostic jobs 21494/21495). No original shipped-payload latency is claimed. TriMul comparison timings use the separately labeled **Anthropic CUDA12.9 rebuild**, whose 36 CUDA/Python source files are byte-identical to the upstream payload and whose directory has no optimization overlay. The existing rebuilt package has 24 units and a recorded CUDA12.9.41 compiler. The harness label is `anthropic_hybrid` because it selects an external payload; this specific TriMul payload contains unchanged upstream algorithms, not the project's optimized K1/K3 overlay. The binding override is scoped to rebuilt baseline processes; MiniWorld uses its default driver binding. Failed nondefault ctypes MiniWorld pilot attempts are excluded and retained in raw history.

## Anthropic TriangleAttention current-ABI rebuild

The original v11 source was built by its own `build_prebuilt.py` with CUDA12.9 / CUTLASS4.2 for Torch2.13+cu129 and CPython312. All four upstream byte vectors passed bitwise equality. The builder's fresh install initially rejected the new ABI binaries because their paths were absent from the shipped binary registry. Only three entries for newly generated `.so` files were appended to the isolated `triattn_native/SHA256SUMS`; existing entries and validation code were preserved. A fresh GPU interpreter then passed `install(check=True)`, including source/binary digest checks and byte gates (job21519). This is a locally rebuilt upstream native baseline, not the original shipped ABI payload. Successful primary TriangleAttention ratios require logs confirming `core tier:triattn_native -> triattn_native`. Earlier flash fallback attempts remain in history and are excluded.

## Benchmark support and adapter

The H100 branch's common runner predates A100's `dit_atom_local` target. An isolated measurement adapter imports A100's local-shape benchmark function and measures the unchanged H100 `LocalDiTBlock` with `cross_attention=True`; it adds configuration/precision registration in that process only. Its source digest is recorded separately because the common source hash describes repository files. Current H100 module adapters do not implement Anthropic Bias-only DiT or SWA DiT, refuse dense Atom at pair width16, and do not expose A100's new Anthropic local composition. Those are integration coverage limits, not a claim that Anthropic has no corresponding kernels. Unavailable baseline cells are retained and receive no ratio. Transition D256 fails in both compiled and graph-only Anthropic processes. TriangleAttention D64 lacks the requested prologue cell; D384 refuses the non-power-of-two H12 layout. D512 native attention-core loading succeeds, but all six lengths fail in the unchanged upstream epilogue: PTXAS C7600 register allocation at target255, followed by `pair_fused:epilogue` refusal after safe-settings compilation also fails. These are full-module failures; core-only byte checks do not establish a working block.

## Module summary

Ratios are baseline ms / MiniWorld ms: greater than one favors MiniWorld. Geometric means exclude unavailable/failed/incompatible rows; the minimum and maximum remain visible. They do not establish a speedup at every shape.

| Module | Comparison baseline | Valid / requested shapes | Geometric mean | Minimum–maximum |
|---|---|---:|---:|---:|
| triangle_multiplication | Anthropic CUDA12.9 rebuild | 24/30 | 1.087x | 0.947–1.312x |
| triangle_multiplication_bidirectional | Anthropic CUDA12.9 rebuild | 12/30 | 1.115x | 0.965–1.201x |
| triangle_attention | Anthropic native CUDA12.9 rebuild | 12/30 | 0.805x | 0.673–1.015x |
| transition | Anthropic upstream | 4/10 | 2.847x | 1.360–6.173x |
| attention_pair_bias | Anthropic upstream | 30/30 | 0.578x | 0.450–0.694x |
| outer_product | Anthropic upstream | 18/18 | 1.498x | 1.257–1.973x |
| msa_pair_weighted_averaging | Anthropic upstream | 18/18 | 1.576x | 1.231–1.911x |
| dit | Anthropic upstream | 6/6 | 0.532x | 0.511–0.555x |
| bias_only_dit | Anthropic upstream | 0/6 | — | — |
| dit_atom | Anthropic upstream | 0/6 | — | — |
| swa_dit | Anthropic upstream | 0/6 | — | — |
| dit_atom_local | Anthropic upstream | 0/4 | — | — |

## Shape latency tables

Unavailable/failed cases remain in the table; their full error text and actual tensor shapes are in the artifact.

### triangle_multiplication

| Variant | L / actual N | A | MiniWorld ms | Anthropic rebuild ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| d128 | 128 | — | 0.031264 | 0.036288 | 1.161x | 0.090944 | False | ok |
| d128 | 256 | — | 0.075488 | 0.086432 | 1.145x | 0.316912 | False | ok |
| d128 | 384 | — | 0.146208 | 0.168000 | 1.149x | 0.760992 | False | ok |
| d128 | 512 | — | 0.250672 | 0.284544 | 1.135x | 1.536736 | False | ok |
| d128 | 640 | — | 0.386848 | 0.434144 | 1.122x | 3.573344 | False | ok |
| d128 | 768 | — | 0.547872 | 0.613264 | 1.119x | 6.203936 | False | ok |
| d64 | 128 | — | 0.019904 | 0.026112 | 1.312x | 0.062080 | False | ok |
| d64 | 256 | — | 0.041280 | 0.050720 | 1.229x | 0.156576 | False | ok |
| d64 | 384 | — | 0.079744 | 0.093088 | 1.167x | 0.363936 | False | ok |
| d64 | 512 | — | 0.140704 | 0.158432 | 1.126x | 0.752128 | False | ok |
| d64 | 640 | — | 0.215856 | 0.237952 | 1.102x | 1.586144 | False | ok |
| d64 | 768 | — | 0.301056 | 0.328960 | 1.093x | 2.707168 | False | ok |
| d256 | 128 | — | 0.049216 | 0.055328 | 1.124x | 0.158624 | False | ok |
| d256 | 256 | — | 0.166176 | 0.176512 | 1.062x | 0.625280 | False | ok |
| d256 | 384 | — | 0.357296 | 0.370336 | 1.036x | 1.442944 | False | ok |
| d256 | 512 | — | 0.608096 | 0.649632 | 1.068x | 2.923104 | False | ok |
| d256 | 640 | — | 0.995360 | 0.976832 | 0.981x | 7.137376 | False | ok |
| d256 | 768 | — | 1.464128 | 1.392400 | 0.951x | 12.384272 | False | ok |
| d384 | 128 | — | 0.096880 | 0.104256 | 1.076x | 0.238016 | False | ok |
| d384 | 256 | — | 0.349664 | 0.364768 | 1.043x | 0.949936 | False | ok |
| d384 | 384 | — | 0.774464 | 0.767296 | 0.991x | 2.126176 | False | ok |
| d384 | 512 | — | 1.291968 | 1.320608 | 1.022x | 4.441760 | False | ok |
| d384 | 640 | — | 2.143328 | 2.168976 | 1.012x | 11.045296 | False | ok |
| d384 | 768 | — | 3.173472 | 3.003888 | 0.947x | 19.004192 | False | ok |
| d512 | 128 | — | 0.254336 | — | — | 0.322144 |  | Anthropic: unsupported |
| d512 | 256 | — | 0.961792 | — | — | 1.282432 |  | Anthropic: unsupported |
| d512 | 384 | — | 1.282080 | — | — | 2.939168 |  | Anthropic: unsupported |
| d512 | 512 | — | 3.824608 | — | — | 6.116496 |  | Anthropic: unsupported |
| d512 | 640 | — | 6.153296 | — | — | 14.983968 |  | Anthropic: unsupported |
| d512 | 768 | — | 5.043744 | — | — | 25.830816 |  | Anthropic: unsupported |

### triangle_multiplication_bidirectional

| Variant | L / actual N | A | MiniWorld ms | Anthropic rebuild ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| d128 | 128 | — | 0.048480 | 0.046784 | 0.965x | 0.146656 | False | ok |
| d128 | 256 | — | 0.130784 | 0.139472 | 1.066x | 0.550320 | False | ok |
| d128 | 384 | — | 0.250688 | 0.282944 | 1.129x | 1.293872 | False | ok |
| d128 | 512 | — | 0.420352 | 0.470080 | 1.118x | 2.653232 | False | ok |
| d128 | 640 | — | 0.654512 | 0.739168 | 1.129x | 6.642016 | False | ok |
| d128 | 768 | — | 0.974304 | 1.089904 | 1.119x | 11.712672 | False | ok |
| d64 | 128 | — | 0.027872 | 0.033472 | 1.201x | 0.089504 | False | ok |
| d64 | 256 | — | 0.058784 | 0.070368 | 1.197x | 0.271840 | False | ok |
| d64 | 384 | — | 0.119232 | 0.137584 | 1.154x | 0.669456 | False | ok |
| d64 | 512 | — | 0.208272 | 0.235040 | 1.129x | 1.388832 | False | ok |
| d64 | 640 | — | 0.324896 | 0.357696 | 1.101x | 3.246496 | False | ok |
| d64 | 768 | — | 0.465344 | 0.505792 | 1.087x | 5.798112 | False | ok |
| d256 | 128 | — | 0.139584 | — | — | 0.273728 |  | Anthropic: unsupported |
| d256 | 256 | — | 0.492512 | — | — | 1.107264 |  | Anthropic: unsupported |
| d256 | 384 | — | 0.584992 | — | — | 2.563744 |  | Anthropic: unsupported |
| d256 | 512 | — | 1.919232 | — | — | 5.465440 |  | Anthropic: unsupported |
| d256 | 640 | — | 3.043728 | — | — | 13.767936 |  | Anthropic: unsupported |
| d256 | 768 | — | 2.523136 | — | — | 24.113248 |  | Anthropic: unsupported |
| d384 | 128 | — | 0.265424 | — | — | 0.438480 |  | Anthropic: unsupported |
| d384 | 256 | — | 0.914928 | — | — | 1.703232 |  | Anthropic: unsupported |
| d384 | 384 | — | 1.163728 | — | — | 3.928928 |  | Anthropic: unsupported |
| d384 | 512 | — | 3.622272 | — | — | 8.377216 |  | Anthropic: unsupported |
| d384 | 640 | — | 5.890080 | — | — | 21.342960 |  | Anthropic: unsupported |
| d384 | 768 | — | 4.705808 | — | — | 37.129425 |  | Anthropic: unsupported |
| d512 | 128 | — | 0.426704 | — | — | 0.602496 |  | Anthropic: unsupported |
| d512 | 256 | — | 1.581536 | — | — | 2.348368 |  | Anthropic: unsupported |
| d512 | 384 | — | 1.911616 | — | — | 5.393728 |  | Anthropic: unsupported |
| d512 | 512 | — | 6.491776 | — | — | 11.510832 |  | Anthropic: unsupported |
| d512 | 640 | — | 10.584768 | — | — | 29.183489 |  | Anthropic: unsupported |
| d512 | 768 | — | 8.150048 | — | — | 50.224865 |  | Anthropic: unsupported |

### triangle_attention

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| d128 | 128 | — | 0.056096 | 0.051232 | 0.913x | 0.100480 | True | ok |
| d128 | 256 | — | 0.184928 | 0.169600 | 0.917x | 0.478064 | True | ok |
| d128 | 384 | — | 0.395680 | 0.401472 | 1.015x | 1.637376 | True | ok |
| d128 | 512 | — | 0.955968 | 0.723232 | 0.757x | 3.257120 | True | ok |
| d128 | 640 | — | 1.606048 | 1.164832 | 0.725x | 8.331936 | True | ok |
| d128 | 768 | — | 1.927344 | 1.746144 | 0.906x | 13.046112 | True | ok |
| d64 | 128 | — | 0.039808 | — | — | 0.061792 |  | Anthropic: failed |
| d64 | 256 | — | 0.103232 | — | — | 0.237920 |  | Anthropic: failed |
| d64 | 384 | — | 0.241760 | — | — | 0.824896 |  | Anthropic: failed |
| d64 | 512 | — | 0.486416 | — | — | 1.505952 |  | Anthropic: failed |
| d64 | 640 | — | 0.832464 | — | — | 3.814112 |  | Anthropic: failed |
| d64 | 768 | — | 1.280512 | — | — | 6.564096 |  | Anthropic: failed |
| d256 | 128 | — | 0.091264 | 0.076032 | 0.833x | 0.185312 | True | ok |
| d256 | 256 | — | 0.396128 | 0.307840 | 0.777x | 1.169376 | True | ok |
| d256 | 384 | — | 0.981856 | 0.757568 | 0.772x | 3.616176 | True | ok |
| d256 | 512 | — | 1.907008 | 1.390816 | 0.729x | 7.811696 | True | ok |
| d256 | 640 | — | 3.277120 | 2.329600 | 0.711x | 16.539167 | True | ok |
| d256 | 768 | — | 5.171264 | 3.481760 | 0.673x | 26.161345 | True | ok |
| d384 | 128 | — | 0.139424 | — | — | 0.269280 |  | Anthropic: failed |
| d384 | 256 | — | 0.607360 | — | — | 1.998080 |  | Anthropic: failed |
| d384 | 384 | — | 1.455792 | — | — | 5.923392 |  | Anthropic: failed |
| d384 | 512 | — | 2.887280 | — | — | 16.920736 |  | Anthropic: failed |
| d384 | 640 | — | 4.995264 | — | — | 31.664896 |  | Anthropic: failed |
| d384 | 768 | — | 8.050016 | — | — | 55.018497 |  | Anthropic: failed |
| d512 | 128 | — | 0.184832 | — | — | 0.451456 |  | Anthropic: failed |
| d512 | 256 | — | 0.794752 | — | — | 3.134096 |  | Anthropic: failed |
| d512 | 384 | — | 1.985664 | — | — | 9.645456 |  | Anthropic: failed |
| d512 | 512 | — | 3.983088 | — | — | 22.700048 |  | Anthropic: failed |
| d512 | 640 | — | 7.019024 | — | — | 42.417345 |  | Anthropic: failed |
| d512 | 768 | — | 11.158272 | — | — | 73.709152 |  | Anthropic: failed |

### transition

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| d128 | 384 | — | 0.120672 | 0.164064 | 1.360x | 0.434624 | True | ok |
| d128 | 768 | — | 0.438160 | 0.607792 | 1.387x | 1.653920 | True | ok |
| d64 | 384 | — | 0.057696 | — | — | 0.219904 |  | Anthropic: unsupported |
| d64 | 768 | — | 0.176512 | — | — | 0.837008 |  | Anthropic: unsupported |
| d256 | 384 | — | 0.384032 | — | — | 0.845952 | — | Anthropic: build unavailable / not measured |
| d256 | 768 | — | 1.557728 | — | — | 3.292704 | — | Anthropic: build unavailable / not measured |
| d384 | 384 | — | 0.886240 | 5.471040 | 6.173x | 1.464832 | True | ok |
| d384 | 768 | — | 3.832512 | 21.632048 | 5.644x | 5.732800 | True | ok |
| d512 | 384 | — | 1.478464 | — | — | 2.170704 |  | Anthropic: unsupported |
| d512 | 768 | — | 5.819040 | — | — | 8.759232 |  | Anthropic: unsupported |

### attention_pair_bias

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| h8d384 | 128 | — | 0.040288 | 0.027360 | 0.679x | 0.043072 | False | ok |
| h8d384 | 256 | — | 0.055680 | 0.037024 | 0.665x | 0.062112 | False | ok |
| h8d384 | 384 | — | 0.086080 | 0.048480 | 0.563x | 0.096352 | False | ok |
| h8d384 | 512 | — | 0.125568 | 0.063392 | 0.505x | 0.142784 | False | ok |
| h8d384 | 640 | — | 0.168944 | 0.080512 | 0.477x | 0.192416 | False | ok |
| h8d384 | 768 | — | 0.220928 | 0.099328 | 0.450x | 0.250176 | False | ok |
| h16d384 | 128 | — | 0.041248 | 0.028096 | 0.681x | 0.042976 | False | ok |
| h16d384 | 256 | — | 0.057088 | 0.038656 | 0.677x | 0.062208 | False | ok |
| h16d384 | 384 | — | 0.090816 | 0.053248 | 0.586x | 0.097440 | False | ok |
| h16d384 | 512 | — | 0.133280 | 0.072992 | 0.548x | 0.145280 | False | ok |
| h16d384 | 640 | — | 0.181792 | 0.094144 | 0.518x | 0.201568 | False | ok |
| h16d384 | 768 | — | 0.237792 | 0.116288 | 0.489x | 0.263904 | False | ok |
| h12d384 | 128 | — | 0.041696 | 0.027776 | 0.666x | 0.044768 | False | ok |
| h12d384 | 256 | — | 0.061408 | 0.038400 | 0.625x | 0.068160 | False | ok |
| h12d384 | 384 | — | 0.092896 | 0.053024 | 0.571x | 0.104448 | False | ok |
| h12d384 | 512 | — | 0.134752 | 0.071648 | 0.532x | 0.152384 | False | ok |
| h12d384 | 640 | — | 0.171936 | 0.092320 | 0.537x | 0.196480 | False | ok |
| h12d384 | 768 | — | 0.227808 | 0.116480 | 0.511x | 0.259712 | False | ok |
| h24d384 | 128 | — | 0.040128 | 0.027744 | 0.691x | 0.043008 | False | ok |
| h24d384 | 256 | — | 0.055936 | 0.038816 | 0.694x | 0.063136 | False | ok |
| h24d384 | 384 | — | 0.088896 | 0.053856 | 0.606x | 0.101952 | False | ok |
| h24d384 | 512 | — | 0.130336 | 0.073440 | 0.563x | 0.151040 | False | ok |
| h24d384 | 640 | — | 0.179360 | 0.096032 | 0.535x | 0.206304 | False | ok |
| h24d384 | 768 | — | 0.240352 | 0.122592 | 0.510x | 0.280512 | False | ok |
| h16d512 | 128 | — | 0.042368 | 0.028976 | 0.684x | 0.043840 | False | ok |
| h16d512 | 256 | — | 0.058464 | 0.039488 | 0.675x | 0.064256 | False | ok |
| h16d512 | 384 | — | 0.089856 | 0.054560 | 0.607x | 0.100256 | False | ok |
| h16d512 | 512 | — | 0.129952 | 0.074080 | 0.570x | 0.146944 | False | ok |
| h16d512 | 640 | — | 0.177184 | 0.096032 | 0.542x | 0.203664 | False | ok |
| h16d512 | 768 | — | 0.231072 | 0.118976 | 0.515x | 0.263744 | False | ok |

### outer_product

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| msa1024 | 128 | — | 0.113248 | 0.168640 | 1.489x | 0.134944 | True | ok |
| msa1024 | 256 | — | 0.319168 | 0.566272 | 1.774x | 0.444528 | True | ok |
| msa1024 | 384 | — | 0.657536 | 1.219344 | 1.854x | 0.922736 | True | ok |
| msa1024 | 512 | — | 1.120400 | 2.105600 | 1.879x | 1.581600 | True | ok |
| msa1024 | 640 | — | 1.639136 | 3.234688 | 1.973x | 2.403248 | True | ok |
| msa1024 | 768 | — | 2.366144 | 4.625152 | 1.955x | 3.367120 | True | ok |
| msa2048 | 128 | — | 0.171840 | 0.230784 | 1.343x | 0.218496 | True | ok |
| msa2048 | 256 | — | 0.524928 | 0.766512 | 1.460x | 0.709216 | True | ok |
| msa2048 | 384 | — | 1.134720 | 1.622416 | 1.430x | 1.423168 | True | ok |
| msa2048 | 512 | — | 1.802848 | 2.792352 | 1.549x | 2.421920 | True | ok |
| msa2048 | 640 | — | 3.035904 | 4.415936 | 1.455x | 3.626496 | True | ok |
| msa2048 | 768 | — | 4.305312 | 6.400144 | 1.487x | 5.153152 | True | ok |
| msa4096 | 128 | — | 0.279712 | 0.358848 | 1.283x | 0.394848 | True | ok |
| msa4096 | 256 | — | 0.951520 | 1.200384 | 1.262x | 1.218432 | True | ok |
| msa4096 | 384 | — | 1.938224 | 2.568512 | 1.325x | 2.440992 | True | ok |
| msa4096 | 512 | — | 3.592608 | 4.516368 | 1.257x | 4.214656 | True | ok |
| msa4096 | 640 | — | 5.685440 | 7.199360 | 1.266x | 6.608608 | True | ok |
| msa4096 | 768 | — | 8.021632 | 10.165952 | 1.267x | 9.294144 | True | ok |

### msa_pair_weighted_averaging

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| msa1024 | 128 | — | 0.119744 | 0.219424 | 1.832x | 0.315072 | True | ok |
| msa1024 | 256 | — | 0.240608 | 0.439296 | 1.826x | 0.631168 | True | ok |
| msa1024 | 384 | — | 0.404608 | 0.701456 | 1.734x | 0.969728 | True | ok |
| msa1024 | 512 | — | 0.635216 | 1.008704 | 1.588x | 1.317984 | True | ok |
| msa1024 | 640 | — | 0.978272 | 1.343872 | 1.374x | 1.714880 | True | ok |
| msa1024 | 768 | — | 1.297392 | 1.721440 | 1.327x | 2.131904 | True | ok |
| msa2048 | 128 | — | 0.214208 | 0.403296 | 1.883x | 0.612272 | True | ok |
| msa2048 | 256 | — | 0.450944 | 0.823072 | 1.825x | 1.207520 | True | ok |
| msa2048 | 384 | — | 0.777408 | 1.321040 | 1.699x | 1.863040 | True | ok |
| msa2048 | 512 | — | 1.227840 | 1.897936 | 1.546x | 2.517552 | True | ok |
| msa2048 | 640 | — | 1.902400 | 2.507760 | 1.318x | 3.349152 | True | ok |
| msa2048 | 768 | — | 2.521568 | 3.212944 | 1.274x | 4.075744 | True | ok |
| msa4096 | 128 | — | 0.403104 | 0.770272 | 1.911x | 1.193440 | True | ok |
| msa4096 | 256 | — | 0.867936 | 1.600576 | 1.844x | 2.375696 | True | ok |
| msa4096 | 384 | — | 1.511056 | 2.562752 | 1.696x | 3.642336 | True | ok |
| msa4096 | 512 | — | 2.398368 | 3.678944 | 1.534x | 4.935648 | True | ok |
| msa4096 | 640 | — | 3.812416 | 4.800608 | 1.259x | 6.473120 | True | ok |
| msa4096 | 768 | — | 4.991264 | 6.145920 | 1.231x | 7.961472 | True | ok |

### dit

L200/450/700 below are historical diagnostic measurements. Production Token DiT always uses bucketing; exclude these three rows from its coverage requirements and aggregate.

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| main | 128 | 5 | 0.175840 | 0.089792 | 0.511x | 0.105232 | False | ok |
| main | 256 | 5 | 0.207392 | 0.113472 | 0.547x | 0.161024 | False | ok |
| main | 384 | 5 | 0.264928 | 0.141632 | 0.535x | 0.244096 | False | ok |
| main | 512 | 5 | 0.327936 | 0.181888 | 0.555x | 0.343952 | False | ok |
| main | 640 | 5 | 0.407968 | 0.216800 | 0.531x | 0.472432 | False | ok |
| main | 768 | 5 | 0.485968 | 0.249872 | 0.514x | 0.618592 | False | ok |
| pad200 | 200 | 5 | 0.140416 | 0.107200 | 0.763x | 0.137056 | False | ok |
| pad450 | 450 | 5 | 0.281840 | 0.183904 | 0.653x | 0.323632 | False | ok |
| pad700 | 700 | 5 | 0.452000 | 0.234432 | 0.519x | 0.573600 | False | ok |

### bias_only_dit

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| main | 128 | 5 | 0.085888 | — | — | 0.086080 |  | Anthropic: unsupported |
| main | 256 | 5 | 0.119648 | — | — | 0.118848 |  | Anthropic: unsupported |
| main | 384 | 5 | 0.171392 | — | — | 0.170720 |  | Anthropic: unsupported |
| main | 512 | 5 | 0.241728 | — | — | 0.241792 |  | Anthropic: unsupported |
| main | 640 | 5 | 0.305056 | — | — | 0.303968 |  | Anthropic: unsupported |
| main | 768 | 5 | 0.389280 | — | — | 0.390240 |  | Anthropic: unsupported |

### dit_atom

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| a1n1024_2048 | 1024 | 1 | 0.172224 | — | — | 0.155744 |  | Anthropic: unsupported |
| a1n1024_2048 | 2048 | 1 | 0.473792 | — | — | 0.424672 |  | Anthropic: unsupported |
| a1n4096 | 4096 | 1 | 1.599104 | — | — | 1.370512 |  | Anthropic: unsupported |
| a5n1024_2048 | 1024 | 5 | 0.189056 | — | — | 0.244544 |  | Anthropic: unsupported |
| a5n1024_2048 | 2048 | 5 | 0.563936 | — | — | 0.819328 |  | Anthropic: unsupported |
| a5n4096 | 4096 | 5 | 1.963200 | — | — | 3.044480 |  | Anthropic: unsupported |

### swa_dit

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| a1n1024_2048 | 1024 | 1 | 0.047072 | — | — | 0.092288 |  | Anthropic: unsupported |
| a1n1024_2048 | 2048 | 1 | 0.057216 | — | — | 0.153632 |  | Anthropic: unsupported |
| a1n4096 | 4096 | 1 | 0.086976 | — | — | 0.336032 |  | Anthropic: unsupported |
| a5n1024_2048 | 1024 | 5 | 0.108416 | — | — | 0.139008 |  | Anthropic: unsupported |
| a5n1024_2048 | 2048 | 5 | 0.180032 | — | — | 0.402368 |  | Anthropic: unsupported |
| a5n4096 | 4096 | 5 | 0.324736 | — | — | 1.196992 |  | Anthropic: unsupported |

### dit_atom_local

| Variant | L / actual N | A | MiniWorld ms | Anthropic ms | Anthropic / MiniWorld | PyTorch ms | Anthropic compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| a5n2048 | 2048 | 5 | 0.205584 | — | — | 0.201232 |  | Anthropic: failed |
| a5n4096 | 4096 | 5 | 0.342368 | — | — | 0.332800 |  | Anthropic: failed |
| a1n4096 | 4096 | 1 | 0.141360 | — | — | 0.147040 |  | Anthropic: failed |
| a48n4096 | 4096 | 48 | 2.525008 | — | — | 2.705792 |  | Anthropic: failed |

## Evidence and limits

This is an inference-only comparison. Current engine autotune-cache warnings report stale Triton/compiler metadata and the normal heuristic miss-cap24; no fresh exhaustive autotuning is claimed. It does not measure training/backward, optimizer updates, SOL90, every upstream variant, or the entire model. Atom lengths and sample counts are not reduced to fill failed cells. Successful rows retain the harness's per-row validation evidence; this campaign is not a new exhaustive sanitizer/gradient qualification. No production kernel or dispatch changes were made for the comparison.

Token pair inputs are fixed across replay; upstream/engine pair-bias caches can therefore be reused. The ratios do not describe a workload whose pair changes on every call. First-time compilation, source import, and payload preparation are outside steady-state latency. Different shapes can use different allocations, but each reported backend ratio uses the same allocation. The final D512 TriangleAttention cases are split into six concurrent length jobs; warmed compiler caches are copied into isolated per-job directories and preparation remains outside measured latency.

[Full measurements, commands, provenance, tensor shapes, and failed attempts](data/anthropic_comparison/protocol_20261005.json). Experiment scripts and logs: `experiments/h100_anthropic_protocol_20261005`.

## Attention dispatch clarification

The raw Token DiT and TriangleAttention execution-path strings understate existing H100 native coverage. [The source and GPU dispatch audit](attention_dispatch_audit_20261005.md) distinguishes native CUDA, Triton, external FlashAttention, and reference block routes. Reported latencies and ratios are unchanged.
