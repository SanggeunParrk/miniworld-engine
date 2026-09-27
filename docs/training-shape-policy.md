# Training and inference shape policy

Updated 2026-09-20 to match the training recipe.

| Stream | Training lengths | Inference build lengths (unchanged) |
| --- | --- | --- |
| Token pair / token single / MSA token | **384, 768** | 128, 256, 384, 512, 640, 768 |
| Atom single / atom pair | **4096, 8192** | 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192 |
| Noise conditioning vector | 1 | 1 |

These are length axes, not channel dimensions. Token and atom streams are
enumerated separately; this does not impose a new cross-product of molecule
shapes or change the model's augmentation count. Small correctness probes
(for example B1–B4 L64) are still useful and are not training cache buckets.

## Where the policy is applied

- `autotune/module_registry.py` keeps the inference ladders and declares
  `TRAIN_TOKEN_LENGTHS`, `TRAIN_ATOM_LENGTHS`, and `TRAIN_STREAM_LADDERS`.
  `ModuleRow.lengths_for(mode)` selects the appropriate subset.
- `derive.units` and the module builder use those mode-specific lengths for
  both training forward and backward, including dropout/dispatch variants.
- The legacy Case builder, cache replay audit, and CLI caller audit honor
  the same policy; they cannot reintroduce intermediate training lengths.
- `kernels/registry.csv` explicitly declares `build_modes`. Backward-only
  driver kernels use `train`, and their work list is limited to the two
  lengths on each side. Shared forward kernels use `eval|train` and retain
  the inference work list. A `saveact` or `train` name is not sufficient to
  classify a forward helper: autograd forward may run under `no_grad` too.

## Cache compatibility

Runtime shape-key buckets, packing, tuning configurations, channel-width
coverage, kernel source, and `build_rev` are unchanged. Existing tuning
results remain usable; no cache entries are deleted or rebuilt by this edit.
The source-dependent **derived build plan** becomes stale and is regenerated
by the normal `build all` workflow. The next build then fills only required
missing entries. No new GPU cache-build job was launched for this policy edit.

The MiniWorld recipes already pad phase-a batches to384/4096 and phase-b
batches to768/8192. No running training job or inference padding was changed.

## Verification

CPU plan enumeration on node02, with the SM90 build matrix, before set-cover
selection or skipping cached work:

| Candidate work list | Before | After |
| --- | ---: | ---: |
| Training module invocations | 6756 | **2188** |
| Inference module invocations | 2350 | **2350** |
| Backward-only driver units | 2238 | **782** |
| Shared-forward driver units | 2843 | **2843** |

All inference module labels and shared-forward driver unit records were
compared with the pre-edit snapshot and are identical. Training retains
augmentation48 and dropout0.25 variants. Noise length1 remains separate.

559 tests passed: training shape policy, model shape contracts, and the
module-registry contract suite. This is a scheduling/configuration change;
no GPU kernel implementation changed and no new speedup is claimed.

```bash
python -m pytest -q \
  tests/builder/test_training_shape_policy.py \
  tests/builder/test_model_shape_contracts.py \
  tests/registry/test_the_module_registry_says_what_the_model_runs.py
```
