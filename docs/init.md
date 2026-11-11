# Parameter initialisation: presets and config

Every engine module declares the init of each of its projections in code, in team-gm's vocabulary
(`src/miniworld_engine/modules/primitives.py`, `Linear(..., init=...)`). That is the module's default, and **construction never changes**:
building a module draws exactly what it drew before (MiniWorld and every checkpoint path keep reproducing). After construction,
`apply_init(model, cfg)` (`miniworld_engine.modules`, code in `modules/init.py`) re-initialises the model from a **preset** (a YAML file
shipped with the package, or your own) and **overrides** from config. `describe_init(model, cfg)` lists what every parameter gets.

Initialisation does not affect any measurement: the accuracy and speed measurers redraw every floating parameter of every case
(`measure/config.py`, `randomize`: matrices `N(0, 1/fan_in)`, norm gains `1 + 0.1 N`, other vectors `0.2 N`), so class `version` /
`backend_versions`, the judge's tokens and the prebuild plan are untouched by this feature.

## The vocabulary

| `init` | weight | bias it implies |
|---|---|---|
| `default` | LeCun fan-in truncated normal (variance `1/fan_in`, truncated at +-2 sigma) | PyTorch uniform (`default`) |
| `relu` | He: the same with variance `2/fan_in` | `default` |
| `normal` | `kaiming_normal_` (`std = 1/sqrt(fan_in)`) | `default` |
| `glorot` | `xavier_uniform_` (fan average) | `default` |
| `gating` | 0 | 1 |
| `zero` | 0 | 0 |
| `one` | 1 | 0 |

A **bias spec** is `zero` | `one` | `default` (PyTorch `nn.Linear`'s `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, today's behaviour) | a number. In
code, `Linear(..., bias_init=-2.0)` states a bias the init cannot (the conditioning gates of `AugmentedAttentionPairBias` /
`BiasOnlyAttention`, `sigmoid(-2) ~ 0.1`); in a preset or config the same is `{init: default, bias: -2.0}`.

Managed parameters: the weight and bias of every engine `Linear` (`MPLinear` included: it is re-normalised after a draw, and refuses
`zero` / `gating`) and of every `LayerNorm` / `RMSNorm` with an affine (code default `{init: one, bias: zero}`; `init` `one` | `zero`, the
bias `zero` | `one` | a number). Everything else (plain `nn.Linear` / `nn.Embedding` of ProteinMPNN, the raw `nn.Parameter` tables of
PairEmbed* / TokenPairInit) is listed as `unmanaged` and never touched.

## Presets

`src/miniworld_engine/modules/init_presets/<name>.yaml` (package data; `**/*.yaml` in `[tool.setuptools.package-data]`), or any YAML path:

```yaml
description: ...
extends: engine          # optional: another preset (a shipped name or a path, relative to this file); its entries are the base
rng: legacy              # the preset's default rng
entries:
  ConditionedTransition.to_scale: {init: zero, bias: -2.0}
  TokenDiT*.attention.to_out: {init: default}       # wildcards on the class name
```

A key is `<ModuleClass>.<attribute path>`: the path from a module of that class to a `Linear` / `LayerNorm` (`TokenDiTProtenix.transition.squeeze`,
`AdaptiveLayerNorm.to_scale`; a bare `<ModuleClass>` names a `Linear` / `LayerNorm` class itself). The class matches a module of that class
**or a subclass** (any name along its MRO, so `ConditionedTransition.*` also covers `UpGatedConditionedTransition`, and
`_TriangleAttentionBase.*` every triangle attention); `*`, `?`, `[...]` glob the class name. An entry sets `init` (and the bias it
implies, unless it names `bias`), `bias`, or both. Two classes of one name (an engine class and a MiniWorld class) are told apart by
the module-qualified form `<python module>:<Class>`, e.g. `miniworld.modules.bias_only_token_dit:BiasOnlyAttention.to_scale`; it is
more specific than the bare name. A preset covers many models, so a preset entry that matches nothing in the given model (another
model's class, an attribute this model was built without, e.g. `to_query` of a triangle attention without self-attention) is skipped
and listed by `describe_init` as `not present`; only `overrides` are strict.

| preset | what it is | use it when |
|---|---|---|
| `engine` (default) | no entries: whatever the code declares now | new models that follow the engine's defaults; applying it changes nothing |
| `miniworld` | MiniWorld's trained path (phase 1a MiniSWAModel, phase 2 DiffusionModel with the bias-only token DiT, phase 3 ConfidenceModel) as it is today: every engine Linear's current code default and MiniWorld's own layers (module-qualified, citing MiniWorld file:line), written out | reproducing / continuing MiniWorld's trained models: pinned, so a later change of an engine code default does not move them. Equal to `engine` on every engine class today (tested) |
| `af3` | AlphaFold 3's own initialisation of the engine classes (the table below) | training an AF3-style model from scratch with AF3's init |

One caveat for MiniWorld layers: a bias a MiniWorld constructor sets after the `Linear` is built (`bias.fill_(-2.0)` in
`bias_only_token_dit.py:96`) is invisible to the code default `describe_init` reports (`bias default`); the `miniworld` preset states
it (`bias: -2.0`), so `preset: miniworld` keeps it, while `engine` with `reinit: all` would redraw that bias uniform.

## The config block

```yaml
init:
  preset: af3            # engine (default) | miniworld | af3 | a path to a YAML preset
  rng: torch             # legacy | torch (default: the preset's; both shipped presets say legacy)
  seed: 0                # optional
  reinit: changed        # changed (default) | all
  overrides:
    ConditionedTransition.to_scale: {init: zero, bias: -2.0}
    diffusion.token_dit.blocks.3.attention.to_out: {init: default}   # an instance path of the model (model.named_modules())
```

```python
from miniworld_engine.modules import apply_init, describe_init

model = build_model(...)                       # construction as today
apply_init(model, cfg["init"])                 # a dict, a YAML path (a whole config: its `init:` block is read), "af3", or None
print(describe_init(model, cfg["init"]))       # the audit table (log it next to the run)
```

MiniWorld (or any model built from engine modules) calls `apply_init` once after building its model; nothing in the engine calls it.

Typo safety: an unknown key of the block, of an entry or of a preset, an init / bias outside the vocabulary, an override that matches no
parameter of the model, an instance path that is not a `Linear` / `LayerNorm`, or a bias override on a layer without a bias raises a
`ValueError` naming the closest valid targets.

## Precedence

instance path (overrides) > class pattern (overrides) > preset > the module's code default.

Inside one layer the most specific class pattern wins: the longer attribute path (`TokenDiTProtenix.transition.squeeze` over
`ConditionedTransition.squeeze`), then the nearer class in the module's MRO (`UpGatedConditionedTransition.to_scale` over
`ConditionedTransition.to_scale`), then a name without wildcards. Two equally specific keys with different specs are an error. Layers
refine key by key: af3's `ConditionedTransition.to_scale: {init: zero, bias: -2.0}` under an override `{bias: -1.0}` gives weight 0, bias -1.

## Random draws

- `reinit: changed` (default): only parameters whose effective init / bias differs from the code default are redrawn; the rest keep their
  construction values. `apply_init(model, {"preset": "engine", "rng": "legacy"})` changes nothing (tested bit for bit for every
  requirements class).
- `reinit: all`: every managed parameter is redrawn, in construction order (`Linear._init_order`).
- `rng: legacy`: the truncated normals from numpy's global state through scipy (`_trunc_normal_init`, as construction draws them),
  `normal` / `glorot` / bias `default` from torch's global state. With `reinit: all` and the construction's `np.random.seed` /
  `torch.manual_seed`, the discarded `nn.Linear` draws are replayed too, so the engine preset reproduces construction exactly (tested for
  every requirements class except TokenDiTChai1 / AtomDiTChai1, which build two AdaLNs and replace them). With `seed`: a
  `numpy.random.RandomState(seed)` and a forked torch state seeded with it (equal to seeding the globals; the globals are left as they were).
- `rng: torch`: everything from torch: `torch.nn.init.trunc_normal_` with the same standard deviation as `_trunc_normal_init`
  (`trunc_normal_std`) and truncation at +-2 sigma, `kaiming_normal_`, `xavier_uniform_`, the bias uniform; drawn in fp32 on the
  parameter's device and copied into the parameter's dtype. Without `seed` from the global torch generator (`torch.manual_seed` alone
  reproduces the model with `reinit: all`); with `seed` from a `torch.Generator` per device seeded with it.

Initialisation is backend independent: every backend of a class constructs the same parameters (on CPU too), and `apply_init` gives them
the same values (tested for every class and backend).

## `describe_init`

```text
# init: preset af3, rng legacy, reinit changed, seed None, 0 override(s)
parameter                          module      target                                  code          effective   source
attention.ada_ln_in.ln_cond.weight LayerNorm   AdaptiveLayerNorm.ln_cond               one           one         code
attention.ada_ln_in.to_scale.weight Linear     AdaptiveLayerNorm.to_scale              gating        zero        preset:af3
attention.ada_ln_in.to_scale.bias  Linear      AdaptiveLayerNorm.to_scale              bias one      bias zero   preset:af3
attention.to_query.bias            Linear      AugmentedAttentionPairBias.to_query     bias default  bias zero   preset:af3
attention.to_scale.weight          Linear      AugmentedAttentionPairBias.to_scale     default       zero        preset:af3
...
```

`describe_init(model, cfg, rows=True)` returns the rows as dicts (`parameter`, `module`, `target`, `code`, `effective`, `source`): one per
`model.named_parameters()` entry, then one per preset entry the model has no parameter for (`parameter` None, `source` `not present`).

## The af3 preset

AlphaFold 3 JAX (`alphafold3/src/alphafold3/model`): `hm.Linear` defaults to `initializer='linear'` = LeCun fan-in truncated normal = our
`default` (`components/haiku_modules.py:203`, `:318-335`), `'relu'` = He = our `relu`, `final_init` `'zeros'` (`model_config.py:26`);
`use_bias=False` by default (`:205`) and `bias_init` reaches a parameter only with `use_bias=True` (`:306-312`), so AF3's gates are weight 0
with no bias, as ours. Entries (citations in `init_presets/af3.yaml`):

| key | af3 | engine code default | AF3 source |
|---|---|---|---|
| `AdaptiveLayerNorm.to_scale` | `{init: zero, bias: zero}` | `gating` (bias 1) | diffusion_transformer.py:41-46 |
| `AugmentedAttentionPairBias.to_query` | `{bias: zero}` | LeCun, bias uniform | diffusion_transformer.py:147, :293-295 |
| `AugmentedAttentionPairBias.to_bias` | `{init: default}` | `zero` | diffusion_transformer.py:243-246, :393-396 |
| `AugmentedAttentionPairBias.to_out` | `{init: default}` | `zero` | diffusion_transformer.py:65 |
| `AugmentedAttentionPairBias.to_scale` | `{init: zero, bias: -2.0}` | LeCun, bias -2 | diffusion_transformer.py:67-73 |
| `BiasOnlyAttention.to_bias` / `.to_out` / `.to_scale` | as the three above | `zero` / `zero` / LeCun, bias -2 | the same lines |
| `ConditionedTransition.squeeze` | `{init: default}` | `zero` | diffusion_transformer.py:65 |
| `ConditionedTransition.to_scale` | `{init: zero, bias: -2.0}` | LeCun, bias uniform | diffusion_transformer.py:67-73 |
| `AttentionPairBias.to_query` | `{init: default, bias: zero}` | `glorot`, bias uniform | diffusion_transformer.py:147 (via modules.py:515) |
| `AttentionPairBias.to_key` / `.to_value` | `{init: default}` | `glorot` | diffusion_transformer.py:148, :162 |
| `MSAPairWeightedAveraging.to_value` | `{init: default}` | `glorot` | modules.py:111-112 |
| `_TriangleAttentionBase.to_query` / `.to_key` / `.to_value` | `{init: default}` | `glorot` | modules.py:163-169 |
| `BidirectionalTriangleAttention.to_query` / `.to_key` / `.to_value` | `{init: default}` | `glorot` | modules.py:163-169 (two side by side) |

Equal already (no entry): TriangleMultiplication (modules.py:276-327), OuterProductMean (modules.py:358-389), the MSA pair-weighted
averaging logits / gate / output (modules.py:103-128), Transition (modules.py:60-84), the triangle attention's pair bias / gate / output
(modules.py:184-217), the AdaLN bias projection (diffusion_transformer.py:47-49), the conditioned transition's expand (relu,
diffusion_transformer.py:93-98). Classes with no AF3 counterpart (IdentityAdaptiveLayerNorm, AdaptiveRMSNorm, the up-gate of
UpGatedConditionedTransition, the SWA / MiniWorld / ESMFold2 atom blocks) have no entry; the patterns above do reach the other models'
blocks built from the same engine classes (Boltz-2 / RoseTTAFold3 / Chai-1 token and atom DiTs).
