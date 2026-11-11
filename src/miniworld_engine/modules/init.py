"""Parameter initialisation from presets and config: :func:`apply_init` re-initialises a constructed model, :func:`describe_init` lists it.

Every :class:`~miniworld_engine.modules.primitives.Linear` declares its own init in code (``Linear(..., init="gating")``, the team-gm
vocabulary ``default | relu | normal | glorot | gating | zero | one``, plus ``bias_init=`` where the vocabulary has no word for the bias,
e.g. a conditioning gate's -2.0). That is the module's default, and construction is unchanged by this file. A *preset* (a YAML file:
``init_presets/<name>.yaml`` or a user path) maps ``<ModuleClass>.<attribute path>`` to ``{init: <name>, bias: <spec>}``; a config block
picks a preset and overrides entries::

    init:
      preset: af3                 # engine (default: empty = the code defaults) | miniworld | af3 | a path to a YAML preset
      rng: torch                  # legacy (numpy-global truncated normal, as construction draws) | torch; default: the preset's (legacy)
      seed: 0                     # optional: draw from generators seeded here instead of the global random states
      reinit: changed             # changed (default): only parameters whose effective init differs from the code default | all
      overrides:
        ConditionedTransition.to_scale: {init: zero, bias: -2.0}
        diffusion.token_dit.blocks.3.attention.to_out: {init: default}   # an instance path of the model

Precedence: instance path (overrides) > class pattern (overrides) > preset > the code default. Within a layer the most specific class
pattern wins: the longer attribute path (more context), then the nearer class in the module's MRO, then a name without wildcards; a tie
between different specs is an error. A layer's entry sets the init (and the bias the init implies, unless it names ``bias``) and / or the
bias; a later layer refines an earlier one key by key. Managed parameters: the weight and bias of every engine ``Linear`` (``MPLinear``
included) and of every LayerNorm / RMSNorm with an affine (code default ``{init: one, bias: zero}``). Anything else (plain ``nn.Linear``,
embeddings, raw ``nn.Parameter`` tables) is listed by :func:`describe_init` as ``unmanaged`` and never touched. See docs/init.md.
"""

from __future__ import annotations

import contextlib
import difflib
import fnmatch
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import torch
import torch.nn as nn

from miniworld_engine.modules.primitives import (
    INIT_NAMES,
    BackendModule,
    InitType,
    Linear,
    MasterWeights,
    MPLinear,
    _trunc_normal_init,
    fill_bias,
    implied_bias,
    trunc_normal_std,
)

__all__ = ["PRESET_DIR", "apply_init", "describe_init", "load_init_config", "load_preset", "preset_names"]

#: The shipped presets (package data): ``<name>.yaml``.
PRESET_DIR = Path(__file__).with_name("init_presets")
RNGS = ("legacy", "torch")
REINITS = ("changed", "all")
CONFIG_KEYS = ("preset", "rng", "seed", "reinit", "overrides")
PRESET_KEYS = ("description", "extends", "rng", "entries")
SPEC_KEYS = ("init", "bias")
BIAS_WORDS = ("zero", "one", "default")
NORM_INITS = ("one", "zero")
_NOT_NAMES = (BackendModule, MasterWeights)


# ------------------------------------------------------------------------------------------------------------------------ the spec
@dataclass(frozen=True)
class Spec:
    """One entry: ``init`` (a vocabulary name) and / or ``bias`` (``"zero" | "one" | "default"`` or a value); ``None`` = not given."""

    init: str | None = None
    bias: str | float | None = None

    def __str__(self) -> str:
        parts = ([f"init: {self.init}"] if self.init is not None else []) + ([f"bias: {self.bias}"] if self.bias is not None else [])
        return "{" + ", ".join(parts) + "}"


def _parse_spec(key: str, value: Any, where: str) -> Spec:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where}: {key!r} must map to {{init: <name>, bias: <spec>}} (either key), got {value!r}")
    unknown = sorted(set(value) - set(SPEC_KEYS))
    if unknown:
        raise ValueError(f"{where}: {key!r} has unknown key(s) {unknown}; an entry takes {list(SPEC_KEYS)}")
    init = value.get("init")
    if init is not None and init not in INIT_NAMES:
        raise ValueError(f"{where}: {key!r} init {init!r} is not one of {list(INIT_NAMES)}")
    bias = value.get("bias")
    if bias is not None:
        if isinstance(bias, bool) or not (bias in BIAS_WORDS or isinstance(bias, (int, float))):
            raise ValueError(f"{where}: {key!r} bias {bias!r} is not one of {list(BIAS_WORDS)} or a number")
        if isinstance(bias, int):
            bias = float(bias)
    return Spec(init, bias)


# -------------------------------------------------------------------------------------------------------------------- presets / config
def preset_names() -> list[str]:
    """The shipped presets."""
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def _read_yaml(path: Path) -> Any:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def _preset_path(name: str | Path, base: Path | None = None) -> Path:
    if isinstance(name, str) and name in preset_names():
        return PRESET_DIR / f"{name}.yaml"
    path = Path(name).expanduser()
    if not path.is_absolute() and base is not None and (base / path).exists():
        path = base / path
    if not path.is_file():
        raise ValueError(f"init preset {str(name)!r}: neither a shipped preset ({', '.join(preset_names())}) nor a YAML file")
    return path


def load_preset(name: str | Path, _seen: tuple[Path, ...] = (), _base: Path | None = None) -> tuple[dict[str, Spec], str | None]:
    """A preset's entries (its ``extends`` chain resolved: a derived entry replaces the base's entry of the same key) and its ``rng``."""
    path = _preset_path(name, _base).resolve()
    if path in _seen:
        raise ValueError(f"init preset {path}: an extends cycle ({' -> '.join(map(str, (*_seen, path)))})")
    data = _read_yaml(path) or {}
    if not isinstance(data, dict):
        raise ValueError(f"init preset {path}: the file must be a mapping of {list(PRESET_KEYS)}")
    unknown = sorted(set(data) - set(PRESET_KEYS))
    if unknown:
        raise ValueError(f"init preset {path}: unknown key(s) {unknown}; a preset takes {list(PRESET_KEYS)}")
    entries: dict[str, Spec] = {}
    rng = None
    if data.get("extends") is not None:
        entries, rng = load_preset(data["extends"], (*_seen, path), path.parent)
    raw = data.get("entries") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"init preset {path}: entries must be a mapping")
    for key, value in raw.items():
        if not _is_pattern(str(key)):
            raise ValueError(f"init preset {path}: {key!r} is not <ModuleClass>[.<attribute path>] (a preset names classes, not instances)")
        entries[str(key)] = _parse_spec(str(key), value, f"init preset {path.name}")
    if data.get("rng") is not None:
        rng = data["rng"]
        if rng not in RNGS:
            raise ValueError(f"init preset {path}: rng {rng!r} is not one of {list(RNGS)}")
    return entries, rng


def load_init_config(cfg: dict | str | Path | None) -> dict:
    """The init block as a checked dict (keys :data:`CONFIG_KEYS`). ``cfg``: a dict, a path to a YAML file, a shipped preset's name
    (``"af3"``: that preset, nothing else), or ``None`` (the engine preset). A mapping with an ``init:`` block (a whole model config) is
    read for that block alone."""
    raw: Any = cfg
    if raw is None:
        raw = {}
    elif isinstance(raw, str) and raw in preset_names():
        raw = {"preset": raw}
    elif isinstance(raw, (str, Path)):
        raw = _read_yaml(Path(raw).expanduser()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"init config must be a mapping, got {type(raw).__name__}")
    if isinstance(raw.get("init"), dict):
        raw = raw["init"]
    unknown = sorted(set(raw) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"init config: unknown key(s) {unknown}; the block takes {list(CONFIG_KEYS)}")
    out = {
        "preset": raw.get("preset") or "engine",
        "rng": raw.get("rng"),
        "seed": raw.get("seed"),
        "reinit": raw.get("reinit") or "changed",
        "overrides": raw.get("overrides") or {},
    }
    if out["rng"] is not None and out["rng"] not in RNGS:
        raise ValueError(f"init config: rng {out['rng']!r} is not one of {list(RNGS)}")
    if out["reinit"] not in REINITS:
        raise ValueError(f"init config: reinit {out['reinit']!r} is not one of {list(REINITS)}")
    if out["seed"] is not None and (isinstance(out["seed"], bool) or not isinstance(out["seed"], int)):
        raise ValueError(f"init config: seed must be an integer, got {out['seed']!r}")
    if not isinstance(out["overrides"], dict):
        raise ValueError("init config: overrides must be a mapping")
    return out


# --------------------------------------------------------------------------------------------------------------------------- targets
@dataclass
class _Target:
    path: str  # the leaf's instance path ("" for the root)
    module: nn.Module
    kind: str  # "linear" | "norm"
    order: tuple[float, int]  # construction order (Linear._init_order; norms draw nothing), then module order
    code: Spec  # the code default (init, bias spec or None without a bias)
    ancestors: list[tuple[str, list[type]]]  # (attribute path from the ancestor, the ancestor's classes along its MRO), root first


def _classes(module: nn.Module) -> list[type]:
    """The classes a pattern may name ``module`` by: its MRO's module classes, minus PyTorch's own and the backend / master mixins."""
    return [
        c
        for c in type(module).__mro__
        if isinstance(c, type) and issubclass(c, nn.Module) and c not in _NOT_NAMES and not c.__module__.startswith("torch.")
    ]


def _kind(module: nn.Module) -> str | None:
    if isinstance(module, Linear):
        return "linear"
    if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) and isinstance(getattr(module, "weight", None), torch.Tensor):
        return "norm"
    return None


def _targets(model: nn.Module) -> list[_Target]:
    modules = dict(model.named_modules())
    classes = {path: _classes(m) for path, m in modules.items()}
    out = []
    for index, (path, m) in enumerate(modules.items()):
        kind = _kind(m)
        if kind is None:
            continue
        parts = path.split(".") if path else []
        ancestors = [(".".join(parts[i:]), classes[".".join(parts[:i])]) for i in range(len(parts) + 1)]
        has_bias = getattr(m, "bias", None) is not None
        if kind == "linear":
            assert isinstance(m, Linear)
            init, bias = m.code_init()
            code = Spec(init, bias)
            order = float(getattr(m, "_init_order", math.inf))
        else:
            code = Spec("one", "zero" if has_bias else None)
            order = math.inf
        out.append(_Target(path, m, kind, (order, index), code, ancestors))
    return out


def _split(key: str) -> tuple[str, str]:
    """``(class pattern, attribute path)`` of a key. The class pattern is a class name (globs allowed) or, to tell apart two classes
    of one name, ``<python module>:<class name>`` (e.g. ``miniworld.modules.bias_only_token_dit:BiasOnlyAttention``)."""
    colon = key.find(":")
    dot = key.find(".", colon + 1 if colon >= 0 else 0)
    return (key, "") if dot < 0 else (key[:dot], key[dot + 1 :])


def _is_pattern(key: str) -> bool:
    head = _split(key)[0]
    return bool(head) and (":" in head or head[0].isupper() or head[0] in "_*?[")


def _matches(cls: type, glob: str) -> bool:
    if ":" in glob:
        return fnmatch.fnmatchcase(f"{cls.__module__}:{cls.__name__}", glob)
    return fnmatch.fnmatchcase(cls.__name__, glob)


def _target_key(t: _Target) -> str:
    """The nearest ``<Class>.<attribute path>`` naming ``t``: the closest named ancestor above the leaf (the leaf's own class at the root)."""
    for rel, classes in reversed(t.ancestors[:-1]):
        if classes:
            return f"{classes[0].__name__}.{rel}"
    classes = t.ancestors[-1][1]
    return classes[0].__name__ if classes else type(t.module).__name__


def _valid_targets_msg(targets: list[_Target], key: str) -> str:
    keys = sorted({_target_key(t) for t in targets})
    close = difflib.get_close_matches(key, keys + [t.path for t in targets], n=5, cutoff=0.6)
    shown = ", ".join(keys[:60]) + (f", ... ({len(keys) - 60} more)" if len(keys) > 60 else "")
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return f"{hint} Valid targets (any instance path of a Linear / LayerNorm also works): {shown}"


# --------------------------------------------------------------------------------------------------------------------------- matching
def _best(t: _Target, index: dict[str, list[tuple[str, str, Spec]]], layer: str, matched: set[str]) -> tuple[str, Spec] | None:
    """The most specific class-pattern entry of one layer for ``t``: (key, spec), or None. Every key that matches (winning or not) is
    added to ``matched``."""
    found: dict[str, tuple[tuple[int, int, int, int], Spec]] = {}
    for rel, classes in t.ancestors:
        for glob, key, spec in index.get(rel, ()):
            for mro, cls in enumerate(classes):
                if _matches(cls, glob):
                    rank = (-(rel.count(".") + 1 if rel else 0), mro, int(any(ch in glob for ch in "*?[")), int(":" not in glob))
                    if key not in found or rank < found[key][0]:
                        found[key] = (rank, spec)
                    break
    matched.update(found)
    if not found:
        return None
    best = min(r for r, _ in found.values())
    winners = {k: s for k, (r, s) in found.items() if r == best}
    if len(set(winners.values())) > 1:
        raise ValueError(
            f"init {layer}: {', '.join(sorted(winners))} match {t.path or '(root)'} equally specifically with different "
            "specs; make one more specific (a longer attribute path, the exact or the module-qualified class name)"
        )
    key = min(winners)
    return key, winners[key]


def _index(entries: dict[str, Spec]) -> dict[str, list[tuple[str, str, Spec]]]:
    out: dict[str, list[tuple[str, str, Spec]]] = {}
    for key, spec in entries.items():
        glob, rel = _split(key)
        out.setdefault(rel, []).append((glob, key, spec))
    return out


@dataclass
class _Resolved:
    target: _Target
    init: str
    bias: str | float | None
    weight_source: str
    bias_source: str
    weight_key: str
    bias_key: str


def _resolve(model: nn.Module, cfg: dict) -> tuple[list[_Resolved], str, dict, list[str]]:
    """Every target's effective init / bias with where it came from, the rng, the config, and the preset entries this model has no
    parameter for (skipped: a preset covers many models). Raises on unknown override keys and invalid specs."""
    targets = _targets(model)
    entries, preset_rng = load_preset(cfg["preset"])
    preset_label = f"preset:{Path(str(cfg['preset'])).stem}"

    by_path = {t.path: t for t in targets}
    paths, patterns = {}, {}
    for key, value in cfg["overrides"].items():
        key = str(key)
        spec = _parse_spec(key, value, "init overrides")
        if key in by_path:
            paths[key] = spec
        elif _is_pattern(key):
            patterns[key] = spec
        else:
            raise ValueError(
                f"init overrides: {key!r} is neither an instance path of a Linear / LayerNorm of this model nor "
                f"<ModuleClass>.<attribute path>.{_valid_targets_msg(targets, key)}"
            )
        if key in by_path and spec.bias is not None and getattr(by_path[key].module, "bias", None) is None:
            raise ValueError(f"init overrides: {key!r} sets a bias, but {key} has none")

    preset_index, override_index = _index(entries), _index(patterns)
    used: set[str] = set()
    preset_used: set[str] = set()
    out = []
    for t in targets:
        init, bias = t.code.init, t.code.bias
        assert init is not None
        wsrc = bsrc = "code"
        wkey = bkey = _target_key(t)
        layers: list[tuple[str, tuple[str, Spec] | None]] = [
            (preset_label, _best(t, preset_index, preset_label, preset_used)),
            ("override", _best(t, override_index, "overrides", used)),
            ("override", (t.path, paths[t.path]) if t.path in paths else None),
        ]
        for label, hit in layers:
            if hit is None:
                continue
            key, spec = hit
            if label == "override":
                used.add(key)  # (instance paths; patterns are added by _best when they match, winning or not)
            _check(t, spec, f"{label} {key!r}")
            if spec.init is not None:
                init, wsrc, wkey = spec.init, label, key
                if t.code.bias is not None and spec.bias is None:
                    bias = implied_bias(init) if t.kind == "linear" else "zero"
                    bsrc, bkey = label, key
            if spec.bias is not None and t.code.bias is not None:
                bias, bsrc, bkey = spec.bias, label, key
        out.append(_Resolved(t, init, bias, wsrc, bsrc, wkey, bkey))

    unused = sorted(set(patterns) - used)
    if unused:
        raise ValueError(
            f"init overrides: {', '.join(map(repr, unused))} match(es) no parameter of this model.{_valid_targets_msg(targets, unused[0])}"
        )
    rng = cfg["rng"] or preset_rng or "legacy"
    return out, rng, cfg, sorted(set(entries) - preset_used)


def _check(t: _Target, spec: Spec, where: str) -> None:
    if t.kind == "norm":
        if spec.init is not None and spec.init not in NORM_INITS:
            raise ValueError(f"init {where}: {t.path or '(root)'} is a {type(t.module).__name__}: init must be one of {list(NORM_INITS)}")
        if spec.bias == "default":
            raise ValueError(
                f"init {where}: {t.path or '(root)'} is a {type(t.module).__name__}: bias 'default' (PyTorch's Linear uniform) "
                "does not apply"
            )
    elif isinstance(t.module, MPLinear) and spec.init in ("zero", "gating"):
        raise ValueError(f"init {where}: {t.path} is an MPLinear: init {spec.init!r} cannot be normalised")


# --------------------------------------------------------------------------------------------------------------------------- drawing
class _Draws:
    """The random source of one :func:`apply_init` call."""

    def __init__(self, rng: str, seed: int | None, devices: list[torch.device]) -> None:
        self.rng, self.seed = rng, seed
        self.np_state = None if (seed is None or rng != "legacy") else np.random.RandomState(seed)
        self._gens: dict[torch.device, torch.Generator] = {}
        self._devices = [d for d in devices if d.type == "cuda"]

    def context(self):
        """legacy + seed: the torch draws (PyTorch's own initialisers) on a forked, seeded global state (restored after)."""
        if self.rng == "legacy" and self.seed is not None:
            stack = contextlib.ExitStack()
            stack.enter_context(torch.random.fork_rng(devices=self._devices))
            torch.manual_seed(self.seed)
            return stack
        return contextlib.nullcontext()

    def gen(self, device: torch.device) -> torch.Generator | None:
        if self.rng != "torch" or self.seed is None:
            return None
        if device not in self._gens:
            self._gens[device] = torch.Generator(device=device).manual_seed(self.seed)
        return self._gens[device]

    def weight(self, linear: Linear, init: str) -> None:
        w = linear.weight
        if init in ("zero", "gating"):
            w.zero_()
        elif init == "one":
            w.fill_(1.0)
        elif self.rng == "legacy":
            if init in ("default", "relu"):
                _trunc_normal_init(w, scale=2.0 if init == "relu" else 1.0, random_state=self.np_state)
            else:
                InitType[init.upper()].apply(w)
        else:
            g = self.gen(w.device)
            tmp = torch.empty(w.shape, dtype=torch.float32, device=w.device)
            if init in ("default", "relu"):
                std = trunc_normal_std(w, scale=2.0 if init == "relu" else 1.0)
                nn.init.trunc_normal_(tmp, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std, generator=g)
            elif init == "normal":
                nn.init.kaiming_normal_(tmp, nonlinearity="linear", generator=g)
            else:
                nn.init.xavier_uniform_(tmp, generator=g)
            w.copy_(tmp)
        linear._after_weight_init()  # as the constructor (MPLinear normalises; Linear does nothing)

    def default_bias(self, linear: Linear) -> None:
        """PyTorch ``nn.Linear``'s bias draw: ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``."""
        b = linear.bias
        assert b is not None
        bound = 1 / math.sqrt(linear.in_features) if linear.in_features > 0 else 0
        g = self.gen(b.device)
        if self.rng == "legacy":
            nn.init.uniform_(b, -bound, bound)
        else:
            tmp = torch.empty(b.shape, dtype=torch.float32, device=b.device)
            nn.init.uniform_(tmp, -bound, bound, generator=g)
            b.copy_(tmp)


@torch.no_grad()
def apply_init(model: nn.Module, cfg: dict | str | Path | None = None) -> nn.Module:
    """Re-initialise ``model``'s managed parameters from an init config (module docstring; ``cfg`` a dict, a YAML path or None) after
    construction. ``reinit: changed`` (default) redraws only parameters whose effective init / bias differs from the code default (the
    engine preset changes nothing); ``reinit: all`` redraws every managed parameter, in construction order: with ``rng: legacy`` and the
    construction's seeds that reproduces construction (the discarded ``nn.Linear`` draws replayed), with ``rng: torch`` the model depends
    on the torch seed alone. Returns ``model``."""
    resolved, rng, conf, _absent = _resolve(model, load_init_config(cfg))
    devices = sorted({p.device for p in model.parameters()}, key=str)
    draws = _Draws(rng, conf["seed"], devices)
    every = conf["reinit"] == "all"
    with draws.context():
        for r in sorted(resolved, key=lambda r: r.target.order):
            m, code = r.target.module, r.target.code
            weight_changed = every or r.init != code.init
            bias_changed = r.bias is not None and (every or r.bias != code.bias)
            if r.target.kind == "norm":
                weight, bias = getattr(m, "weight"), getattr(m, "bias", None)  # noqa: B009  -- nn.Module attributes: typed Tensor
                if weight_changed:
                    weight.fill_(1.0 if r.init == "one" else 0.0)
                if bias_changed and bias is not None and r.bias is not None:
                    fill_bias(bias, r.bias)
                continue
            assert isinstance(m, Linear)
            if every and rng == "legacy":
                nn.Linear.reset_parameters(m)  # the draws nn.Linear's constructor takes (and Linear overwrites)
                draws.weight(m, r.init)
                if r.bias is not None and r.bias != "default":
                    fill_bias(m.bias, r.bias)  # pyright: ignore[reportArgumentType]
                continue
            if weight_changed:
                draws.weight(m, r.init)
            if bias_changed:
                if r.bias == "default":
                    draws.default_bias(m)
                else:
                    fill_bias(m.bias, r.bias)  # pyright: ignore[reportArgumentType]
    return model


@overload
def describe_init(model: nn.Module, cfg: dict | str | Path | None = ..., *, rows: Literal[False] = ...) -> str: ...
@overload
def describe_init(model: nn.Module, cfg: dict | str | Path | None = ..., *, rows: Literal[True]) -> list[dict]: ...
def describe_init(model: nn.Module, cfg: dict | str | Path | None = None, *, rows: bool = False) -> str | list[dict]:
    """Every parameter of ``model`` with its init: ``parameter``, ``module`` (the owning module's class), ``target`` (the entry key that set
    it, else the nearest ``<Class>.<attribute path>``), ``code`` (the code default), ``effective`` (under ``cfg``) and ``source``
    (``code`` / ``preset:<name>`` / ``override`` / ``unmanaged``), one row per ``model.named_parameters()`` entry, then one row per preset
    entry this model has no parameter for (``parameter`` None, ``source`` ``not present``: skipped). A table string, or the row dicts
    with ``rows=True``."""
    resolved, rng, conf, absent = _resolve(model, load_init_config(cfg))
    by_param: dict[str, dict] = {}
    for r in resolved:
        prefix = f"{r.target.path}." if r.target.path else ""
        cls = type(r.target.module).__name__
        by_param[prefix + "weight"] = {
            "module": cls,
            "target": r.weight_key,
            "code": str(r.target.code.init),
            "effective": r.init,
            "source": r.weight_source,
        }
        if r.bias is not None:
            by_param[prefix + "bias"] = {
                "module": cls,
                "target": r.bias_key,
                "code": f"bias {r.target.code.bias}",
                "effective": f"bias {r.bias}",
                "source": r.bias_source,
            }
    modules = dict(model.named_modules())
    out = []
    for name, _ in model.named_parameters():
        row = by_param.get(name)
        if row is None:
            owner = name.rpartition(".")[0]
            row = {"module": type(modules[owner]).__name__, "target": "-", "code": "-", "effective": "-", "source": "unmanaged"}
        out.append({"parameter": name, **row})
    out += [{"parameter": None, "module": "-", "target": key, "code": "-", "effective": "-", "source": "not present"} for key in absent]
    if rows:
        return out
    cols = ("parameter", "module", "target", "code", "effective", "source")
    width = {c: max([len(c)] + [len(str(r[c])) for r in out]) for c in cols}
    head = f"# init: preset {conf['preset']}, rng {rng}, reinit {conf['reinit']}, seed {conf['seed']}, {len(conf['overrides'])} override(s)"
    lines = [head, "  ".join(c.ljust(width[c]) for c in cols)]
    lines += ["  ".join(str(r[c]).ljust(width[c]) for c in cols) for r in out]
    return "\n".join(lines)
