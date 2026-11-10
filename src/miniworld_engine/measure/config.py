"""The measurement config and the cases it selects: requirements rows -> cases -> a built module and its inputs. Module-agnostic.

A config (YAML or TOML, by extension) names the modules and the backends; every other key is optional::

    modules: [LayerNorm, Transition]     # class names of miniworld_engine.modules, or "all"
    backends: [pytorch, anthropic, cueq] # the backends to measure (a class without one is skipped); "pytorch" is always the reference
    external: [anthropic, cueq]          # baselines (their accuracy light is informational)
    rows: []                             # requirements_module.csv data-row numbers (1 = the first row under the header); [] = every row
    dtypes: []                           # filter: bfloat16 | float32 | master ([] = the row's)
    modes: []                            # filter: eval | train ([] = the row's)
    lengths: min                         # min | all | a list (intersected with the row's)
    depths: min                          # the same for the MSA depth
    samples: all                         # min | all (the row's eval_samples / train_samples)
    device: cuda
    seed: 0
    fixed: {bfloat16: 8.0e-3, master: 8.0e-3, float32: 1.0e-5}   # the accuracy's fixed standard per dtype
    speed: {}                            # read by measure/speed.py only (its keys timing / warmup / iterations / flush_l2 /
                                         # compile_mode may also stand at the top level)

A case is one (row, constructor variant, length, depth, samples, dtype, mode). Constructor kwargs are the row's ``dims`` + ``options``
(``a=1;b=x``; values int / float / bool / None / str; ``a=False|True`` expands into one variant per value). The class is built directly
from its name (a model column needs no selector); a class whose ``__init__`` takes ``config`` gets the kwargs that are fields of that
config's type. The weights are random (seeded): every floating parameter redrawn, so no zero-initialised projection hides the math.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import functools
import inspect
import itertools
import json
import math
import platform
import re
import subprocess
import typing
from pathlib import Path

import torch

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements" / "requirements_module.csv"
# Input activation dtype per case dtype. master (bf16-mixed training): fp32 parameters, bf16 activations (what a module inside a
# bf16-autocast model receives from the previous module), the forward under bf16 autocast.
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "master": torch.bfloat16}
FIXED = {"bfloat16": 8e-3, "master": 8e-3, "float32": 1e-5}
EXTERNAL = ("anthropic", "cueq")
#: Backends measured with TF32 products allowed in float32 PyTorch matmuls: every backend (user decisions 2026-11-10: first the
#: reference and the Triton rung, then "TF32 on for all, always").
TF32_BACKENDS = ("pytorch", "triton", "cuda", *EXTERNAL)
#: Human-managed version of the TF32 rule each backend is measured under (backend -> version): bump a backend's entry when its TF32
#: treatment changes. Only float32 cases depend on it (bf16 / master matmuls never use TF32), so the judge owes exactly those.
#: 1: TF32 for pytorch only (measurers v1); 2: TF32 for every backend (2026-11-10).
TF32_RULE = {"pytorch": "1", "triton": "2", "cuda": "2", "anthropic": "2", "cueq": "2"}
#: Keys read by measure/speed.py only, allowed at the top level (collected into ``Config.speed``).
SPEED_KEYS = ("timing", "warmup", "iterations", "flush_l2", "compile_mode")


@dataclasses.dataclass(frozen=True)
class Config:
    modules: tuple[str, ...] = ("all",)
    backends: tuple[str, ...] = ("pytorch",)
    external: tuple[str, ...] = EXTERNAL
    rows: tuple[int, ...] = ()
    dtypes: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    lengths: str | tuple[int, ...] = "min"
    depths: str | tuple[int, ...] = "min"
    samples: str = "all"
    device: str = "cuda"
    seed: int = 0
    fixed: dict = dataclasses.field(default_factory=lambda: dict(FIXED))
    speed: dict = dataclasses.field(default_factory=dict)
    requirements: str = str(REQUIREMENTS)


@dataclasses.dataclass(frozen=True)
class Case:
    row: int                 # requirements data-row number
    module: str              # class name
    model: str
    stream: str
    kwargs: dict             # constructor kwargs (one variant)
    length: int
    depth: int | None
    samples: int
    dtype: str               # bfloat16 | float32 | master
    mode: str                # eval | train
    seed: int

    @property
    def cls(self) -> type:
        return module_class(self.module)

    @property
    def label(self) -> str:
        kw = ";".join(f"{k}={v}" for k, v in self.kwargs.items())
        depth = f" D{self.depth}" if self.depth else ""
        return f"r{self.row} {self.module}({kw}) L{self.length}{depth} S{self.samples} {self.dtype} {self.mode}"


def module_class(name: str) -> type:
    """The :class:`BackendModule` subclass called ``name`` anywhere in ``miniworld_engine.modules`` (its subpackages included)."""
    import miniworld_engine.modules.mpnn  # noqa: F401 -- registers the subpackage's classes
    from miniworld_engine.modules.primitives import BackendModule

    todo, found = [BackendModule], {}
    while todo:
        for sub in todo.pop().__subclasses__():
            found.setdefault(sub.__name__, sub)
            todo.append(sub)
    if name not in found:
        raise ValueError(f"no BackendModule class {name!r} in miniworld_engine.modules")
    return found[name]


# ------------------------------------------------------------------------------------------------------------------------------ load
def _tuple(value) -> tuple:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _sweep(value) -> str | tuple[int, ...]:
    return value if isinstance(value, str) else tuple(int(v) for v in _tuple(value))


def from_dict(raw: dict) -> Config:
    known = {f.name for f in dataclasses.fields(Config)}
    kw = {k: v for k, v in raw.items() if k not in SPEED_KEYS}
    unknown = sorted(set(kw) - known)
    if unknown:
        raise ValueError(f"unknown config keys {unknown}; known: {sorted(known | set(SPEED_KEYS))}")
    kw["speed"] = {**(kw.get("speed") or {}), **{k: v for k, v in raw.items() if k in SPEED_KEYS}}
    for key in ("modules", "backends", "external", "dtypes", "modes"):
        if key in kw:
            kw[key] = tuple(str(v) for v in _tuple(kw[key]))
    if "rows" in kw:
        kw["rows"] = tuple(int(v) for v in _tuple(kw["rows"]))
    for key in ("lengths", "depths"):
        if key in kw:
            kw[key] = _sweep(kw[key])
    if "fixed" in kw:
        kw["fixed"] = {**FIXED, **{str(k): float(v) for k, v in kw["fixed"].items()}}
    cfg = Config(**kw)
    if "pytorch" not in cfg.backends:
        cfg = dataclasses.replace(cfg, backends=("pytorch", *cfg.backends))
    return cfg


def load(path: str | Path) -> Config:
    """A config file (``.yaml`` / ``.yml`` / ``.toml``) -> :class:`Config`."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".toml":
        import tomllib

        raw = tomllib.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text) or {}
    return from_dict(raw)


def replace(cfg: Config, **changes) -> Config:
    """``cfg`` with some keys replaced (the same parsing as the file)."""
    return from_dict({**dataclasses.asdict(cfg), **{k: v for k, v in changes.items() if v is not None}})


def with_speed(cfg: Config, **speed) -> Config:
    return dataclasses.replace(cfg, speed={**cfg.speed, **speed})


def speed_settings(cfg: Config) -> dict:
    return dict(cfg.speed)


# ----------------------------------------------------------------------------------------------------------------------------- cases
def _value(raw: str):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:  # noqa: PERF203 -- the next cast is tried
            pass
    return {"True": True, "False": False, "None": None}.get(raw, raw)


def _variants(dims: str, options: str) -> list[dict]:
    """``a=1;b=x|y`` (+ options) -> one kwargs dict per combination of the ``|`` alternatives."""
    pairs = [part.partition("=") for part in f"{dims};{options}".split(";") if part.strip()]
    keys = [k.strip() for k, _, _ in pairs]
    choices = [[_value(v.strip()) for v in raw.split("|")] for _, _, raw in pairs]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*choices)]


def _ints(raw: str) -> list[int]:
    return [int(v) for v in raw.split("|") if v.strip()]


def _pick(values: list[int], how) -> list:
    if not values:
        return [None]
    if how == "min":
        return [min(values)]
    if how == "all":
        return sorted(values)
    if isinstance(how, tuple):
        return [v for v in sorted(values) if v in how]
    raise ValueError(f"a sweep is 'min', 'all' or a list of ints, got {how!r}")


def rows(cfg: Config) -> list[tuple[int, dict]]:
    """The (data-row number, row) pairs the config selects."""
    with open(cfg.requirements, newline="") as fh:
        table = list(csv.DictReader(fh))
    every = "all" in cfg.modules
    return [(n, r) for n, r in enumerate(table, 1)
            if (every or r["module"] in cfg.modules) and (not cfg.rows or n in cfg.rows)]


def cases(cfg: Config):
    """Every case of the config, in requirements order."""
    for n, r in rows(cfg):
        for kwargs in _variants(r["dims"], r["options"]):
            for length in _pick(_ints(r["lengths"]), cfg.lengths):
                for depth in _pick(_ints(r["depths"]), cfg.depths):
                    for dtype in r["dtypes"].split("|"):
                        if cfg.dtypes and dtype not in cfg.dtypes:
                            continue
                        for mode in r["modes"].split("|"):
                            if cfg.modes and mode not in cfg.modes:
                                continue
                            for samples in _pick(_ints(r[f"{mode}_samples"]) or [1], "min" if cfg.samples == "min" else "all"):
                                yield Case(row=n, module=r["module"], model=r["model"], stream=r["stream"], kwargs=kwargs,
                                           length=length, depth=depth, samples=samples, dtype=dtype, mode=mode, seed=cfg.seed)


def backends_of(cfg: Config, case: Case) -> list[str]:
    """The config's backends this case's class has, the reference first."""
    have = case.cls.backends
    return ["pytorch", *(b for b in cfg.backends if b != "pytorch" and b in have)]


def describe(case: Case, cfg: Config | None = None) -> dict:
    """The case in a record: what was measured (row number, class, variant, length, ...) and the versions of what measured it: the
    module class's ``version``, the ``version`` of every kernel requirements_kernel.csv lists for the class in this mode and the
    ``version`` of those kernels' families' launch-config tables."""
    return {"label": case.label, "row": case.row, "module": case.module, "model": case.model, "stream": case.stream,
            "kwargs": case.kwargs, "length": case.length, "depth": case.depth, "samples": case.samples, "dtype": case.dtype,
            "mode": case.mode, "gpu": result_gpu(), "versions": versions(case.module, case.mode)}


# ------------------------------------------------------------------------------------------------------------------- record identity
#: Human-managed version of this file's meaning (cases, module building, inputs): bump it when a change can alter what a record means
#: (not for refactors); the judge compares the recorded version with the current one and asks before keeping old results.
CONFIG_VERSION = "1"
KERNELS = Path(__file__).resolve().parents[1] / "requirements" / "requirements_kernel.csv"
#: Which modes a kernel's role serves.
ROLE_MODES = {"forward": ("eval", "train"), "forward_infer": ("eval",), "forward_train": ("train",), "backward": ("train",)}


@functools.lru_cache(maxsize=4)
def _table(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def versions(module: str, mode: str, gpu: str | None = None) -> dict:
    """Human-managed versions a result depends on: ``module`` (the class attribute ``version``: bump it when the class's math, its
    reference or a backend adapter changes a result) and ``kernels`` (name -> the ``version`` column of requirements_kernel.csv for
    every kernel listed for the class whose role serves ``mode`` and, for a card-specific kernel (name suffix ``_smNN``), only on that card: bump a kernel's when its algorithm changes a result). ``tables``: the launch-config table versions of
    those kernels' families for ``gpu`` (default: the card this process measures on, :func:`result_gpu`)."""
    cls = module_class(module)
    rows = [r for r in _table(str(KERNELS))
            if module in (r.get("modules") or "").split("|") and mode in ROLE_MODES.get(r.get("role", ""), ())]
    gpu = gpu or result_gpu()
    # a kernel built for one card (name ending "_sm80", "_sm90", "_sm90a", ...) counts only for that card's results
    card_sm = re.search(r"sm(\d+)", gpu)
    rows = [r for r in rows if (m := re.search(r"_sm(\d+)a?$", r["name"])) is None or (card_sm and m.group(1) == card_sm.group(1))]
    kernels = {r["name"]: r.get("version") or "1" for r in rows}
    tables = {f: table_version(f, gpu) for f in sorted({r["family"] for r in rows})}
    adapters = {b: str(v) for b, v in sorted(getattr(cls, "backend_versions", {}).items())}
    return {"module": str(getattr(cls, "version", "1")), "kernels": dict(sorted(kernels.items())), "tables": tables,
            "backends": adapters, "tf32": dict(TF32_RULE)}


#: The card assumed when no CUDA device is visible (a CPU-side judge run without --gpu).
DEFAULT_GPU = "h100_sm90"


def result_gpu() -> str:
    """The card key results are for (``kernels._launch_config.gpu_key``, e.g. ``h100_sm90``, ``a100_sm80``): the current CUDA device, else
    :data:`DEFAULT_GPU`. Tokens, launch-config tables and their versions are per card."""
    from miniworld_engine.kernels import _launch_config

    key = _launch_config.gpu_key()
    return DEFAULT_GPU if key == "cpu" else key


@functools.cache
def table_version(family: str, gpu: str = DEFAULT_GPU) -> str:
    """The ``version`` of a kernel family's launch-config table for ``gpu`` (``kernels/<family>/configs/<gpu>.json``,
    incremented by the pretuner's merge when the table's content changes); "1" when the table has none or does not exist."""
    path = Path(__file__).resolve().parents[1] / "kernels" / family / "configs" / f"{gpu}.json"
    try:
        return str(json.loads(path.read_text()).get("version", 1))
    except (OSError, ValueError):
        return "1"


@functools.lru_cache(maxsize=1)
def environment() -> dict:
    """Where the record was made: host, toolchain versions and the card (the numerics of the reference and every speed depend on them)."""
    env = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import triton

        env["triton"] = triton.__version__
    except Exception:  # recorded as absent
        env["triton"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(gpu=props.name, capability=list(torch.cuda.get_device_capability(0)), sms=props.multi_processor_count)
        try:
            env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                                           text=True, timeout=10).stdout.splitlines()[0].strip()
        except Exception:  # recorded as absent
            env["driver"] = None
    return env


def settings(cfg: Config) -> dict:
    """What the config says about HOW one case is measured, as plain values: the seed, the fixed standard and the speed settings (the
    keys that only SELECT cases are left out, so a run of a subset measures each case exactly as the full run does)."""
    return {"seed": cfg.seed, "fixed": dict(sorted(cfg.fixed.items())), "speed": dict(sorted(speed_settings(cfg).items()))}


def run_meta(cfg: Config, kind: str, measurer_version: str) -> dict:
    """The part of a record that is the same for the whole run: the versions of this file, of the measurer (``kind``: accuracy |
    speed) and of the prebuilder (how the kernels were built), the measurement settings and the environment."""
    from miniworld_engine.kernels._prebuild import PREBUILD_VERSION

    return {"kind": kind, "config_version": CONFIG_VERSION, "measurer_version": measurer_version, "prebuild_version": PREBUILD_VERSION,
            "settings": settings(cfg), "env": environment()}


def seed(case: Case) -> int:
    return case.seed


def autocast(case: Case) -> torch.dtype | None:
    """The autocast dtype of the case's forward (``master``: bfloat16), else ``None``."""
    return torch.bfloat16 if case.dtype == "master" else None


# ----------------------------------------------------------------------------------------------------------------------------- build
def _config_type(cls: type) -> type | None:
    """The type of ``cls.__init__``'s ``config`` parameter (a dataclass / pydantic model), or ``None`` if it takes none."""
    if "config" not in inspect.signature(cls.__init__).parameters:
        return None
    hint = typing.get_type_hints(cls.__init__, localns=dict(vars(cls))).get("config")
    for t in typing.get_args(hint) or (hint,):
        if isinstance(t, type) and t is not type(None):
            return t
    return None


def construct(cls: type, kwargs: dict):
    """``cls(**kwargs)``, or ``cls(ConfigType(**fields), **rest)`` for a class built from a config object."""
    config_type = _config_type(cls)
    if config_type is None:
        return cls(**kwargs)
    fields = set(inspect.signature(config_type).parameters)
    return cls(config_type(**{k: v for k, v in kwargs.items() if k in fields}), **{k: v for k, v in kwargs.items() if k not in fields})


@torch.no_grad()
def randomize(module: torch.nn.Module, generator: torch.Generator) -> None:
    """Redraw every floating parameter: matrices ``N(0, 1 / fan_in)``, 1-D weights (norm gains) ``1 + 0.1 N``, other vectors ``0.2 N``."""
    for name, p in module.named_parameters():
        if not p.is_floating_point():
            continue
        z = torch.randn(p.shape, device=p.device, generator=generator)
        if p.ndim >= 2:
            p.copy_(z / math.sqrt(p.shape[-1]))
        elif name.endswith("weight"):
            p.copy_(1 + 0.1 * z)
        else:
            p.copy_(0.2 * z)


def cast(module: torch.nn.Module, dtype: str) -> torch.nn.Module:
    """The module at a case dtype: ``bfloat16`` cast down (fp32-pinned parameters stay fp32), ``float32`` / ``master`` fp32, ``master``
    also pins the fp32 masters of every ``MasterWeights`` inside."""
    from miniworld_engine.modules.primitives import MasterWeights

    module = module.to(torch.bfloat16) if dtype == "bfloat16" else module.float()
    if dtype == "master":
        for sub in module.modules():
            if isinstance(sub, MasterWeights):
                sub.set_master_weights()
    return module


def generator(case: Case, device) -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(case.seed)


def build(case: Case, *, device) -> torch.nn.Module:
    """The case's module: constructed, moved to ``device``, random weights (seeded), cast to the case dtype, in the case mode."""
    torch.manual_seed(case.seed)
    module = construct(case.cls, case.kwargs).to(device).float()
    randomize(module, generator(case, device))
    return cast(module, case.dtype).train(case.mode == "train")


def inputs(case: Case, module: torch.nn.Module, *, device) -> dict:
    """``module.make_inputs`` for the case (a generator seeded per case: the same values for every backend and dtype)."""
    return module.make_inputs(length=case.length, samples=case.samples, depth=case.depth, dtype=DTYPES[case.dtype],
                              device=torch.device(device), generator=generator(case, device), stream=case.stream)


def clone(module: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(module)
