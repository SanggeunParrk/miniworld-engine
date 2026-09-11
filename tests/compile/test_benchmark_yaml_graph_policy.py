"""Actual module YAML defaults must reach the requested graph policy."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, model_validator

from miniworld_engine.modules.exceptions import ImplementationType

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "benchmarks/modules").glob("*/configs/bench.yaml"))


@pytest.fixture(scope="module")
def config_class():
    # Importing the full runner requires CUDA. Execute its exact config class on CPU.
    source = ROOT / "benchmarks/runners/bench.py"
    tree = ast.parse(source.read_text())
    nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {"BenchConfig", "is_inference_mode"}]
    namespace: dict[str, Any] = {"BaseModel": BaseModel, "model_validator": model_validator,
                                 "Literal": Literal, "ImplementationType": ImplementationType,
                                 "DictConfig": DictConfig}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    namespace["BenchConfig"].model_rebuild(_types_namespace=namespace)
    return namespace["BenchConfig"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.parent.parent.name)
@pytest.mark.parametrize(("mode", "expected"), [("inference", "manual"), ("training", "disabled")])
def test_yaml_defaults_resolve_graph_by_mode(config_class, path, mode, expected):
    data = yaml.safe_load(path.read_text())
    assert data["cudagraph"] == "auto"
    config = config_class.model_validate({**data, "mode": mode})
    assert config.cudagraph == expected
    # Explicit requests remain supported; defaults must not override a requested regime.
    explicit = config_class.model_validate({**data, "mode": mode, "cudagraph": "disabled"})
    assert explicit.cudagraph == "disabled"


def test_native_default_comparisons_include_only_equivalent_paths():
    assert len(CONFIGS) == 12
    configs = {path.parent.parent.name: yaml.safe_load(path.read_text()) for path in CONFIGS}
    for name in ("triangle_multiplication", "triangle_attention", "triangle_multiplication_bidirectional"):
        assert {"pytorch", "cuequivariance", "miniworld"} <= set(configs[name]["implementations"])
    assert {"pytorch", "miniworld"} <= set(configs["swa_atom_attention"]["implementations"])


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.parent.parent.name)
@pytest.mark.parametrize(("mode", "expected"), [("inference", 5), ("training", 48)])
def test_yaml_augmentation_resolves_actual_hydra_config(config_class, path, mode, expected):
    data = yaml.safe_load(path.read_text())
    assert data["n_augment"] == "auto"
    # Native entry receives DictConfig, while programmatic callers pass a dict.
    for make_config in (dict, OmegaConf.create):
        config = config_class.model_validate(make_config({**data, "mode": mode}))
        assert config.n_augment == expected
        assert config.model_dump()["n_augment"] == expected
        explicit = config_class.model_validate(make_config({**data, "mode": mode, "n_augment": 7}))
        assert explicit.n_augment == 7


@pytest.mark.parametrize(("mode", "expected"), [("inference", 5), ("training", 48)])
def test_programmatic_module_default_and_positive_count(config_class, mode, expected):
    data = {"level": "module", "target": "adaptive_layernorm", "mode": mode, "metric": "time"}
    assert config_class.model_validate(data).n_augment == expected
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            config_class.model_validate({**data, "n_augment": invalid})
    assert config_class.model_validate({**data, "level": "kernel"}).n_augment == 32
