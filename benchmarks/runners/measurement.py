"""Execution evidence for benchmark results; requested settings are never evidence."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch


class UnsupportedBenchmark(NotImplementedError):
    """A requested execution regime cannot be measured faithfully by this target."""


@dataclass
class ExecutionEvidence:
    graphs: set[tuple[int, int]] = field(default_factory=set)
    scopes: set[str] = field(default_factory=set)

    @property
    def compiled(self) -> bool:
        return bool(self.graphs)


_ACTIVE: ContextVar[ExecutionEvidence | None] = ContextVar("benchmark_execution", default=None)


@contextmanager
def observe_execution():
    evidence = ExecutionEvidence()
    token = _ACTIVE.set(evidence)
    try:
        yield evidence
    finally:
        _ACTIVE.reset(token)


class CompileProbe:
    """Record an Inductor graph only when its returned executable is actually called.

    A compile request, graph tracing, or a reference-only call does not establish
    that the measured callable used compiled code. Non-fullgraph calls explicitly
    describe their scope as partial-allowed; they do not claim whole-step fusion.
    """

    def __init__(self, scope: str, *, fullgraph: bool, backend=None):
        self.scope = scope + (":fullgraph" if fullgraph else ":partial_allowed")
        self.graphs_created = 0
        self._backend = backend

    def __call__(self, graph_module, example_inputs, **kwargs):
        if self._backend is None:
            from torch._dynamo.backends.registry import lookup_backend
            self._backend = lookup_backend("inductor")
        # A callable backend receives options directly; the string Inductor backend
        # normally translates these to compile_fx(config_patches=...). Preserve that contract.
        options = kwargs.pop("options", None)
        if options is not None:
            kwargs["config_patches"] = options
        executable = self._backend(graph_module, example_inputs, **kwargs)
        self.graphs_created += 1
        graph_id = (id(self), self.graphs_created)

        def run(*args, **call_kwargs):
            evidence = _ACTIVE.get()
            if evidence is not None:
                evidence.graphs.add(graph_id)
                evidence.scopes.add(self.scope)
            return executable(*args, **call_kwargs)

        # Do not copy __wrapped__ or _torchdynamo_orig_callable: Dynamo unwraps those
        # and would bypass the execution witness for real Inductor executables.
        if getattr(executable, "_boxed_call", False):
            run.__dict__["_boxed_call"] = True
        return run


def compile_for_benchmark(func: Callable, *, fullgraph: bool = False):
    probe = CompileProbe("callable", fullgraph=fullgraph)
    compiled = torch.compile(func, backend=probe, dynamic=False, fullgraph=fullgraph,
                             options={"triton.cudagraphs": False})
    compiled.__dict__["_benchmark_compile_probe"] = probe
    return compiled


def compile_module_for_benchmark(model: torch.nn.Module) -> None:
    probe = CompileProbe("module_forward", fullgraph=False)
    model.compile(backend=probe, dynamic=False, fullgraph=False,
                  options={"triton.cudagraphs": False})


def require_compile_evidence(requested: bool, evidence: ExecutionEvidence) -> None:
    if requested and not evidence.compiled:
        raise UnsupportedBenchmark(
            "compile=True requested, but the measured callable executed no compiled graph; "
            "choose compile=False explicitly or provide a compilable implementation")
    if not requested and evidence.compiled:
        raise RuntimeError("compile=False requested, but the measured callable executed compiled code")


def parameter_dtype_of(model: torch.nn.Module) -> str:
    return "+".join(sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters()}))


def input_shapes_of(func: Callable) -> str:
    """Describe captured tensor operands without guessing batch/token axes from CLI flags."""
    import inspect
    import json

    found = {}
    seen = set()

    def visit(value, name, depth):
        if isinstance(value, torch.Tensor):
            found[name] = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
            return
        if depth > 5 or id(value) in seen:
            return
        seen.add(id(value))
        original = getattr(value, "_torchdynamo_orig_callable", None)
        if original is not None:
            visit(original, name, depth + 1)
        elif inspect.isfunction(value):
            for key, item in inspect.getclosurevars(value).nonlocals.items():
                visit(item, f"{name}.{key}" if name else key, depth + 1)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{name}[{index}]", depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{name}[{key}]", depth + 1)

    visit(func, "", 0)
    return json.dumps(found, sort_keys=True)


def snapshot_outputs(value):
    """Retain independent tensor values for a check outside the timed region."""
    return {name: tensor.detach().clone() for name, tensor in _output_tensors(value).items()}


def _output_tensors(value, prefix="output"):
    result = {}
    if isinstance(value, torch.Tensor):
        result[prefix] = value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_output_tensors(item, f"{prefix}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(_output_tensors(item, f"{prefix}[{key}]"))
    return result


def check_execution_outputs(actual, expected):
    """Check timed execution against its eager/capture preparation, not a model accuracy gate.

    BF16 eager primitives may round between operations that Inductor fuses. The
    reported errors and tolerances describe this execution-consistency check;
    target-specific reference accuracy remains a separate result.
    """
    tensors = _output_tensors(actual)
    if tensors.keys() != expected.keys():
        raise RuntimeError("timed execution changed the output tensor structure")
    diagnostics = {}
    for name, tensor in tensors.items():
        reference = expected[name]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise RuntimeError(f"timed execution changed shape/dtype for {name}")
        if not bool(torch.isfinite(tensor).all()) or not bool(torch.isfinite(reference).all()):
            raise RuntimeError(f"timed execution produced non-finite values for {name}")
        if tensor.is_floating_point():
            lhs, rhs = tensor.float(), reference.float()
            relative = float((lhs - rhs).norm() / rhs.norm().clamp_min(1e-12))
            limit = 0.02 if tensor.dtype in (torch.bfloat16, torch.float16) else 1e-4
            diagnostics[name] = {"relative_frobenius": relative, "limit": limit}
            if relative > limit:
                raise RuntimeError(f"timed execution differs for {name}: relative={relative:g} > {limit:g}")
        elif not torch.equal(tensor, reference):
            raise RuntimeError(f"timed execution changed non-floating output {name}")
    return diagnostics


def check_finite_outputs(value) -> int:
    tensors = _output_tensors(value)
    if not tensors:
        raise RuntimeError("measurement produced no observable output or gradient")
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"measurement produced non-finite values for {name}")
    return len(tensors)


def make_run_provenance(config: dict, source_hash: str) -> dict:
    import hashlib
    import json
    import uuid

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return {"run_id": uuid.uuid4().hex, "config_hash": hashlib.sha256(canonical.encode()).hexdigest(),
            "source_hash": source_hash, "config": config}


def benchmark_source_hash() -> str:
    import hashlib
    from pathlib import Path

    from miniworld_engine.autotune.plan import source_identity

    digest = hashlib.sha256(source_identity().encode())
    for name in ("bench.py", "measurement.py", "bench_policy.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_source_identity(expected: str) -> None:
    if benchmark_source_hash() != expected:
        raise RuntimeError("benchmark or engine sources changed during this run; rerun from stable sources")
