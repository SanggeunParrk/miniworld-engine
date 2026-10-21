"""The engine's own coverage driver for ``dev bundle-kernels``: run every production module once per shape class.

Every compiled artifact a run can request is selected by a module's widths, its mode (inference / training), its dtype,
and for a few families by a length class (TriMul cells by L, the token pair-init ``TPI_RI`` bucket, SWA atom buckets) or
the number of batched samples. This driver runs, on the current GPU:

* every case of ``autotune.builder.cases()`` -- the modules and checkpoint leaves of ``registry_module.csv``, each row
  with its own widths -- at the stream's production lengths: MiniWorld's inference padding ladder (token rungs
  128 ... 2560, atom rungs 1024 ... 49152) for inference, the training crops (384, 768 tokens; the registry's atom
  ladder) for training, in each declared dtype and compute dtype, plus two batched samples on the pair stream;
* the architecture blocks the model is built from (Pairformer with one- and two-directional TriMul, the token
  DiffusionTransformer with its pair-bias / core-precision options and batched pairs, the bias-only token DiT (one
  pair shared by the samples or one per sample, masked, hoisted, d_cond 384 / 768), the SWA atom transformer in its
  three styles, the block-local atom DiT and the template-stack TriMul (one direction, hidden twice the pair width)),
  inference and training,

every one in bf16, fp32, bf16-mixed (fp32 parameters under bf16 autocast, MiniWorld's precision) and
fp32 parameters fed bf16 activations (``PRECISIONS``).

The ``kernels/prebuilt`` recorder (``MINIWORLD_ENGINE_BUNDLE_RECORD``) captures what was built or loaded; this module
only has to reach it. Cases are JSON descriptors so ``kernel_bundle`` can deal them to worker processes
(``python -m miniworld_engine.build.bundle_coverage serve``: one descriptor per stdin line, one ``RESULT`` line each).
A case that cannot run (an unsupported width, a shape a module refuses) is reported, not fatal: the point is to reach
what production reaches.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback

#: MiniWorld's inference padding ladder (scripts/run_miniworld_diffusion_inference.py) and training crops.
TOKEN_RUNGS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 2560)
ATOM_RUNGS = (1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152)
TRAIN_TOKENS = (384, 768)
BATCHED = 2  # samples per call for the batched pair-stream runs
#: Shapes that want most of a GPU: run one at a time per GPU, after the rest.
HEAVY_TOKENS = 1024
HEAVY_ATOMS = 16384


#: Precisions a call can arrive in: (parameter dtype, activation dtype, bf16 autocast). "mixed" is MiniWorld's bf16-mixed
#: training / inference (fp32 master parameters, autocast), "master" fp32 parameters fed bf16 activations.
PRECISIONS = {
    "bfloat16": ("bfloat16", "bfloat16", False),
    "float32": ("float32", "float32", False),
    "mixed": ("float32", "float32", True),
    "master": ("float32", "bfloat16", False),
}


def _lengths(stream: str, registry: tuple[int, ...], train: bool) -> tuple[int, ...]:
    if stream in ("token_pair", "token_single", "msa_token"):
        return tuple(sorted(set(TRAIN_TOKENS) | set(registry))) if train else TOKEN_RUNGS
    if stream in ("atom_single", "atom_pair"):
        return registry if train else tuple(sorted(set(registry) | set(ATOM_RUNGS)))
    return registry


def _heavy(stream: str, length: int) -> bool:
    if stream in ("token_pair", "msa_token"):
        return length >= HEAVY_TOKENS
    if stream in ("atom_single", "atom_pair"):
        return length >= HEAVY_ATOMS
    return False


def cases() -> list[dict]:
    """Every descriptor, in a stable order (the expensive ones are flagged ``heavy``)."""
    from miniworld_engine.autotune.builder import CASE_NAMES, _shapes

    out: list[dict] = []
    for name in CASE_NAMES:
        shapes = _shapes(name)
        dtypes = [str(d).removeprefix("torch.") for d in shapes["dtypes"]] or ["bfloat16"]
        # "master": fp32 parameters fed bf16 activations (pdall's training, no autocast) selects other kernel builds
        # (single-direction TriMul backward with -DMASTER_FP32=1); 2.3.4 shipped without them.
        precs = [*dtypes, "mixed", *(["master"] if "bfloat16" in dtypes else [])]
        computes = [str(d).removeprefix("torch.") for d in shapes["compute_dtypes"]] or [None]
        modes = [False, True] if shapes["train"] else [False]
        for i, (stream, registry) in enumerate(zip(shapes["streams"], shapes["lengths_by_dim"], strict=True)):
            for train in modes:
                for length in _lengths(stream, tuple(registry), train):
                    for prec in precs:
                        for compute in computes:
                            batches = [None, BATCHED] if stream == "token_pair" and length in (384, 768) else [None]
                            for b in batches:
                                out.append({"kind": "case", "case": name, "dim": i, "L": length, "train": train,  # noqa: PERF401
                                            "dtype": prec, "compute": compute, "batch": b,
                                            "heavy": _heavy(stream, length)})
    block_train = (128, 256, 384, 512, 640, 768)
    for train in (False, True):
        lengths = block_train if train else TOKEN_RUNGS
        for prec in PRECISIONS:
            for length in lengths:
                for bidir in (False, True):
                    for b in ((1, BATCHED) if length in (384, 768) else (1,)):
                        out.append({"kind": "pairformer", "L": length, "train": train, "dtype": prec, "bidir": bidir,  # noqa: PERF401
                                    "batch": b, "heavy": length >= HEAVY_TOKENS})
                for opts in ({}, {"hoist_pair_bias": True}, {"hoist_pair_bias": True, "attention_core_dtype": "bf16_sm90"},
                             {"use_qk_norm": True}):
                    for b in ((1, BATCHED) if length in (384, 768) else (1,)):
                        out.append({"kind": "diffusion", "L": length, "train": train, "dtype": prec, "options": opts,  # noqa: PERF401
                                    "samples": 48 if train else 5, "batch": b, "heavy": length >= HEAVY_TOKENS})
                # The bias-only token DiT: one pair shared by the samples, or a pair of its own per sample (training
                # block_psb); masked or not; trained stacks also from one hoisted pair_bias_all.
                for d_cond in (384, 768):
                    for pair in ("shared", "per_sample"):
                        if pair == "per_sample" and length > TRAIN_TOKENS[-1]:
                            continue
                        for masked, hoist in ((False, False), (True, False), *(((True, True),) if train else ())):
                            out.append({"kind": "bias_only_dit", "L": length, "train": train, "dtype": prec,  # noqa: PERF401
                                        "d_cond": d_cond, "pair": pair, "masked": masked, "hoist": hoist,
                                        "samples": 48 if train else 5,
                                        "heavy": length >= HEAVY_TOKENS or (pair == "per_sample" and length >= 640)})
            atoms = (1024, 2048, 3072, 4096, 6144) if train else ATOM_RUNGS
            for n in atoms:
                for variant in ("esmfold2", "af3", "af3_global"):
                    out.append({"kind": "swa_atom", "N": n, "train": train, "dtype": prec, "variant": variant,  # noqa: PERF401
                                "samples": 48 if train else 5, "heavy": n >= HEAVY_ATOMS})
                out.append({"kind": "local_dit", "N": n, "train": train, "dtype": prec,
                            "samples": 48 if train else 5, "heavy": n >= HEAVY_ATOMS})
    # Template-stack TriMul: one direction, hidden twice the pair width (pair 64 / hidden 128, 128 / 256), B 1 and 4 templates.
    for train in (False, True):
        for prec in PRECISIONS:
            for d in (64, 128):
                for length in (TRAIN_TOKENS if train else TOKEN_RUNGS):
                    for outgoing in (True, False):
                        for b in (1, 4):
                            out.append({"kind": "trimul_hidden2x", "L": length, "train": train, "dtype": prec,  # noqa: PERF401
                                        "d": d, "outgoing": outgoing, "batch": b, "heavy": length >= HEAVY_TOKENS})
    for i, c in enumerate(out):
        c["id"] = i
    return out


# --------------------------------------------------------------------------------------------------- running
def _dt(name: str | None):
    import torch

    return None if name is None else getattr(torch, name)


def _prec(name: str):
    """(parameter dtype, activation dtype, autocast) of a precision name."""
    p, a, autocast = PRECISIONS[name]
    return _dt(p), _dt(a), autocast


def _step(module, args, train: bool, autocast_bf16: bool = False, kwargs: dict | None = None):
    import torch

    module.train(train)
    if train:
        args = tuple(a.detach().clone().requires_grad_(True) if torch.is_tensor(a) and a.is_floating_point() else a
                     for a in args)
    with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_bf16):
        out = module(*args, **(kwargs or {}))
        out = out if torch.is_tensor(out) else out[0]
        if train:
            out.float().sum().backward()
    torch.cuda.synchronize()


def _randomize(module) -> None:
    import torch

    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.05)


_BUILDER_CASES: list = []


def _builder_cases_cached(factory):
    if not _BUILDER_CASES:
        _BUILDER_CASES.extend(factory())
    return _BUILDER_CASES


def _run_case(c: dict) -> None:
    from miniworld_engine.autotune.builder import cases as builder_cases

    case = next(k for k in _builder_cases_cached(builder_cases) if k.name == c["case"])
    dims = case.dims[c["dim"]]
    pdt, adt, autocast = _prec(c["dtype"])
    module = case.factory(dims, 0.25 if c["train"] else 0.0, "miniworld", pdt)
    _randomize(module)
    batch = c["batch"] or case.augmentation_for(c["dim"], c["train"])
    args = case.inputs(batch, c["L"], dims, adt, case.stream_for(c["dim"]))
    kwargs = {"compute_dtype": _dt(c["compute"])} if c["compute"] else {}
    _step(module, args, c["train"], autocast, kwargs)


def _run_block(c: dict) -> None:
    import torch

    from miniworld_engine.architecture.exceptions import ImplementationType

    pdt, dt, autocast = _prec(c["dtype"])
    impl = ImplementationType.MINIWORLD_ENGINE
    if c["kind"] == "pairformer":
        from miniworld_engine.architecture.blocks.pairformer import Pairformer

        model = Pairformer(Pairformer.Config(n_block=1, p_drop=0.25 if c["train"] else 0.0,
                                             use_bidirectional_trimul=c["bidir"], implementation=impl))
        b, n = c["batch"], c["L"]
        args = (torch.randn(b, n, n, 128, device="cuda", dtype=dt), torch.randn(b, n, 384, device="cuda", dtype=dt))
    elif c["kind"] == "diffusion":
        from miniworld_engine.architecture.blocks.diffusion_transformer import (
            DiffusionTransformer,
        )

        model = DiffusionTransformer(DiffusionTransformer.Config(n_block=1, implementation=impl, **c["options"]))
        a, n, b = c["samples"], c["L"], c.get("batch", 1)
        args = (torch.randn(a, b, n, 768, device="cuda", dtype=dt), torch.randn(a, b, n, 384, device="cuda", dtype=dt),
                torch.randn(b, n, n, 128, device="cuda", dtype=dt))
    elif c["kind"] == "swa_atom":
        from miniworld_engine.architecture.blocks.swa_atom_transformer import (
            SWAAtomTransformer,
        )
        from miniworld_engine.modules.swa_atom_attention import build_attention_params

        config = SWAAtomTransformer.Config(n_block=1, block_style="af3" if c["variant"].startswith("af3") else "esmfold2",
                                           global_attn=c["variant"] == "af3_global")
        model = SWAAtomTransformer(config).cuda().to(pdt)
        _randomize(model)
        a, s = c["samples"], c["N"]
        q = torch.randn(a, s, config.d_atom, device="cuda", dtype=dt)
        cond = torch.randn(a, s, config.d_cond, device="cuda", dtype=dt)
        cos, sin = model.build_rope(torch.randn(1, s, 3, device="cuda"), torch.arange(s, device="cuda")[None])
        params = build_attention_params(cos, sin, torch.ones(a, s, dtype=torch.bool, device="cuda"), a)
        model.train(c["train"])
        if c["train"]:
            q = q.requires_grad_(True)
        with torch.set_grad_enabled(c["train"]), torch.autocast("cuda", dtype=torch.bfloat16,
                                                                 enabled=autocast or dt is torch.bfloat16):
            out = model(q, cond, params)
            if c["train"]:
                out.float().sum().backward()
        torch.cuda.synchronize()
        return
    elif c["kind"] == "trimul_hidden2x":
        from miniworld_engine.modules.exceptions import ImplementationType as MI
        from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication

        d, b, n = c["d"], c["batch"], c["L"]
        model = TriangleMultiplication(d, d_hidden=2 * d, outgoing=c["outgoing"], implementation=MI.MINIWORLD,
                                       p_drop=0.25 if c["train"] else 0.0)
        args = (torch.randn(b, n, n, d, device="cuda", dtype=dt), torch.ones(b, n, device="cuda", dtype=torch.bool))
    elif c["kind"] == "bias_only_dit":
        from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock, pair_bias_all
        from miniworld_engine.modules.exceptions import ImplementationType as MI

        blocks = [BiasOnlyDiTBlock(d_cond=c["d_cond"], implementation=MI.MINIWORLD).cuda().to(pdt)
                  for _ in range(2 if c["hoist"] else 1)]
        for b in blocks:
            _randomize(b)
            b.train(c["train"])
        a, n, per = c["samples"], c["L"], c["pair"] == "per_sample"
        lead, nb = ((1, a), a) if per else ((a, 1), 1)  # per sample: [1, A, L, d] reading pair [A, L, L, 128]
        x = torch.randn(*lead, n, 768, device="cuda", dtype=dt)
        cond = torch.randn(*lead, n, c["d_cond"], device="cuda", dtype=dt)
        pair = torch.randn(nb, n, n, 128, device="cuda", dtype=dt)
        mask = torch.rand(nb, n, device="cuda") > 0.15 if c["masked"] else None
        if c["train"]:
            x, cond, pair = (t.requires_grad_(True) for t in (x, cond, pair))
        with torch.set_grad_enabled(c["train"]), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            biases = pair_bias_all(blocks, pair) if c["hoist"] else [None] * len(blocks)
            for b, bias in zip(blocks, biases, strict=True):
                x = b(x, cond, None if c["hoist"] else pair, mask, bias=bias)
            if c["train"]:
                x.float().sum().backward()
        torch.cuda.synchronize()
        return
    elif c["kind"] == "local_dit":
        from miniworld_engine.modules.exceptions import ImplementationType as MI
        from miniworld_engine.modules.local_dit import LocalDiTBlock

        model = LocalDiTBlock(128, 128, 16, 4, n=2, cross_attention=True, implementation=MI.TRITON)
        a, n = c["samples"], c["N"]
        args = (torch.randn(a, 1, n, 128, device="cuda", dtype=dt), torch.randn(a, 1, n, 128, device="cuda", dtype=dt),
                torch.randn(1, (n + 31) // 32, 32, 128, 16, device="cuda", dtype=dt))
    else:
        raise ValueError(c["kind"])
    model = model.cuda().to(pdt)
    _randomize(model)
    _step(model, args, c["train"], autocast)


def run(c: dict) -> dict:
    """Run one descriptor; never raises. ``status``: ok / skip (the module refused the shape) / oom / error."""
    import torch

    started = time.perf_counter()
    try:
        (_run_case if c["kind"] == "case" else _run_block)(c)
        status, detail = "ok", ""
    except torch.cuda.OutOfMemoryError as exc:
        status, detail = "oom", str(exc).splitlines()[0]
    except Exception as exc:
        from miniworld_engine.kernels.prebuilt import PrebuiltMissing

        status = "error" if isinstance(exc, PrebuiltMissing) else "skip"
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:400]
        if status == "error":
            traceback.print_exc()
    finally:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return {"id": c["id"], "status": status, "detail": detail, "seconds": round(time.perf_counter() - started, 2)}


def serve() -> int:
    """Worker loop: one JSON descriptor per stdin line, one ``RESULT <json>`` line per descriptor on stdout."""
    from miniworld_engine import settings

    settings.configure(triton_on_hopper_blackwell=False, engine_backend="auto")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        result = run(json.loads(line))
        print("RESULT " + json.dumps(result), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        return serve()
    if argv[:1] == ["list"]:
        for c in cases():
            print(json.dumps(c))
        return 0
    print("usage: python -m miniworld_engine.build.bundle_coverage serve|list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
