"""Native BF16 full-model MPNN training via the existing shared measurement harness."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from benchmarks.runners.bench import BenchConfig, measured_result
from benchmarks.runners.measurement import (
    benchmark_source_hash,
    compile_for_benchmark,
    observe_execution,
)
from benchmarks.runners.mpnn_native_reference import NativeNaiveMPNN

from miniworld_engine.modules.mpnn import (
    NaiveProteinMPNN,
    ProteinMPNN,
    ProteinMPNNConfig,
    iter_reference_parameter_pairs,
    load_cssb_weights,
    production_tensor_in_reference_layout,
)
from miniworld_engine.modules.mpnn.layers import EncoderLayer

POLICIES = ("naive", "pytorch", "current", "compute", "compute-save-rbf")


def config_for(policy, dropout):
    pure = policy == "pytorch"
    return ProteinMPNNConfig(
        encoder_depth=3, decoder_depth=3, node_width=128, edge_width=128, hidden_width=128,
        k_neighbors=48, coordinate_noise=0, dropout=dropout, block_linear_min_edges=0,
        feature_backend="pytorch" if policy in {"pytorch", "compute-save-rbf"} else "memory",
        knn_backend="cdist", edge_w1_recompute="off", encoder_node_w1_recompute="off",
        decoder_node_w1_recompute="off", transition_recompute="off",
        message_backend="pytorch" if pure else "triton_compute",
        edge_mlp_backend="pytorch" if pure else "triton_compute", edge_norm_backend="pytorch",
        relative_position_backend="off" if pure else "triton", edge_dropout_backend="pytorch",
        edge_tail_backend="off" if pure else "triton_compute",
        node_message_backend="off" if pure else "triton_compute",
    )


def make_model(policy, dropout):
    torch.manual_seed(20260913)
    reference = NativeNaiveMPNN(dropout=dropout)
    torch.nn.init.normal_(reference.W_out.weight, std=128**-0.5)
    if policy == "naive":
        return reference.cuda().bfloat16().train(), None
    config = config_for(policy, dropout)
    model = ProteinMPNN(config)
    load_cssb_weights(model, reference)
    if policy == "current":
        cast(EncoderLayer, model.encoder.layers[0]).node_message_backend = "triton"
    return model.cuda().bfloat16().train(), config


def make_inputs(batch, length):
    torch.manual_seed(20260914)
    ca = torch.randn(batch, length, 1, 3, device="cuda").cumsum(1)
    return [
        ca + torch.randn(batch, length, 4, 3, device="cuda") * .3,
        torch.randint(21, (batch, length), device="cuda"),
        torch.ones(batch, length, device="cuda"),
        torch.arange(length, device="cuda")[None].repeat(batch, 1),
        torch.zeros(batch, length, device="cuda", dtype=torch.int64),
        torch.stack([torch.randperm(length, device="cuda") for _ in range(batch)]),
        (torch.arange(length, device="cuda") // 8)[None].repeat(batch, 1),
    ]


def logits_for(model, inputs):
    assert not torch.is_autocast_enabled("cuda")
    if isinstance(model, NaiveProteinMPNN):
        return model(*inputs, inputs[2], None, use_checkpoint=False)
    return model(*inputs, checkpoint_layers=False)


def param_audit(model, require_grad=False):
    norms = {id(p) for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
             for p in m.parameters(recurse=False)}
    counts = {"bfloat16": 0, "float32": 0}
    for p in model.parameters():
        expected = torch.float32 if id(p) in norms else torch.bfloat16
        assert p.dtype == expected
        counts[str(p.dtype).removeprefix("torch.")] += p.numel()
        if require_grad:
            assert p.grad is not None and p.grad.dtype == expected
            assert bool(torch.isfinite(p.grad).all())
    assert counts == {"bfloat16": 1656389, "float32": 4096}, counts
    return counts


def dispatch_witness(step):
    from miniworld_engine.kernels.mpnn_edge_tail.triton import compute as tail
    from miniworld_engine.kernels.mpnn_message.triton import main as message
    from miniworld_engine.kernels.mpnn_node_message.triton import main as node
    from miniworld_engine.modules.mpnn import features
    calls, originals = {}, []
    def wrap(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        def run(*a, **kw):
            key = label + (f":SAVE_PREACT={kw['SAVE_PREACT']}" if label == "node" else "")
            calls[key] = calls.get(key, 0) + 1
            return original(*a, **kw)
        setattr(owner, name, run)
    for label, tuner in [("node", node._node_message_fwd_kernel),
                         ("message", message._projection_fwd_kernel),
                         ("tail_fwd", tail._project_edge), ("tail_bwd", tail._edge_backward)]:
        wrap(tuner, "run", label)
    wrap(features, "_radial_weight_gradient", "radial_bwd")
    try:
        step()
        torch.cuda.synchronize()
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
    return calls


def relative(a, b):
    return float((a.detach().double() - b.detach().double()).norm() / b.detach().double().norm())


def verify():
    inputs = make_inputs(2, 128)
    torch.manual_seed(20260913)
    adapted = NativeNaiveMPNN(dropout=0).cuda().train()
    torch.nn.init.normal_(adapted.W_out.weight, std=128**-0.5)
    frozen = NaiveProteinMPNN(k_neighbors=48, dropout=0, augment_trans=0, augment_rot=0).cuda().train()
    frozen.load_state_dict(adapted.state_dict())
    with torch.no_grad():
        a, b = logits_for(adapted, inputs), logits_for(frozen, inputs)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    del adapted, frozen, a, b
    reference, _ = make_model("naive", 0)
    expected = logits_for(reference, inputs)
    F.cross_entropy(expected.float(), inputs[1]).backward()
    rows = []
    for policy in ("pytorch", "current", "compute", "compute-save-rbf"):
        model, _ = make_model(policy, 0)
        forward = compile_for_benchmark(lambda model=model: logits_for(model, inputs), fullgraph=True)
        with observe_execution() as evidence:
            actual = forward()
            F.cross_entropy(actual.float(), inputs[1]).backward()
        assert evidence.compiled
        param_audit(model, True)
        pairs = list(iter_reference_parameter_pairs(reference, model))
        actual_grad = torch.cat([production_tensor_in_reference_layout(n, cast(torch.Tensor, p.grad)).flatten().double()
                                 for n, _, _, p in pairs])
        expected_grad = torch.cat([cast(torch.Tensor, p.grad).flatten().double() for _, p, _, _ in pairs])
        row = {"policy": policy, "logit_relative_l2": relative(actual, expected),
               "gradient_relative_l2": relative(actual_grad, expected_grad),
               "gradient_cosine": float(F.cosine_similarity(actual_grad, expected_grad, dim=0)),
               "compile_observed": evidence.compiled}
        print(json.dumps(row), flush=True)
        assert row["logit_relative_l2"] < .02 and row["gradient_relative_l2"] < .02
        assert row["gradient_cosine"] > .999
        rows.append(row)
        del forward, model, actual, actual_grad, expected_grad, pairs
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
    return {"status": "ok", "batch": 2, "length": 128, "dropout": 0,
            "native_adapter_fp32_matches_frozen_bitwise": True, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, default="current")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--validation-steps", type=int, default=20)
    parser.add_argument("--expected-device", default="A6000")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing record")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    assert args.expected_device in props.name
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint_calls = []
    def forbidden(*a, **kw):
        checkpoint_calls.append(True)
        raise AssertionError("checkpoint API is forbidden in this experiment")
    torch.utils.checkpoint.checkpoint = forbidden
    source = benchmark_source_hash()
    record = {"status": "running", "phase": "setup", "configuration": {**vars(args), "out": str(args.out)},
              "device": props.name, "total_memory_gib": props.total_memory / 2**30,
              "source_hash": source, "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "autocast": False, "precision": "bf16-native-fp32-norm", "compile": True,
              "cudagraph": "disabled", "timing_includes_optimizer": True,
              "allocator": os.environ.get("PYTORCH_ALLOC_CONF", ""), "allocator_cap": None,
              "scope": "fullgraph forward/loss + AOT backward + eager AdamW; no accumulation or checkpoint; dropout .25",
              "checkpoint_api_calls": 0}
    def save():
        record["checkpoint_api_calls"] = len(checkpoint_calls)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(args.out)
        print(json.dumps({k: record[k] for k in ["status", "phase"]}), flush=True)
    phase = ["setup"]
    save()
    try:
        if args.verify:
            record.update(verify())
            record["phase"] = "validation_done"
            save()
            return 0
        model, config = make_model(args.policy, .25)
        params = list(model.parameters())
        record["parameter_counts"] = param_audit(model)
        record["model_config"] = asdict(config) if config else {"naive": True, "k_neighbors": 48, "dropout": .25}
        record["initial_weight_sha256"] = hashlib.sha256(b"".join(
            p.detach().cpu().view(torch.uint8).numpy().tobytes() for p in params)).hexdigest()
        inputs = make_inputs(args.batch, args.length)
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        def loss_fn():
            logits = logits_for(model, inputs)
            assert logits.dtype == torch.bfloat16
            return F.cross_entropy(logits.float(), inputs[1])
        forward = compile_for_benchmark(loss_fn, fullgraph=True)
        def step():
            optimizer.zero_grad(set_to_none=True)
            phase[0] = "forward"
            loss = forward()
            phase[0] = "backward"
            loss.backward()
            phase[0] = "optimizer"
            optimizer.step()
            return loss.detach(), tuple(p.grad for p in params)
        record["phase"] = "warmup"
        save()
        start = time.monotonic()
        warmup = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with observe_execution() as evidence:
                loss, _ = step()
                torch.cuda.synchronize()
            assert evidence.compiled and torch.isfinite(loss)
            param_audit(model, True)
            warmup.append({"loss": float(loss), "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                           "reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        record.update(warmup=warmup, warmup_seconds=time.monotonic()-start, phase="timing")
        for p, state in optimizer.state.items():
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == p.dtype
        save()
        conf = BenchConfig(target="protein_mpnn_native_step", level="module", compile=True,
                           mode="training", metric="time", precision="bf16", cudagraph="disabled")
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.repeats):
            result = measured_result(conf=conf, func=step, grad_to_none=[], params=params,
                                     is_train=True, input_dtype="float32", parameter_dtype="bfloat16+float32",
                                     execution_path=args.policy, reference="naive")
            assert result.compiled and result.cudagraph == "disabled"
            samples.append(result._asdict())
        record.update(samples=samples, median_step_ms=statistics.median(s["value"] for s in samples),
                      timing_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      timing_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, phase="validation")
        save()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        calls = dispatch_witness(step)
        if args.policy in {"naive", "pytorch"}:
            assert not calls, calls
        else:
            assert calls.get("tail_fwd") == calls.get("tail_bwd") == 3, calls
            assert calls.get("message") == 3, calls
            assert calls.get("node:SAVE_PREACT=False", 0) == (1 if args.policy == "current" else 0), calls
            assert calls.get("node:SAVE_PREACT=True", 0) == (2 if args.policy == "current" else 3), calls
            assert calls.get("radial_bwd", 0) == (0 if args.policy == "compute-save-rbf" else 1), calls
        record["actual_kernel_dispatch"] = calls
        memory = []
        for _ in range(args.validation_steps):
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loss, _ = step()
            torch.cuda.synchronize()
            assert torch.isfinite(loss)
            param_audit(model, True)
            memory.append({"loss": float(loss), "allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                           "reserved_gib": torch.cuda.max_memory_reserved()/2**30})
        assert benchmark_source_hash() == source
        record.update(status="ok", phase="done", memory=memory,
                      peak_allocated_gib=max(m["allocated_gib"] for m in warmup+memory),
                      peak_reserved_gib=max(m["reserved_gib"] for m in warmup+memory),
                      measurement_note="Timer includes a separate 256 MiB flush buffer; reported training peak uses warmup and validation.")
        save()
        return 0
    except Exception as exc:
        record.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                      error=f"{type(exc).__name__}: {exc}", operation_phase=phase[0],
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
