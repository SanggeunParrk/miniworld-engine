"""Native BF16 MPNN leave-one-group-out training and compiler attribution.

Uses the shared compilation witness/timer and the full-model experiment's inputs.
Run only in an allocated GPU job. Each policy must run in a fresh process.
"""
import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from benchmarks.runners.bench import BenchConfig, measured_result
from benchmarks.runners.measurement import (
    benchmark_source_hash,
    compile_for_benchmark,
    observe_execution,
)
from benchmarks.runners.mpnn_training import (
    dispatch_witness,
    logits_for,
    make_inputs,
    make_model,
    param_audit,
)

POLICIES = ("pytorch", "all_compute", "no_node", "no_tail", "no_decoder", "no_position", "current")


def model_for(policy, dropout):
    model, _config = make_model("pytorch" if policy == "pytorch" else
                               "current" if policy == "current" else "compute-save-rbf", dropout)
    if policy == "no_node":
        for layer in model.encoder.layers:
            layer.node_message_backend = "off"
            layer.node_message.reduction_backend = "pytorch"
    elif policy == "no_tail":
        for layer in model.encoder.layers:
            layer.edge_tail_backend = "off"
            layer.edge_message.edge_mlp_backend = "pytorch"
    elif policy == "no_decoder":
        for layer in model.decoder.layers:
            layer.node_message.reduction_backend = "pytorch"
    elif policy == "no_position":
        model.backbone_features.relative_position.backend = "off"
    resolved = {
        "node": [x.node_message_backend for x in model.encoder.layers],
        "encoder_message": [x.node_message.reduction_backend for x in model.encoder.layers],
        "tail": [x.edge_tail_backend for x in model.encoder.layers],
        "edge_mlp": [x.edge_message.edge_mlp_backend for x in model.encoder.layers],
        "decoder_message": [x.node_message.reduction_backend for x in model.decoder.layers],
        "position": model.backbone_features.relative_position.backend,
        "feature": model.backbone_features.feature_backend,
    }
    return model, resolved


def check_calls(policy, calls):
    if policy == "pytorch":
        assert not calls, calls
        return
    expected = {
        "node:SAVE_PREACT=True": 0 if policy == "no_node" else 2 if policy == "current" else 3,
        "node:SAVE_PREACT=False": int(policy == "current"),
        "tail_fwd": 0 if policy == "no_tail" else 3,
        "tail_bwd": 0 if policy == "no_tail" else 3,
        "message": 0 if policy == "no_decoder" else 3,
        "radial_bwd": int(policy == "current"),
    }
    assert all(calls.get(k, 0) == v for k, v in expected.items()), (calls, expected)


def verify(policy):
    inputs = make_inputs(2, 128)
    reference, _ = model_for("pytorch", 0)
    expected = logits_for(reference, inputs)
    F.cross_entropy(expected.float(), inputs[1]).backward()
    model, _ = model_for(policy, 0)
    forward = compile_for_benchmark(lambda: logits_for(model, inputs), fullgraph=True)
    with observe_execution() as evidence:
        actual = forward()
        F.cross_entropy(actual.float(), inputs[1]).backward()
    assert evidence.compiled
    param_audit(model, True)
    actual_grad = torch.cat([p.grad.detach().flatten().double() for p in model.parameters()])
    expected_grad = torch.cat([p.grad.detach().flatten().double() for p in reference.parameters()])
    errors = {
        "logit_relative_l2": float((actual.detach().double()-expected.detach().double()).norm()/expected.detach().double().norm()),
        "gradient_relative_l2": float((actual_grad-expected_grad).norm()/expected_grad.norm()),
        "gradient_cosine": float(F.cosine_similarity(actual_grad, expected_grad, dim=0)),
        "compile_observed": evidence.compiled, "batch": 2, "length": 128, "dropout": 0,
    }
    assert errors["logit_relative_l2"] < .02 and errors["gradient_relative_l2"] < .02
    assert errors["gradient_cosine"] > .999
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert "A6000" in torch.cuda.get_device_name()
    assert not torch.is_autocast_enabled("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint API not allowed")
    torch.utils.checkpoint.checkpoint = forbidden
    source = benchmark_source_hash()
    record = {"policy": args.policy, "status": "running", "source_hash": source,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": torch.cuda.get_device_name(), "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
              "torch": torch.__version__, "cuda": torch.version.cuda, "autocast": False,
              "precision": "native_bf16_fp32_norm", "batch": 8, "length": 8192, "neighbors": 48,
              "dropout": .25, "compile": True, "cudagraph": "disabled",
              "timing_scope": "forward+loss+backward+eager_AdamW+zero_grad", "checkpoint": False}
    def save():
        args.out.write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps({k: record[k] for k in ["policy", "status", "phase"] if k in record}), flush=True)
    save()
    try:
        if args.verify:
            record.update(accuracy=verify(args.policy), status="ok", phase="verified")
            save()
            return
        model, resolved = model_for(args.policy, .25)
        record.update(resolved_policy=resolved, parameter_counts=param_audit(model))
        inputs = make_inputs(8, 8192)
        params = list(model.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        def loss_fn():
            logits = logits_for(model, inputs)
            assert logits.dtype == torch.bfloat16
            return F.cross_entropy(logits.float(), inputs[1])
        forward = compile_for_benchmark(loss_fn, fullgraph=True)
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = forward()
            loss.backward()
            optimizer.step()
            return loss.detach(), tuple(p.grad for p in params)
        record["phase"] = "warmup"
        save()
        for _ in range(2):
            with observe_execution() as evidence:
                loss, _ = step()
                torch.cuda.synchronize()
            assert evidence.compiled and torch.isfinite(loss)
            param_audit(model, True)
        assert all(s["exp_avg"].dtype == p.dtype == s["exp_avg_sq"].dtype for p, s in optimizer.state.items())
        conf = BenchConfig(target="protein_mpnn_attribution", level="module", compile=True,
                           mode="training", metric="time", precision="bf16", cudagraph="disabled")
        record["phase"] = "timing"
        save()
        samples = []
        for _ in range(7):
            result = measured_result(conf=conf, func=step, grad_to_none=[], params=params,
                                     is_train=True, input_dtype="float32", parameter_dtype="bfloat16+float32",
                                     execution_path=args.policy, reference="pytorch")
            assert result.compiled and result.cudagraph == "disabled"
            samples.append(result._asdict())
        record.update(samples=samples, median_step_ms=statistics.median(s["value"] for s in samples), phase="validation")
        save()
        calls = dispatch_witness(step)
        check_calls(args.policy, calls)
        record["actual_kernel_dispatch"] = calls
        memory = []
        for _ in range(5):
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
        record.update(memory=memory, peak_allocated_gib=max(x["allocated_gib"] for x in memory))
        if args.profile:
            record["phase"] = "profiling"
            save()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, with_stack=False) as profile:
                step()
                torch.cuda.synchronize()
            profile.export_chrome_trace(str(args.out.with_suffix(".trace.json")))
            args.out.with_suffix(".profile.txt").write_text(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
        assert benchmark_source_hash() == source
        record.update(status="ok", phase="done")
        save()
    except Exception as exc:
        record.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
