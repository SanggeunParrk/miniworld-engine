"""Accuracy check for Transition Triton training backward."""

from __future__ import annotations

import torch

from miniworld_kernels.modules import ImplementationType, Transition


def clone_module(d_hidden: int) -> tuple[Transition, Transition]:
    ref = Transition(d_hidden=d_hidden, implementation=ImplementationType.PYTORCH).cuda()
    opt = Transition(d_hidden=d_hidden, implementation=ImplementationType.TRITON).cuda()
    opt.load_state_dict(ref.state_dict())
    ref.train()
    opt.train()
    return ref, opt


def metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.detach().float().reshape(-1)
    expected_f = expected.detach().float().reshape(-1)
    diff = actual_f - expected_f
    rel = diff.norm().div(expected_f.norm().clamp_min(1e-20))
    return float(diff.abs().max().item()), float(rel.item())


def main() -> None:
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ref, opt = clone_module(128)
    x_ref = torch.randn(1, 64, 64, 128, device="cuda", dtype=torch.bfloat16)
    x_opt = x_ref.detach().clone()
    dy = torch.randn_like(x_ref)
    x_ref.requires_grad_(True)
    x_opt.requires_grad_(True)

    y_ref = ref(x_ref)
    y_opt = opt(x_opt)
    y_ref.backward(dy)
    y_opt.backward(dy)

    checks = {
        "out": metric(y_opt, y_ref),
        "x.grad": metric(x_opt.grad, x_ref.grad),
    }
    for name, ref_param in ref.named_parameters():
        opt_param = dict(opt.named_parameters())[name]
        checks[f"{name}.grad"] = metric(opt_param.grad, ref_param.grad)

    for name, (max_abs, rel) in checks.items():
        print(f"{name}: max_abs={max_abs:.6g} rel={rel:.6g}")

    worst_rel = max(rel for _, rel in checks.values())
    if worst_rel > 1.5e-2:
        raise SystemExit(f"relative error too high: {worst_rel:.6g}")


if __name__ == "__main__":
    main()
