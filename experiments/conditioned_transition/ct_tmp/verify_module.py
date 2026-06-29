"""End-to-end module check: ConditionedTransition (pytorch vs triton; inference + training)."""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from miniworld_kernels.modules.conditioned_transition import ConditionedTransition
from miniworld_kernels.modules.exceptions import ImplementationType


def cos(a, b):
    a = a.double().reshape(-1); b = b.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def run(d, dc, n, L, batch=2):
    dev = "cuda"
    ref = ConditionedTransition(d, dc, n, ImplementationType.PYTORCH).to(dev).float()
    tri = ConditionedTransition(d, dc, n, ImplementationType.TRITON).to(dev).float()
    # squeeze init is "zero" -> y would be identically 0; randomize so the test is meaningful.
    with torch.no_grad():
        for m in (ref.expand_a, ref.expand_b, ref.squeeze, ref.to_scale):
            m.weight.normal_(0, 0.05)
        ref.to_scale.bias.normal_(0, 0.1)
    tri.load_state_dict(ref.state_dict())  # same weights

    x = torch.randn(batch, L, d, device=dev, requires_grad=True)
    cond = torch.randn(batch, L, dc, device=dev)

    # inference (eval -> dispatch path)
    ref.eval(); tri.eval()
    with torch.no_grad():
        yr = ref(x, cond); yt = tri(x, cond)
    ci = cos(yt, yr)

    # training (train -> autograd Function)
    ref.train(); tri.train()
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    yr = ref(x1, cond); yt = tri(x2, cond)
    cf = cos(yt, yr)
    yr.sum().backward(); yt.sum().backward()
    cb = cos(x2.grad, x1.grad)
    cwa = cos(tri.expand_a.weight.grad, ref.expand_a.weight.grad)
    cbsc = cos(tri.to_scale.bias.grad, ref.to_scale.bias.grad)
    print(f"d={d:4d} L={L:5d} | inf_cos={ci:.6f} | train fwd={cf:.6f} dx={cb:.6f} dWa={cwa:.6f} dbsc={cbsc:.6f} | yshape={tuple(yt.shape)}")


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    run(128, 384, 2, 2048)   # atom
    run(768, 384, 2, 512)    # token
    print("DONE")


if __name__ == "__main__":
    main()
