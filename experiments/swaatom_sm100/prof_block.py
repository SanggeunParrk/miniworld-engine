"""Per-kernel CUDA time of the whole block (hoisted modulation), H100-fused Triton path vs sm100 kernels installed."""
import sys, torch
from torch.profiler import profile, ProfilerActivity
from common import make, HW
import h100_fused as SF
import b200_block
mode, L, which = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if which == "sm100":
    b200_block.install()
else:
    SF._qkvg_fwd_cuda_ok = SF._ffn_fwd_cuda_ok = SF._ffn_bwd_cuda_ok = (lambda *x: False)
A = 5 if mode == "inference" else 48
q, cb, cos, sin, su, w = make(A, 8 * L if L < 1000 else L)
W = ("wqkv", "wg", "wo", "wu", "wd")
train = mode == "training"
if train:
    q.requires_grad_(); cb.requires_grad_(); [w[k].requires_grad_() for k in w]
dy = torch.randn_like(q)
modc = SF.hoist_mod(cb, w["wmod"]).detach()
def step():
    if train:
        SF.swa_block(q, SF.hoist_mod(cb, w["wmod"]), cos, sin, su, *(w[k] for k in W), 1, HW).backward(dy)
    else:
        with torch.no_grad():
            SF.swa_block(q, modc, cos, sin, su, *(w[k] for k in W), 1, HW)
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(3): step()
    torch.cuda.synchronize()
rows = sorted(((e.key, e.device_time_total / 3, e.count // 3) for e in pr.key_averages() if e.device_time_total > 0), key=lambda r: -r[1])
tot = sum(r[1] for r in rows)
print(f"{which} {mode} L{L}: total kernel time {tot:.0f} us")
for k, t, n in rows[:16]:
    print(f"  {t:9.1f} us  {100 * t / tot:5.1f}%  x{n:<3d} {k[:90]}")
