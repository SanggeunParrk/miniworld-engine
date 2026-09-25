"""Which aten ops (with input shapes) launch the small copy / cast kernels in one module call (ours)."""
import sys, pathlib, torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import make_inputs, make_module, module_fn
op, mode = sys.argv[1], sys.argv[2]
mod = make_module(op, "ours"); msa, pair, mask = make_inputs(op, 384, 1024)
mod.train(mode == "train")
if mode == "train":
    msa.requires_grad_(True); pair.requires_grad_(True); params = [msa, pair, *mod.parameters()]
    g = module_fn(op, mod, msa, pair, mask); gout = torch.randn(msa.shape if op == "pwa" else pair.shape, device="cuda", dtype=torch.bfloat16)
    f = lambda: torch.autograd.grad(g(), params, gout)
else:
    torch.set_grad_enabled(False); f = module_fn(op, mod, msa, pair, mask)
for _ in range(3): f()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as p:
    f(); torch.cuda.synchronize()
for e in p.key_averages(group_by_input_shape=True):
    if e.key in ("aten::copy_", "aten::to", "aten::_to_copy", "aten::contiguous", "aten::cat", "aten::fill_", "aten::zeros", "aten::clone", "aten::bmm", "aten::mm") and e.device_time_total > 0:
        print(f"{e.device_time_total:8.1f} us  {e.key:16s} x{e.count}  {str(e.input_shapes)[:150]}")
