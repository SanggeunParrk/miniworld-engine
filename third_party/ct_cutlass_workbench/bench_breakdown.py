"""FWD-only and FWD+BWD breakdown: ct_full vs cond_transition_train, CUDA graph.
Reports forward us and (fwd+bwd) us separately so we see where CUTLASS wins/loses."""
import sys
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except Exception:
    pass
sys.path.insert(0, "/home/psk6950/miniworld-kernels/_ct_cutlass")
from miniworld_kernels.kernels.conditioned_transition.triton.training import cond_transition_train, set_forward_mode
from ct_full import cond_transition_train_full


def make(M, d, n=2, dc=384, dev="cuda"):
    g = torch.Generator(dev).manual_seed(0)
    f = lambda *s: torch.randn(*s, device=dev, generator=g)
    ts = [f(M, d), f(M, dc), f(n*d, d)/d**0.5, f(n*d, d)/d**0.5,
          f(d, n*d)/(n*d)**0.5, f(d, dc)/dc**0.5, torch.full((d,), -2.0, device=dev)]
    return tuple(t.detach().requires_grad_(True) for t in ts)


def gbench(make_t, run, it=100, wu=10):
    t = make_t()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(t)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): run(t)
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3


def fwd(fn):
    return lambda t: fn(*t)
def fwdbwd(fn):
    return lambda t: torch.autograd.grad(fn(*t), t, torch.ones_like(fn(*t)))


STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]
set_forward_mode("auto")
shapes = [(s, d, M) for s, d, Ms in STREAMS for M in Ms]
print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cut_fwd':>8} {'our_fwd':>8} {'fwd_x':>6} | "
      f"{'cut_fb':>8} {'our_fb':>8} {'fb_x':>6} | {'cut_bwd':>8} {'our_bwd':>8} {'bwd_x':>6}")
for s, d, M in shapes:
    cf = gbench(lambda: make(M, d), fwd(cond_transition_train_full))
    of = gbench(lambda: make(M, d), fwd(cond_transition_train))
    cfb = gbench(lambda: make(M, d), fwdbwd(cond_transition_train_full))
    ofb = gbench(lambda: make(M, d), fwdbwd(cond_transition_train))
    cb, ob = cfb - cf, ofb - of
    print(f"{s:>6} {M:>6} {d:>4} | {cf:8.1f} {of:8.1f} {of/cf:5.2f}x | "
          f"{cfb:8.1f} {ofb:8.1f} {ofb/cfb:5.2f}x | {cb:8.1f} {ob:8.1f} {ob/cb:5.2f}x")
print("BREAKDOWN DONE")
