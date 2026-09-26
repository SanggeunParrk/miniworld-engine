"""sm100 triangle attention forward: correctness vs an fp32 reference (and the Triton flash row), sustained timing.
    TA_SRC=<cu> python t_triattn.py"""
import os, sys, pathlib, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
sys.path.insert(0, os.environ["OPT_CORE_DIR"])
from opt_core.kernels import triattn as ta
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
tag = os.environ.get("TA_TAG", "cur")
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / f"triattn_{tag}"; d.mkdir(parents=True, exist_ok=True)
ext = load(f"triattn_{tag}", [os.environ.get("TA_SRC", str(src / "triattn_sm100.cu"))], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
H, D = 4, 32
Ls = [int(x) for x in os.environ.get("TA_L", "384,768").split(",")]
for L in Ls:
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(1, L, H, L, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))
    bias = torch.randn(1, 1, H, L, L, device="cuda", generator=g).to(torch.bfloat16)
    sc = D ** -0.5
    ref = ta.sdpa_reference(q.float(), k.float(), v.float(), bias.float())
    out, flags = ext.triattn_fwd(q, k, v, bias, sc)
    fl_ = ta.triangle_attention(q, k, v, bias.float(), None, None, word="flash")
    e_ours = ((out.float() - ref).norm() / ref.norm()).item(); e_fl = ((fl_.float() - ref).norm() / ref.norm()).item()
    print(f"L={L}: rel err vs fp32  ours {e_ours:.2e}  flash {e_fl:.2e}  max|ours-ref| {(out.float()-ref).abs().max().item():.3e}  flags {int(flags.item())}", flush=True)
    fl = 4 * L * H * L * L * D
    for name, fn in (("ours", lambda: ext.triattn_fwd(q, k, v, bias, sc)), ("flash", lambda: ta.triangle_attention(q, k, v, bias.float(), None, None, word="flash"))):
        r = sustained(fn, secs=2.0)
        print(f"   {name:6s} {r['ms']*1e3:8.1f} us  {r['J']*1e3:7.2f} mJ  {r['W']:5.0f} W  {fl/(r['ms']*1e-3)/1e12:6.1f} TF/s", flush=True)
