"""RTX 5090: where the default token DiT / atom DiT block spends its time (step 1 of the FP8 routes), and the e4m3 GEMM rates of the shapes.

    python dit_profile.py gemm                         # bf16 mm vs torch._scaled_mm e4m3 (row-wise / tensor-wise) on the token DiT GEMM shapes
    python dit_profile.py prof Module:A:L ...          # torch.profiler kernel table of one block call (eager, after warm-up)
"""

import sys

import torch
from dit_common import Rig, case, graph, time_graph

E4 = torch.float8_e4m3fn


def gemm(atom=False):
    print(torch.cuda.get_device_name(), torch.__version__)
    shapes = ((128, 256, "atom q|g"), (128, 256, "atom k|v"), (128, 128, "atom out"), (128, 512, "atom expand"), (256, 128, "atom squeeze")) if atom else ((768, 3072, "expand a|b"), (768, 4608, "expand a|b|u"), (768, 3072, "qkvg"), (1536, 768, "squeeze"), (768, 768, "out"))
    for K, N, name in shapes:
        for M in (5120, 20480, 81920) if atom else (1280, 2560, 5120, 10240):
            x = torch.randn(M, K, device="cuda").bfloat16()
            w = torch.randn(N, K, device="cuda").bfloat16()
            x8, w8 = x.to(E4), w.to(E4)
            sx, sw = torch.ones(M, 1, device="cuda"), torch.ones(1, N, device="cuda")
            one = torch.ones((), device="cuda")
            fs = {"bf16": lambda: torch.mm(x, w.t())}
            try:
                torch._scaled_mm(x8, w8.t(), sx, sw, out_dtype=torch.bfloat16)
                fs["e4m3 rowwise"] = lambda: torch._scaled_mm(x8, w8.t(), sx, sw, out_dtype=torch.bfloat16)
            except Exception as e:  # noqa: BLE001
                print("rowwise unsupported:", str(e).splitlines()[0])
            fs["e4m3 tensor"] = lambda: torch._scaled_mm(x8, w8.t(), one, one, out_dtype=torch.bfloat16)
            res = {}
            for k, f in fs.items():
                g, gf, _h = graph(f, n=10, flush=False)
                res[k] = time_graph(g, gf, n=10)
            fl = 2 * M * K * N
            print(f"{name:14s} M {M:6d} K {K} N {N}: " + "  ".join(f"{k} {v:7.1f} us ({fl / v / 1e6:4.0f} TF)" for k, v in res.items()), flush=True)


def prof(spec):
    mod, A, L, *opt = spec.split(":")
    rig = Rig(case(mod, int(L), int(A), pick=opt[0] if opt else None))
    for _ in range(3):
        rig.run()
    g, gf, _h = graph(rig.step, n=3)
    print(f"{spec}: graph {time_graph(g, gf):.1f} us / call", flush=True)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            rig.run()
    tot = {}
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            tot.setdefault(e.name, [0.0, 0])
            tot[e.name][0] += e.device_time / 5
            tot[e.name][1] += 1
    s = sum(v[0] for v in tot.values())
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {v[0]:8.1f} us {100 * v[0] / s:5.1f} %  x{v[1] // 5:<3d} {k[:150]}")
    print(f"  sum {s:.1f} us", flush=True)


if __name__ == "__main__":
    if sys.argv[1] in ("gemm", "gemm_atom"):
        gemm(sys.argv[1] == "gemm_atom")

    else:
        for sp in sys.argv[2:]:
            prof(sp)
