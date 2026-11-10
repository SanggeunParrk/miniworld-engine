"""A100 probe: SmoothLddtLoss cuda launch variants vs triton (graph replay, L2 flushed, median ms)."""
import sys
import torch
from miniworld_engine.modules.smooth_lddt import SmoothLddtLoss
from miniworld_engine.kernels.smooth_lddt.cuda import sm100 as K

flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")


def bench(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    lengths = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
    gen = torch.Generator(device="cuda").manual_seed(0)
    for n in lengths:
        for mode in ("eval", "train"):
            m = SmoothLddtLoss(backend="cuda")
            inp = m.make_inputs(length=n, samples=1, depth=None, dtype=torch.float32, device="cuda", generator=gen)
            x = inp["x_pred"].requires_grad_(mode == "train")

            def step(backend):
                def f():
                    if mode == "train":
                        x.grad = None
                        getattr(m, f"_forward_{backend}")(x, inp["gt"], inp["is_nuc"], inp["mask"]).backward()
                    else:
                        with torch.no_grad():
                            getattr(m, f"_forward_{backend}")(x, inp["gt"], inp["is_nuc"], inp["mask"])
                return f
            res = {"triton": bench(step("triton"))}
            nb = -(-n // 128)
            variants = [("default", {})]
            if n * 32 <= K.FULL_SMEM:
                for v in K.FWD_VARIANTS:
                    for fb in (128, 256, 512):
                        if mode == "eval" and fb != 128:
                            continue
                        variants.append((f"full{v}/b{fb}", {"FORCE_FWD": v, "FORCE_FB": fb}))
            for p in sorted({1, 2, 4, 8, 16, 32, nb} - {0}):
                if p <= nb:
                    variants.append((f"split{p}", {"FORCE_SPLIT": p}))
            for name, kn in variants:
                old = {k: getattr(K, k) for k in ("FORCE_FWD", "FORCE_FB", "FORCE_SPLIT")}
                for k, v in kn.items():
                    setattr(K, k, v)
                try:
                    res[name] = bench(step("cuda"))
                except Exception as e:  # noqa: BLE001
                    res[name] = f"ERR {type(e).__name__}"
                for k, v in old.items():
                    setattr(K, k, v)
            best = min((v, k) for k, v in res.items() if isinstance(v, float) and k != "triton")
            print(f"N={n} {mode} triton={res['triton']:.4f} best={best[1]}:{best[0]:.4f} ::", " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)


main()
