import sys, torch
from pathlib import Path
from miniworld_engine.kernels.triangle_attention.cuda import b200_bidir as BB
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
ensure_cuda_home()
src = Path(BB.__file__).parent / "b200_sources"
new = load_extension(name="n1_sm_t", sources=[str(src / "lnbwd_sm100.cu")], extra_cuda_cflags=[*host_flags(), *BB._FLAGS, *gencodes("100a")], verbose=False)
old = BB.ext()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[n // 2]
H = 4
for L in [int(x) for x in sys.argv[1:]]:
    LL = L * L; torch.manual_seed(0)
    big = (torch.randn(260, LL, device="cuda", dtype=torch.bfloat16) * 3); pl = big[256:260]
    mask = torch.rand(L, device="cuda") > 0.1; mask[0] = True
    for nstart, nm in ((H, "start"), (0, "end")):
        for mk in (None, mask):
            ref = lambda: torch.softmax(old.logits(pl, mk, nstart, L), -1).to(torch.bfloat16)
            r = ref(); n = new.softmax_planes(pl, mk, nstart, L)
            print(f"L{L} {nm} mask={mk is not None}: torch chain {tm(ref):.1f}  fused {tm(lambda: new.softmax_planes(pl, mk, nstart, L)):.1f}  bitwise equal {torch.equal(r, n)}  maxdiff {float((r.float()-n.float()).abs().max()):.1e}", flush=True)
    # bidirectional: 2H planes, nstart = H
    pl2 = big[248:256]
    r = torch.softmax(old.logits(pl2, None, H, L), -1).to(torch.bfloat16); n = new.softmax_planes(pl2, None, H, L)
    print(f"L{L} bidir 2H planes: bitwise equal {torch.equal(r, n)}", flush=True)
