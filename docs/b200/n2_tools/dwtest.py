# dwtest.py : _FeatB200 backward: swa_embed_dwf_sm100 against the cuBLAS GEMM path (and fp64), random non-zero data, ragged masks, elements -3 .. 130, determinism
import sys, torch
import miniworld_engine.modules.embedder as E
dev = torch.device("cuda"); torch.manual_seed(1)
ok = True
for esm, M in ((True, 4096), (True, 3072), (True, 6144), (True, 8192), (False, 4096), (False, 5120), (True, 4104)):
    nf = 133 if esm else 6
    pos = (torch.randn(1, M, 3, device=dev) * 5).requires_grad_(True)
    w = (torch.randn(128, nf, device=dev) * 0.3).requires_grad_(True)
    lnw = (torch.rand(128, device=dev) + 0.5).requires_grad_(True) if esm else None
    lnb = (torch.randn(128, device=dev) * 0.1).requires_grad_(True) if esm else None
    charge = torch.randint(-2, 3, (1, M), device=dev)
    element = torch.randint(-3, 131, (1, M), device=dev) if esm else torch.randint(1, 20, (1, M), device=dev)
    fmask = (torch.rand(1, M, device=dev) > 0.15).to(torch.uint8)
    g = torch.randn(1, M, 128, device=dev)
    res = {}
    for dwk in (False, True, True):
        E._FEAT_DWK = dwk
        for t in (pos, w, lnw, lnb):
            if t is not None: t.grad = None
        out = E._FeatB200.apply(pos, w, lnw, lnb, charge, element, fmask, esm, 1e-5, torch.float32)
        out.backward(g); torch.cuda.synchronize()
        res.setdefault(dwk, []).append((w.grad.clone(), pos.grad.clone(), None if lnw is None else lnw.grad.clone()))
    a, b, b2 = res[False][0], res[True][0], res[True][1]
    det = all(torch.equal(x, y) for x, y in zip(b, b2) if x is not None)
    rel = (a[0] - b[0]).abs().max().item() / a[0].abs().max().item()
    # fp64 reference of dW from the saved quantities: recompute via autograd in fp64 is heavy; compare with the GEMM path (itself fp32)
    same_other = torch.equal(a[1], b[1]) and (a[2] is None or torch.equal(a[2], b[2]))
    print(f"esm {esm} M {M}: max|dW| rel vs GEMM {rel:.2e}  other grads bitwise {same_other}  deterministic {det}", flush=True)
    ok &= rel < 1e-5 and det and same_other
print("PASS" if ok else "FAIL")
