"""FP8 pieces vs the bf16 ones on the same operands: dequantised A2 / BT vs bf16 A2 / BT (e4m3 rounding only), e4m3 product vs bf16 product."""
import torch, time
from miniworld_engine.kernels.integrations import opm_train as ot, opm_fp8 as F
dev = torch.device("cuda"); bf = torch.bfloat16
for cm, cz, L, S, pbias in ((64, 128, 256, 1024, False), (64, 128, 192, 2048, True), (128, 256, 128, 512, False), (256, 512, 64, 256, False)):
    torch.manual_seed(0)
    m = torch.randn(S, L, cm, device=dev, dtype=bf); mk = torch.rand(S, L, device=dev) > 0.1
    gam = 1 + 0.1 * torch.randn(cm, device=dev); bet = 0.1 * torch.randn(cm, device=dev)
    wa = (torch.randn(32, cm, device=dev) * cm ** -0.5).to(bf); wb = (torch.randn(32, cm, device=dev) * cm ** -0.5).to(bf)
    pb = 0.1 * torch.randn(64, device=dev) if pbias else None
    e16 = ot._ext(dev, native=True, tma=False, cm=cm, cz=cz); e8 = ot._ext(dev, native=True, tma=False, cm=cm, cz=cz, fp8=True)
    a2, bt, _, mk16, bits = e16.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, pb, False)
    s = F.bound_scales(gam, bet, wa, wb, pb).to(dev); amax = torch.zeros(2, device=dev)
    a8, b8, _, mk8, bits8 = e8.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, pb, False, f8q=(1 / s).contiguous(), f8amax=amax)
    torch.cuda.synchronize()
    ref_a, ref_b = a2.float(), bt.float()
    da, db = a8.float() * s[0], b8.float() * s[1]
    qa = (ref_a / s[0]).to(torch.float8_e4m3fn).float() * s[0]   # torch's cast of the bf16 values (the kernel casts the fp32 accumulator)
    print(f"cm{cm} cz{cz} L{L} S{S} pb{pbias}: scale {s.tolist()} amax(scaled) {amax.tolist()} true max {ref_a.abs().max().item():.3f} {ref_b.abs().max().item():.3f}")
    print("  mask16 equal", torch.equal(mk16, mk8), "bits equal", torch.equal(bits, bits8))
    rel = lambda x, y: ((x - y).norm() / y.norm()).item()
    print(f"  A rel {rel(da, ref_a):.4e} (torch cast of bf16 {rel(qa, ref_a):.4e}), B rel {rel(db, ref_b):.4e}; |kernel - torchcast| max {(da - qa).abs().max().item():.3e}")
    print(f"  zeros where masked: A {(da[ref_a == 0] == 0).float().mean().item():.4f}")
    O16 = (a2.float() @ bt.float().t())
    for fast in (False, True):
        F.FAST_ACCUM = fast
        O8 = F.product(a8, b8, s, L)
        O8r = (da @ db.t())
        torch.cuda.synchronize()
        print(f"  fast={fast}: O8 vs fp32 of dequant {rel(O8.float(), O8r):.4e}; O8 vs bf16-operand product {rel(O8.float(), O16):.4e}")
print("ok")
