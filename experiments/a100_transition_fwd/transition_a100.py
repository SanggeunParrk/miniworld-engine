"""A100 (sm_80) pair Transition forward, one CUDA kernel: out = x + Ws (silu(Wa LN(x)) * (Wb LN(x))), D = 128, n = 4.

    ext = build()
    pk = pack(module)                 # miniworld_engine.modules.Transition(128, n=4)
    out = forward(ext, x, pk)         # x [..., 128] bf16
"""
import hashlib
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "a100_trimul_fwd" / "csrc"          # sm80_common.cuh (PTX helpers shared with the TriMul kernels)
_EXT = {}
_EMPTY = {}
D, H = 128, 512


def build(verbose=False, extra=()):
    key = tuple(extra)
    if key in _EXT:
        return _EXT[key]
    from torch.utils.cpp_extension import load
    srcs = sorted((HERE / "csrc").glob("*.cu*")) + [COMMON / "sm80_common.cuh"]
    src_hash = hashlib.sha1(b"".join(f.read_bytes() for f in srcs) + repr(key).encode()).hexdigest()[:12]
    name = f"a100_transition_fwd_{src_hash}"
    bdir = Path(os.environ.get("A100_TRANSITION_BUILD", Path.home() / ".cache/miniworld-a100/ext")) / name
    bdir.mkdir(parents=True, exist_ok=True)
    _EXT[key] = load(name=name, sources=[str(HERE / "csrc/ops.cu")], build_directory=str(bdir), verbose=verbose or bool(os.environ.get("A100_VERBOSE")),
                     extra_include_paths=[str(COMMON)],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-lineinfo", "-Xptxas=-v", *extra],
                     extra_cflags=["-O3"])
    return _EXT[key]


def k_perm(dev):
    """Physical GEMM1 k index 16 s + kk -> logical input column (thread q of a quad then holds the 16 B vectors at 32 i + 8 q)."""
    s = torch.arange(8, device=dev).view(-1, 1)
    kk = torch.arange(16, device=dev).view(1, -1)
    return (32 * (s // 2) + 8 * ((kk % 8) // 2) + 4 * (s % 2) + 2 * (kk // 8) + kk % 2).reshape(-1)


def o_perm(dev):
    """Physical GEMM2 output column 8 J + 2 q + e -> logical output column 32 (J / 4) + 8 q + 2 (J % 4) + e."""
    pcol = torch.arange(D, device=dev)
    J, q, e = pcol // 8, (pcol % 8) // 2, pcol % 2
    return 32 * (J // 4) + 8 * q + 2 * (J % 4) + e


@torch.no_grad()
def pack(m, ch=32):
    """Weights of a Transition(128, n=4) module in the kernel's layout: [16 chunks][W1 16 KB bf16 | W2 8 KB f16 bits]."""
    f = lambda t: t.detach().float()  # noqa: E731
    wa, wb, ws = f(m.expand_a.weight), f(m.expand_b.weight), f(m.squeeze.weight)   # [H, D], [H, D], [D, H]
    assert wa.shape == (H, D) and ws.shape == (D, H)
    dev = wa.device
    kp = k_perm(dev)
    wa_p, wb_p = (0.5 * wa)[:, kp], wb[:, kp]                              # 0.5: silu(a) b = a' b (1 + tanh a'), exact in bf16
    ws_p = ws[o_perm(dev)]                                                 # [D phys out][H]
    CH, nstep = ch, ch // 16
    nchunk = H // CH
    # W1 rows of chunk c: r = 32 p + 16 ab + hh  <->  hidden CH c + 16 p + hh of Wa (ab = 0) | Wb (ab = 1)
    w1 = torch.stack([wa_p.view(nchunk, nstep, 16, D), wb_p.view(nchunk, nstep, 16, D)], 2).reshape(nchunk, 2 * CH, D)
    w1 = w1.view(nchunk, 2 * CH, 16, 8).transpose(1, 2).reshape(nchunk, -1)     # [chunk][16 k-granules][2 CH rows][8]
    w2 = ws_p.view(D, nchunk, CH // 8, 8).permute(1, 2, 0, 3).reshape(nchunk, -1)  # [chunk][CH / 8 hidden-granules][128 rows][8]
    w2 = w2.to(torch.float16).view(torch.bfloat16)                          # GEMM2 runs in f16
    w = torch.cat([w1.to(torch.bfloat16), w2], 1).contiguous().view(-1)
    # LN affine: float4 slot [gamma | beta][s][q] = columns 32 (s / 2) + 8 q + 4 (s % 2) .. + 3 (a quad reads 64 consecutive bytes)
    s_, q_ = torch.arange(8, device=dev).view(-1, 1), torch.arange(4, device=dev).view(1, -1)
    c0 = (32 * (s_ // 2) + 8 * q_ + 4 * (s_ % 2)).reshape(-1, 1) + torch.arange(4, device=dev).view(1, -1)
    gb = torch.stack([f(m.ln_in.weight)[c0], f(m.ln_in.bias)[c0]]).contiguous().view(-1)
    return dict(w=w, gb=gb, eps=float(m.ln_in.eps))


def chunk(extra):
    """Chunk size selected by the build flags (-DTR_CH=..); pack() must match."""
    for f in " ".join(extra).split():
        if f.startswith("-DTR_CH="):
            return int(f.split("=")[1])
    return 32


def forward(ext, x, pk, out=None, grid=0, trace=None, stats=None, xn=None):
    """stats: optional [M, 2] fp32 (mean, rstd), xn: optional [M, 128] bf16 LN(x), both written for the backward."""
    assert x.shape[-1] == D and x.dtype == torch.bfloat16
    xf = x.reshape(-1, D)
    if not xf.is_contiguous():
        xf = xf.contiguous()
    if out is None:
        out = torch.empty_like(xf)
    if trace is None:
        trace = _EMPTY.setdefault(x.device, torch.empty(0, dtype=torch.int64, device=x.device))
    if stats is None:
        stats = _EMPTY.setdefault((x.device, "f"), torch.empty(0, dtype=torch.float32, device=x.device))
    if xn is None:
        xn = _EMPTY.setdefault((x.device, "b"), torch.empty(0, dtype=torch.bfloat16, device=x.device))
    ext.fwd(xf, pk["w"], pk["gb"], out, pk["eps"], grid, trace, stats, xn)
    return out.view(x.shape)


def fixture(L_or_M, device="cuda"):
    """The baseline fixture (../a100_anthropic_baseline): Transition(128, n=4) with trained-looking weights (the module zero-initialises
    the squeeze weight, which would make GEMM2 multiply zeros -- less power, a higher clock and a flattering time), bf16 input."""
    from miniworld_engine.modules.exceptions import ImplementationType as I
    from miniworld_engine.modules.transition import Transition
    mod = Transition(128, n=4, implementation=I.PYTORCH).to(device).bfloat16().eval()
    torch.manual_seed(1234)
    with torch.no_grad():
        for n, t in mod.named_parameters():
            if t.ndim >= 2:
                t.normal_(std=t.shape[-1] ** -.5)
            elif "weight" in n:
                t.copy_(1 + .1 * torch.randn_like(t))
            else:
                t.normal_(std=.05)
    torch.manual_seed(90323)
    M = L_or_M * L_or_M if L_or_M > 0 else -L_or_M
    x = torch.randn(M, 128, device=device, dtype=torch.bfloat16)
    return mod, x


def sol_us(M, bw=1.602e12, tc=240e12):
    """max(essential bytes / BW, FLOP / tensor): x in + out, 6 M D H FLOP; ceilings measured on this card (a100_trimul_fwd/records/peaks-gpu08.json)."""
    mem, flop = M * 2 * D * 2 / bw, M * 6 * D * H / tc
    return dict(mem=mem * 1e6, tensor=flop * 1e6, total=max(mem, flop) * 1e6)
