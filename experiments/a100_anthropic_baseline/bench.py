"""A100 baseline: Anthropic's published kernels (opt_core @ f4f62fa6) against the same MiniWorld module fixtures.

One (module, length, arm) per process, same measurement as verdicts/version-compare-20260923: CUDA graph replay,
~300 ms warm replays, median of 7 x 50 replays.

Arms
  pytorch / cuequiv / engine   the MiniWorld module (implementation pytorch / cuequivariance / miniworld), static
                               torch.compile + graph, inference and training (fwd+bwd, all input/param grads, dropout RNG)
  anth:<row>                   Anthropic's forward on the same weights, eager call captured in one graph, inference only
                               (the release has no backward for any of these ops)

Every inference output is compared against the fp32 PyTorch module on the same weights (rel RMS).
"""
import argparse
import collections
import copy
import json
import os
import statistics
import traceback
from pathlib import Path

import torch
from torch import nn

from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
from miniworld_engine.modules.transition import Transition

REVISION = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
p = argparse.ArgumentParser()
p.add_argument("--module", required=True, choices=["single", "trimul", "transition", "block", "opm", "pwa"])
p.add_argument("--length", type=int, required=True)
p.add_argument("--arm", required=True)
p.add_argument("--msa-depth", type=int, default=1024)
p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
p.add_argument("--no-training", action="store_true")
args = p.parse_args()
L, S = args.length, args.msa_depth
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
dest = out_dir / f"{args.module}-{L}-{args.arm.replace(':', '_')}.json"
record = dict(module=args.module, L=L, arm=args.arm, gpu=torch.cuda.get_device_name(),
              uuid=str(torch.cuda.get_device_properties(0).uuid), host=os.uname().nodename, job=os.getenv("SLURM_JOB_ID"),
              torch=torch.__version__, cuda=torch.version.cuda, upstream_revision=REVISION, dtype="bf16",
              msa_depth=S if args.module in ("opm", "pwa") else None, modes={})
anthropic = args.arm.startswith("anth:")
row = args.arm.split(":", 1)[1] if anthropic else None
record["timing"] = ("eager Anthropic call captured in one CUDA graph" if anthropic
                    else "static torch.compile(fullgraph) captured in one CUDA graph")


def save():
    dest.write_text(json.dumps(record, indent=2))


class Block(nn.Module):
    """MiniPairformer block as in the H100 table: bidirectional TriMul -> pair Transition."""

    def __init__(self, impl):
        super().__init__()
        self.trimul = BidirectionalTriangleMultiplication(128, implementation=impl, p_drop=.25)
        self.transition = Transition(128, n=4, implementation=I.PYTORCH if impl == I.CUEQUIVARIANCE else impl)

    def forward(self, x, mask):
        return self.transition(self.trimul(x, mask))


class OPMResidual(nn.Module):
    def __init__(self, impl):
        super().__init__()
        from miniworld_engine.modules.outer_product import OuterProductMean
        self.opm = OuterProductMean(64, 128, 32, implementation=impl)

    def forward(self, msa, mask, pair):
        return self.opm(msa, mask, residual=pair)


def build(impl):
    return {"single": lambda: TriangleMultiplication(128, implementation=impl, p_drop=.25),
            "trimul": lambda: BidirectionalTriangleMultiplication(128, implementation=impl, p_drop=.25),
            "transition": lambda: Transition(128, n=4, implementation=impl),
            "block": lambda: Block(impl),
            "opm": lambda: OPMResidual(impl),
            "pwa": lambda: __import__("miniworld_engine.modules.msa_pair_weighted_averaging", fromlist=["x"])
            .MSAPairWeightedAveraging(64, 128, 8, 32, implementation=impl, p_drop=.15)}[args.module]()


def init_(m):
    torch.manual_seed(1234)
    with torch.no_grad():
        for n, t in m.named_parameters():
            if t.ndim >= 2:
                t.normal_(std=t.shape[-1] ** -.5)
            elif "weight" in n:
                t.copy_(1 + .1 * torch.randn_like(t))
            else:
                t.normal_(std=.05)
    return m


def inputs_(grad):
    torch.manual_seed(90323)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    r = lambda *s: torch.randn(*s, **kw).requires_grad_(grad)  # noqa: E731
    mask = torch.rand(1, L, device="cuda") > .1
    if args.module == "transition":
        return (r(1, L, L, 128),)
    if args.module in ("single", "trimul", "block"):
        return (r(1, L, L, 128), mask)
    if args.module == "opm":
        return (r(1, S, L, 64), torch.rand(1, S, L, device="cuda") > .1, r(1, L, L, 128))
    return (r(1, S, L, 64), r(1, L, L, 128), mask)


def rel_rms(y, ref):
    d = y.float() - ref.float()
    return float(d.square().mean().sqrt() / ref.float().square().mean().sqrt())


def time_graph(step, stream):
    """Capture step() once, then the version-compare replay protocol. Returns (median ms, rounds, warm replays)."""
    with torch.cuda.stream(stream):
        for _ in range(6):
            step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            res = step()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        graph.replay()
    b.record()
    b.synchronize()
    est = a.elapsed_time(b) / 10
    warm = min(10000, max(30, int(300 / max(est, .001))))
    for _ in range(warm):
        graph.replay()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(50):
            graph.replay()
        b.record()
        b.synchronize()
        rounds.append(a.elapsed_time(b) / 50)
    return statistics.median(rounds), rounds, warm, graph, res


def kernels_of(graph):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            a = agg[e.name[:160]]
            a[0] += 1
            a[1] += e.device_time_total
    # {kernel: [launches, us]} for one replay, longest first
    return {k: [n, round(us, 1)] for k, (n, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])}


# ---------------------------------------------------------------------------------------------------- Anthropic forwards
def _trimul_weights(m):
    return {"ln_in_w": m.ln_pair.weight, "ln_in_b": m.ln_pair.bias, "w_ag": m.to_left_gate.weight, "w_ap": m.to_left.weight,
            "w_bg": m.to_right_gate.weight, "w_bp": m.to_right.weight, "ln_out_w": m.ln_out.weight, "ln_out_b": m.ln_out.bias,
            "w_o": m.to_out.weight, "w_og": m.to_gate.weight}


def _pairmask(mask):
    return (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(torch.float32).contiguous()


def anth_single(m, word):
    from opt_core.kernels import trimul as T
    w = {k: v.detach().contiguous() for k, v in _trimul_weights(m).items()}
    cache = {}

    def fn(x, mask):
        y = T.triangle_multiplication(x, _pairmask(mask), direction="outgoing", weights=w, word=word, residual=True,
                                      cache=cache, eps=m.ln_pair.eps)
        record["selection"] = str(cache.get("_last"))
        return y
    return fn


NATIVE_REBUILT = Path(os.environ.get("TRIMUL_NATIVE_REBUILT",
                                     Path.home() / ".cache/miniworld-a100/native_rebuilt/v5"))


def _native_sm80():
    """The release's trimul_native v5 tree with only build/sm_80 recompiled by its own build.py under nvcc 12.9: the shipped
    sm_80 cubins are CUDA 13.0 images and this node's driver (570, CUDA 12.8) refuses them (CUDA_ERROR_INVALID_IMAGE).
    The package's own byte gate (sm80 witness vectors, bitwise) runs before any timing."""
    import importlib
    import sys
    py = str(NATIVE_REBUILT / "python")
    if py not in sys.path:
        sys.path.insert(0, py)
    face = importlib.import_module("trimul_native.face")
    if "native_gate" not in record:
        # the vector manifest pins the release cubin digests; ignore_build skips only that pin, every case is still compared bitwise
        face.check(device=torch.cuda.current_device(), gate=False)
        V = importlib.import_module("trimul_native.vectors")
        rep = V.replay(which="all", device_index=torch.cuda.current_device(), raise_on_fail=False, quiet=True, ignore_build=True)
        record["native_gate"] = {k: rep.get(k) for k in ("n", "passed", "failed", "skipped")}
        print("NATIVE_GATE", record["native_gate"], flush=True)
    return importlib.import_module("trimul_native.sm80_ops"), importlib.import_module("trimul_native.launch")


def anth_single_native(m):
    so, _ = _native_sm80()
    w = {k: v.detach().contiguous() for k, v in _trimul_weights(m).items()}
    cache = {}

    def fn(x, mask):
        y = so.serve_sm80(x, _pairmask(mask), direction="outgoing", weights=w, residual=True, cache=cache, eps=m.ln_pair.eps)
        record["selection"] = cache.get("_describe")
        return y
    return fn


def anth_bidir_native(m):
    """Bidirectional TriMul from the release's sm_80 native member, composed the way integrations/anthropic_trimul.py
    composes the sm_90 unit: ONE K1 over 2*c_hidden (shared input LN, planes in token order) -> outgoing half NT bmm,
    incoming half TN bmm -> ONE K3 over 2*c_hidden (shared output LN, gate, residual). Only the kernel-call glue is ours."""
    so, L_ = _native_sm80()
    w = {k: v.detach().contiguous() for k, v in _trimul_weights(m).items()}
    eps = m.ln_pair.eps
    dev = torch.cuda.current_device()
    pk = so.pack_weights(w, torch.device("cuda"))
    C, D, h = pk["C"], pk["D"], m.d_hidden
    k1n, k3n = so._tiles_for(C, D, "bf16", {})
    k1n, k3n = k1n.replace("k1z_", "k1_", 1), k3n.replace("k3p_", "k3_", 1)
    k1, k3 = so._kernel(k1n, dev), so._kernel(k3n, dev)
    g1, g3 = k1.geo, k3.geo
    Np = (L + so.PAD - 1) // so.PAD * so.PAD
    ab = torch.empty((2 * D, Np, Np), dtype=torch.bfloat16, device="cuda")
    xbuf = torch.empty((D, Np, Np), dtype=torch.bfloat16, device="cuda")
    record["selection"] = f"sm80 native composed: {k1n} | bmm NT(out half) + TN(in half) | {k3n}"
    state = {}

    def fn(x, mask):
        z4 = x.contiguous()
        pm = _pairmask(mask)
        out = torch.empty_like(z4)
        if "packs" not in state:
            p1 = L_.Struct([z4, pm, ab, pk["wg"], pk["wp"], pk["g_in"], pk["b_in"], L_.u64(0), L_.i64(L * L * C), L_.i64(0),
                            L_.i64(Np * Np), L_.i32(L), L_.i32(Np), L_.i32(1), L_.i32(D), L_.i32(1), L_.f32(eps)])
            p3 = L_.Struct([xbuf, z4, out, pk["wo"], pk["wog"], pk["g_out"], pk["b_out"], pk["g_in"], pk["b_in"], L_.u64(0),
                            L_.i64(L * L * C), L_.i64(Np * Np), L_.i32(L), L_.i32(Np), L_.i32(1), L_.i32(D), L_.i32(1),
                            L_.i32(0), L_.f32(eps)])
            state["packs"] = (k1.argpack([p1]), k3.argpack([p3]))
        a1, a3 = state["packs"]
        a1.set_ptr(0, z4, field=0); a1.set_ptr(0, pm, field=1); a1.set_ptr(0, ab, field=2)
        a3.set_ptr(0, xbuf, field=0); a3.set_ptr(0, z4, field=1); a3.set_ptr(0, out, field=2)
        so._launch(k1, ((Np + g1["BM"] - 1) // g1["BM"], Np, 1), (g1["threads"], 1, 1), a1, g1["smem"])
        a, b = ab[:D], ab[D:]
        torch.bmm(a[:h], b[:h].transpose(1, 2), out=xbuf[:h])      # outgoing: sum_k a[i,k] b[j,k]
        torch.bmm(a[h:].transpose(1, 2), b[h:], out=xbuf[h:])      # incoming: sum_k a[k,i] b[k,j]
        so._launch(k3, ((L + g3["BM"] - 1) // g3["BM"], L, 1), (g3["threads"], 1, 1), a3, g3["smem"])
        return out
    return fn


def anth_transition(m, word):
    from opt_core.kernels import transition as TR
    W = TR.pack(w_a=m.expand_a.weight, w_b=m.expand_b.weight, w_o=m.squeeze.weight, ln_w=m.ln_in.weight,
                ln_b=m.ln_in.bias, eps=m.ln_in.eps)

    def fn(x):
        y, sel = TR.transition(x, W, word=word, residual=True, n_tokens=L)
        record["selection"] = str(sel)
        return y
    return fn


class _OpmView:
    def __init__(self, m):
        self.norm, self.proj_a, self.proj_b, self.proj_o = m.ln_msa, m.to_left, m.to_right, m.to_out


class _PwaView:
    """Same weights under the upstream attribute names (as integrations/anthropic_msa.py, but with the module's own
    ln_pair so the whole forward is upstream's statement)."""

    def __init__(self, m):
        self.norm_m, self.proj_m, self.proj_g = m.ln_msa, m.to_value, m.to_gate
        self.norm_z, self.proj_z, self.proj_o = m.ln_pair, m.to_bias, m.to_out
        self.inf, self.num_heads = 1e9, m.n_head
        self.c_h = m.to_value.weight.shape[0] // m.n_head


def anth_opm(m, word):
    from opt_core.ops import msa_opm
    if word != "default":
        os.environ["FPF_OPM_CFG"] = word
    view = _OpmView(m.opm)
    record["selection"] = str(msa_opm.cfg_for("mask_norm", 128, torch.device("cuda")))

    def fn(msa, mask, pair):
        return pair + msa_opm.forward_mask_norm(view, msa, mask.to(torch.bfloat16))
    return fn


def anth_pwa(m, word):
    if word != "default":
        os.environ["FPF_PWA_CFG"] = word
    from opt_core.ops import msa_pwa
    view = _PwaView(m)
    record["selection"] = os.environ.get("FPF_PWA_CFG", "module default _CFG")

    def fn(msa, pair, mask):
        pm = mask[:, None, :].expand(-1, L, -1).to(torch.bfloat16)
        return msa + msa_pwa.forward_masked(view, msa, pair, pm, chunk_heads=False)
    return fn


def anth_block(m, word):
    tri = anth_bidir_native(m.trimul)
    tr = anth_transition(m.transition, word)
    return lambda x, mask: tr(tri(x, mask))


def anthropic_fn(m):
    mod = args.module
    if mod == "single":
        return anth_single_native(m) if row == "native_rebuilt" else anth_single(m, row)
    if mod == "trimul":
        if row != "native_bidir":
            raise ValueError("bidirectional TriMul has one Anthropic composition: anth:native_bidir")
        return anth_bidir_native(m)
    if mod == "transition":
        return anth_transition(m, row)
    if mod == "block":
        return anth_block(m, row)
    if mod == "opm":
        return anth_opm(m, row)
    return anth_pwa(m, row)


# ---------------------------------------------------------------------------------------------------- run
try:
    base = init_(build(I.PYTORCH).cuda().bfloat16()).eval()
    ref_m = copy.deepcopy(base).float().eval()
    with torch.no_grad():
        xin = inputs_(False)
        ref = ref_m(*[t.float() if t.is_floating_point() else t for t in xin])
    del ref_m
    torch.cuda.empty_cache()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    if anthropic:
        result = {}
        record["modes"]["inference"] = result
        try:
            fn = anthropic_fn(base)
            with torch.no_grad():
                y = fn(*xin)
                result["rel_rms_vs_fp32"] = rel_rms(y, ref)
                result["finite"] = bool(torch.isfinite(y).all())
                ms, rounds, warm, graph, _ = time_graph(lambda: fn(*xin), side)
            result.update(ms=ms, rounds_ms=rounds, warm_replays=warm, kernels=kernels_of(graph))
            print("RESULT", args.module, L, args.arm, "inference", ms, result["rel_rms_vs_fp32"], flush=True)
        except Exception:
            result["error"] = traceback.format_exc()
            print(result["error"], flush=True)
        save()
    else:
        impl = {"pytorch": I.PYTORCH, "cuequiv": I.CUEQUIVARIANCE, "engine": I.MINIWORLD}[args.arm]
        if impl == I.CUEQUIVARIANCE:
            import cuequivariance_ops_torch  # noqa: F401
        m = build(impl).cuda().bfloat16()
        m.load_state_dict(base.state_dict())
        record["resolved_backends"] = {n: str(v._backend) for n, v in m.named_modules() if hasattr(v, "_backend")}
        for mode in (("inference",) if args.no_training else ("inference", "training")):
            result = {}
            record["modes"][mode] = result
            save()
            try:
                torch.compiler.reset()
                train = mode == "training"
                m.train(train)
                inputs = inputs_(train)
                compiled = torch.compile(m, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
                with torch.set_grad_enabled(train), torch.cuda.stream(side):
                    y = compiled(*inputs)
                    if not train:
                        result["rel_rms_vs_fp32"] = rel_rms(y, ref)
                    dy = torch.randn_like(y)
                    del y
                    leaves = tuple(t for t in inputs if t.requires_grad) + tuple(m.parameters())

                    def step(compiled=compiled, leaves=leaves, dy=dy, train=train, inputs=inputs):
                        o = compiled(*inputs)
                        return (o, torch.autograd.grad(o, leaves, dy)) if train else (o, ())
                with torch.set_grad_enabled(train):
                    ms, rounds, warm, graph, (o, grads) = time_graph(step, side)
                result["finite"] = bool(o.isfinite().all()) and all(bool(g.isfinite().all()) for g in grads)
                result.update(ms=ms, rounds_ms=rounds, warm_replays=warm, kernels=kernels_of(graph))
                print("RESULT", args.module, L, args.arm, mode, ms, result.get("rel_rms_vs_fp32"), flush=True)
                del graph, compiled, o, grads, dy, step
            except Exception:
                result["error"] = traceback.format_exc()
                print(result["error"], flush=True)
            save()
except Exception:
    record["error"] = traceback.format_exc()
    print(record["error"], flush=True)
save()
